"""Verified native retrieval metadata, independent of opaque citation display refs.

``extract_retrieval_sources(response_or_messages)`` returns JSON-safe source records.
A search source list proves retrieval, NOT which opaque marker cites which URL.
The internal registry travels in the existing codex_message_items sidecar; the
Responses replay converter deliberately ignores its non-message type.

Carrier schema: {type: "hermes_retrieval_sources", version: 1,
request_id?: str, response_id?: str, sources: [source, ...]}.
Sources contain url, optional title, provenance ("url_citation" or
"web_search_call.action.sources"), and available request_id/response_id/item_id/
output_index. Annotations additionally retain content_index, annotation_index,
start_index/end_index and location_verified. That flag only verifies the
received range against retained raw item text, not an opaque-ref mapping or
offsets into rendered text. Stale/unattached metadata remains a source with
location_verified=False. No prompts, search queries or arbitrary provider
fields are copied. A local request token is used when no actual request ID is
available; absent provider/item identities stay absent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

SOURCE_REGISTRY_TYPE = "hermes_retrieval_sources"
logger = logging.getLogger(__name__)


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _identity(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_url(value: Any) -> bool:
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not (
            parsed.username is not None or parsed.password is not None)
    except ValueError:
        return False


def _source(value: Any, **scope) -> dict | None:
    url = _field(value, "url")
    if not _safe_url(url):
        return None
    result = {"url": url, **{key: val for key, val in scope.items() if val is not None}}
    title = _field(value, "title")
    if isinstance(title, str) and title:
        result["title"] = title
    return result


def source_from_annotation(annotation: Any, *, request_id=None, response_id=None, item_id=None,
                           output_index=None, content_index=None, annotation_index=None,
                           text=None, location_verified=False) -> dict | None:
    """A URL annotation's actual scope, with no assumed opaque-ref correspondence."""
    if _field(annotation, "type") != "url_citation":
        return None
    scope: dict[str, Any] = dict(request_id=_identity(request_id), response_id=_identity(response_id), item_id=_identity(item_id),
                 output_index=output_index, content_index=content_index, annotation_index=annotation_index)
    for name in ("start_index", "end_index"):
        offset = _field(annotation, name)
        if type(offset) is int:
            scope[name] = offset
    start, end = scope.get("start_index"), scope.get("end_index")
    scope["location_verified"] = bool(location_verified and isinstance(text, str)
        and type(start) is int and type(end) is int and 0 <= start <= end <= len(text))
    return _source(annotation, provenance="url_citation", **scope)


def sources_from_items(items: Any, *, request_id=None, response_id=None, output_indexes=None,
                       location_verified=True) -> list[dict]:
    """Read only documented native URL metadata fields; never queries or ref guesses."""
    sources = []
    for index, item in enumerate(_list(items)):
        scope = dict(request_id=_identity(request_id), response_id=_identity(response_id),
                     item_id=_identity(_field(item, "id")),
                     output_index=output_indexes[index] if output_indexes is not None else _field(item, "_output_index", index))
        if _field(item, "type") == "web_search_call":
            for source in _list(_field(_field(item, "action"), "sources")):
                if _field(source, "type") not in (None, "url"):
                    continue
                record = _source(source, provenance="web_search_call.action.sources", **scope)
                if record is not None:
                    sources.append(record)
        elif _field(item, "type") == "message":
            for content_index, part in enumerate(_list(_field(item, "content"))):
                for annotation_index, annotation in enumerate(_list(_field(part, "annotations"))):
                    record = source_from_annotation(annotation, **scope, content_index=content_index,
                                                    annotation_index=annotation_index, text=_field(part, "text"),
                                                    location_verified=location_verified)
                    if record is not None:
                        sources.append(record)
    return sources


def source_registry(sources: list[dict], *, request_id=None, response_id=None) -> dict | None:
    """One request-local, JSON-safe carrier (None when no verified metadata exists)."""
    result, seen = [], {}
    for source in sources:
        provenance = _field(source, "provenance")
        if not isinstance(provenance, str) or provenance not in {"url_citation", "web_search_call.action.sources"}:
            continue
        scope: dict[str, Any] = {name: _identity(_field(source, name)) for name in ("request_id", "response_id", "item_id")}
        for name in ("output_index", "content_index", "annotation_index", "start_index", "end_index"):
            value = _field(source, name)
            if type(value) is int:
                scope[name] = value
        record = _source(source, provenance=provenance, **scope)
        if record is None:
            continue
        if provenance == "url_citation":
            record["location_verified"] = _field(source, "location_verified") is True
        if _identity(request_id):
            record["request_id"] = request_id
        if _identity(response_id):
            record["response_id"] = response_id
        identity = {name: value for name, value in record.items() if name != "location_verified"
                    and not (name == "output_index" and record.get("item_id"))}
        key = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen[key] = record
            result.append(record)
        elif record.get("location_verified"):
            seen[key]["location_verified"] = True
        if "output_index" not in seen[key] and "output_index" in record:
            seen[key]["output_index"] = record["output_index"]
    if not result:
        return None
    registry = dict(type=SOURCE_REGISTRY_TYPE, version=1, sources=result)
    for name in ("request_id", "response_id"):
        identities = {record[name] for record in result if record.get(name)}
        if len(identities) == 1:
            registry[name] = identities.pop()
    return registry


def extract_retrieval_sources(value: Any) -> list[dict]:
    """Extract verified native sources from a raw/normalized response or messages.

    No ordinary Markdown links, tool-result prose, or opaque ref IDs are treated
    as retrieval metadata. Callers should pass only the current response/turn,
    or filter by the request_id/response_id retained in each record.
    """
    if isinstance(value, list):
        sources = [source for entry in value for source in extract_retrieval_sources(entry)]
    else:
        registry = _field(value, "retrieval_source_registry")
        sidecar = _field(value, "codex_message_items")
        if sidecar is None:
            sidecar = _field(_field(value, "provider_data"), "codex_message_items")
        carriers = [registry] if _field(registry, "type") == SOURCE_REGISTRY_TYPE else [
            item for item in _list(sidecar) if _field(item, "type") == SOURCE_REGISTRY_TYPE]
        if carriers:
            sources = [source for carrier in carriers for source in _list(_field(carrier, "sources"))]
        else:
            sources = sources_from_items(_field(value, "output") or sidecar,
                                         response_id=_field(value, "id"))
    registry = source_registry(sources)
    return registry["sources"] if registry else []


def retrieval_source_counts(value: Any) -> dict:
    """Safe boundary evidence; metadata_absent never asserts what the provider sent."""
    sources = extract_retrieval_sources(value)
    texts = []
    output = _field(value, "output")
    sidecar = _field(value, "codex_message_items") or _field(_field(value, "provider_data"), "codex_message_items")
    for item in _list(output if isinstance(output, list) else sidecar):
        if _field(item, "type") == "message":
            texts.extend(_field(part, "text", "") for part in _list(_field(item, "content"))
                         if isinstance(_field(part, "text"), str))
    if not texts:
        fallback = _field(value, "output_text") or _field(value, "content")
        if isinstance(fallback, str):
            texts.append(fallback)
    content = _field(value, "content", "")
    return dict(source_count=len(sources),
                search_source_count=sum(source["provenance"] == "web_search_call.action.sources" for source in sources),
                url_citation_count=sum(source["provenance"] == "url_citation" for source in sources),
                raw_citation_marker_count=sum(len(re.findall(r"\ue200cite\ue202[^\ue201]*\ue201", text)) for text in texts),
                unresolved_citation_count=content.count("[unresolved citation:") if isinstance(content, str) else 0,
                metadata_status="metadata_present" if sources else "metadata_absent")


def log_retrieval_source_counts(boundary: str, value: Any) -> None:
    """DEBUG opt-in counts only: never URLs, IDs, titles, prompts, credentials or SSE."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if boundary not in {"receive.item", "receive.terminal", "assembly", "normalize"}:
        return
    counts = retrieval_source_counts(value)
    logger.debug("Native retrieval boundary=%s %s", boundary,
                " ".join(f"{name}={count}" for name, count in counts.items()))
