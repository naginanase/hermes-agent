"""Portable, honest Slack display fallback for unresolved provider citations.

The original response and identifiers remain in persisted provider sidecars.
This module changes delivery text only; it never invents a ref-to-URL mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from agent.markdown_code_regions import markdown_code_parts

_UNRESOLVED = re.compile(
    r"\ue200cite\ue202(?P<native>[^\ue201]+)\ue201"
    r"|\[unresolved citation: (?P<diagnostic>[^\]\n]+)\]"
)
_PARTIAL = re.compile(
    r"\ue200cite\ue202[^\ue201]*\Z|\[unresolved citation: [^\]\n]*\Z"
)


@dataclass(frozen=True)
class CitationDelivery:
    text: str
    unresolved_count: int = 0
    diagnostic_refs: tuple[str, ...] = ()
    source_count: int = 0


class SlackCitationPreviewBuffer:
    """Stop the preview at a citation until the authoritative, source-aware final.

    Only a possible marker prefix is held in memory. The provider response keeps
    the full original text; buffering never changes persistence or model inputs.
    """
    _PREFIXES = ("\ue200cite\ue202", "[unresolved citation:")

    def __init__(self) -> None:
        self._pending = ""
        self._blocked = False

    def feed(self, delta: str) -> str:
        if self._blocked:
            return ""
        text = self._pending + delta
        self._pending = ""
        starts = [text.find(prefix) for prefix in self._PREFIXES if prefix in text]
        if starts:
            self._blocked = True
            return text[:min(starts)]
        for length in range(min(len(text), max(map(len, self._PREFIXES))), 0, -1):
            if any(prefix.startswith(text[-length:]) for prefix in self._PREFIXES):
                self._pending = text[-length:]
                return text[:-length]
        return text


def _retrieved_urls(messages: Sequence[dict[str, Any]], history_offset: int) -> list[str]:
    """Only typed retrieval sidecars from this turn are provenance, never prose URLs."""
    rows = list(messages)
    start = max(0, min(history_offset, len(rows)))
    # A prior turn can reuse turn0view0 for another page. Never carry its sources forward.
    # The current transcript's user boundary is authoritative: compaction/splitting
    # can invalidate the caller's pre-run history length before session sync runs.
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("role") == "user":
            start = index + 1
    urls: list[str] = []
    from agent.retrieval_sources import extract_retrieval_sources
    for row in rows[start:]:
        if not isinstance(row, dict) or row.get("role") != "assistant":
            continue
        for source in extract_retrieval_sources(row):
            url = source.get("url")
            if not isinstance(url, str) or len(url) > 4096 or re.search(r"[\s\x00-\x1f\x7f]", url):
                continue
            try:
                parsed = urlsplit(url)
                if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                        or parsed.username is not None or parsed.password is not None
                        or re.search(r"[<>|]", parsed.netloc)):
                    continue
                _ = parsed.port  # malformed ports are not source links
            except ValueError:
                continue
            if url not in urls:
                urls.append(url)
    return urls


def slack_citation_draft_body(text: str) -> str | None:
    """Match a raw native draft to a repaired final, without inventing source mappings.

    A draft can end inside a marker. Only prose is normalized; code examples
    remain literal, and the source-unavailable footer is not part of the body.
    """
    if not (_UNRESOLVED.search(text) or _PARTIAL.search(text)):
        return None
    regions = list(markdown_code_parts(text))
    parts = [part if code else _UNRESOLVED.sub("※", part) for code, part in regions]
    if regions and not regions[-1][0]:
        parts[-1] = _PARTIAL.sub("※", parts[-1])
    body = "".join(parts)
    return body if body != text else None


def prepare_slack_citation_delivery(
    text: str, *, messages: Sequence[dict[str, Any]] = (), history_offset: int = 0,
) -> CitationDelivery:
    """Keep uncertainty visible without exposing renderer-only IDs in prose."""
    if not _UNRESOLVED.search(text):
        return CitationDelivery(text)
    refs: list[str] = []

    def replace(match: re.Match) -> str:
        refs.append(match.group("native") or match.group("diagnostic") or "")
        return "※"

    regions = list(markdown_code_parts(text))
    parts = [part if code else _UNRESOLVED.sub(replace, part)
             for code, part in regions]
    if not refs:
        return CitationDelivery(text)
    # Footer spacing may trim prose, but never original bytes in a code block.
    if regions and not regions[-1][0]:
        parts[-1] = parts[-1].rstrip()
    shown = "".join(parts)
    urls = _retrieved_urls(messages, history_offset)
    japanese = bool(re.search(r"[\u3040-\u30ff]", text))
    if urls:
        warning = ("※ 一部の引用と出典の対応を確認できません。取得した出典：" if japanese else
                   "※ Some citation-to-source associations could not be verified. Retrieved sources:")
        destinations = [url.translate(str.maketrans({
            "(": "%28", ")": "%29", "<": "%3C", ">": "%3E", "\\": "%5C", "|": "%7C",
        })) for url in urls]
        warning += "\n" + "\n".join(
            f"- [{'出典' if japanese else 'Source'} {index}]({url})"
            for index, url in enumerate(destinations, 1)
        )
    else:
        warning = ("※ 出典URLを取得できず、一部の引用を確認できません。" if japanese else
                   "※ Source URLs are unavailable; some citations could not be verified.")
    return CitationDelivery(shown + "\n\n" + warning, len(refs), tuple(refs), len(urls))
