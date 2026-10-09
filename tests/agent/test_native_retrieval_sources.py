"""Request-local native source metadata, not inferred opaque citation mappings."""
import json
from types import SimpleNamespace as NS

import pytest

from agent.codex_runtime import _consume_codex_event_stream
from agent.codex_responses_adapter import _chat_messages_to_responses_input, _preflight_codex_input_items
from agent.transports.codex import ResponsesApiTransport


MARKER = "\ue200cite\ue202turn7search0\ue201"
URL = "https://example.org/retrieved"


def search_item(item_id="ws_1", url=URL):
    return NS(type="web_search_call", id=item_id, status="completed", action=NS(
        type="search", query="private query", sources=[NS(type="url", url=url, title="Retrieved", secret="do not retain")]))


def answer(text="Claim " + MARKER, item_id="msg_1", annotations=()):
    return NS(type="message", id=item_id, role="assistant", status="completed", content=[
        NS(type="output_text", text=text, annotations=list(annotations))])


@pytest.mark.parametrize("boundary", ["done", "terminal"])
def test_retrieved_sources_survive_normalization_persistence_without_wire_carrier(boundary, tmp_path):
    item = search_item()
    events = [NS(type="response.output_item.done", output_index=1, item=answer())]
    if boundary == "done":
        events.insert(0, NS(type="response.output_item.done", output_index=0, item=item))
    events.append(NS(type="response.completed", response=NS(
        id="resp_1", status="completed", output=[item, answer()] if boundary == "terminal" else None)))
    response = _consume_codex_event_stream(events, model="test")
    normalized = ResponsesApiTransport().normalize_response(response)
    carriers = [item for item in normalized.codex_message_items if item.get("type") == "hermes_retrieval_sources"]
    assert len(carriers) == 1
    from agent.retrieval_sources import extract_retrieval_sources
    from hermes_state import SessionDB

    sources = extract_retrieval_sources(normalized)
    assert len(sources) == 1
    assert sources[0]["url"] == URL
    assert sources[0]["provenance"] == "web_search_call.action.sources"
    assert sources[0]["response_id"] == "resp_1"
    assert sources[0]["item_id"] == "ws_1"
    assert sources[0]["output_index"] == 0
    assert sources[0]["request_id"]
    assert not any("ref" in key for key in sources[0])
    assert "private query" not in json.dumps(carriers)
    assert "do not retain" not in json.dumps(carriers)
    assert normalized.content == "Claim [unresolved citation: turn7search0]"
    assert extract_retrieval_sources(response) == sources
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sources", source="cli")
        db.append_message("sources", "assistant", normalized.content,
                          codex_message_items=normalized.codex_message_items)
        history = db.get_messages_as_conversation("sources")
        assert extract_retrieval_sources(history) == sources
        wire = _preflight_codex_input_items(_chat_messages_to_responses_input(history))
        assert len(wire) == 1 and wire[0]["type"] == "message"
        assert "hermes_retrieval_sources" not in json.dumps(wire)
        assert URL not in json.dumps(wire)
    finally:
        db.close()


@pytest.mark.parametrize("boundary", ["done", "terminal", "annotation_event"])
def test_url_annotations_retain_part_scope_separately_from_search_sources(boundary):
    from agent.retrieval_sources import extract_retrieval_sources

    annotation = dict(type="url_citation", url="https://example.org/cited", title="Cited",
                      start_index=6, end_index=6 + len(MARKER))
    annotated = answer(annotations=[annotation])
    events = [NS(type="response.output_item.done", output_index=0, item=search_item())]
    if boundary == "annotation_event":
        events.extend([
            NS(type="response.output_text.delta", item_id="msg_1", output_index=1,
               content_index=0, delta="Claim " + MARKER),
            NS(type="response.output_text.annotation.added", item_id="msg_1", output_index=1,
               content_index=0, annotation_index=0, annotation=annotation),
        ])
    events.append(NS(type="response.output_item.done", output_index=1,
                     item=annotated if boundary == "done" else answer()))
    events.append(NS(type="response.completed", response=NS(id="resp_1", status="completed",
                   output=[search_item(), annotated] if boundary == "terminal" else None)))
    response = _consume_codex_event_stream(events, model="test")
    normalized = ResponsesApiTransport().normalize_response(response)
    sources = extract_retrieval_sources(normalized)
    assert len(sources) == 2
    citation = next(source for source in sources if source["provenance"] == "url_citation")
    assert citation["url"] == annotation["url"]
    assert citation["item_id"] == "msg_1"
    assert citation["output_index"] == 1
    assert citation["content_index"] == 0
    assert citation["annotation_index"] == 0
    assert citation["start_index"] == 6
    assert citation["end_index"] == 6 + len(MARKER)
    registry = normalized.codex_message_items[-1]
    assert registry["request_id"] == citation["request_id"]
    assert registry["response_id"] == "resp_1"
    assert extract_retrieval_sources(response) == sources
    assert normalized.content == "Claim [Cited](https://example.org/cited)"


def test_metadata_only_terminal_is_preserved_as_incomplete_without_fabricating_text():
    from agent.retrieval_sources import extract_retrieval_sources

    response = _consume_codex_event_stream([NS(type="response.completed", response=NS(
        id="resp_sources", status="completed", output=[search_item()]))], model="test")
    transport = ResponsesApiTransport()
    assert transport.validate_response(response)
    normalized = transport.normalize_response(response)
    assert normalized.content == ""
    assert normalized.finish_reason == "incomplete"
    assert extract_retrieval_sources(normalized)[0]["response_id"] == "resp_sources"
    history = [dict(role="assistant", content="", codex_message_items=normalized.codex_message_items)]
    assert _chat_messages_to_responses_input(history) == []


def test_explicit_request_identity_survives_even_when_provider_response_id_is_absent():
    from agent.retrieval_sources import extract_retrieval_sources

    events = [NS(type="response.output_item.done", output_index=4, item=search_item())]
    response = _consume_codex_event_stream(events, model="test", request_id="actual_request_1")
    sources = extract_retrieval_sources(response)
    assert sources[0]["request_id"] == "actual_request_1"
    assert "response_id" not in sources[0]
    assert sources[0]["output_index"] == 4
    other = _consume_codex_event_stream(events, model="test", request_id="actual_request_2")
    combined = extract_retrieval_sources([response, other])
    assert len(combined) == 2
    assert {source["request_id"] for source in combined} == {"actual_request_1", "actual_request_2"}


@pytest.mark.parametrize("has_sources", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_boundary_diagnostics_are_opt_in_counts_only_and_describe_retained_metadata(caplog, has_sources, enabled):
    import logging
    import agent.retrieval_sources as retrieval

    counts_fn = getattr(retrieval, "retrieval_source_counts", None)
    assert callable(counts_fn)
    level = logging.DEBUG if enabled else logging.INFO
    events = [NS(type="response.output_item.done", output_index=1, item=answer(
        text="private answer https://secret.example/?token=credential " + MARKER))]
    if has_sources:
        events.insert(0, NS(type="response.output_item.done", output_index=0, item=search_item()))
    events.append(NS(type="response.completed", response=NS(id="private_response_id", status="completed", output=None)))
    with caplog.at_level(level, logger="agent.retrieval_sources"):
        response = _consume_codex_event_stream(events, model="private_model")
        normalized = ResponsesApiTransport().normalize_response(response)
    counts = counts_fn(response)
    assert counts["source_count"] == int(has_sources)
    assert counts["search_source_count"] == int(has_sources)
    assert counts["url_citation_count"] == 0
    assert counts["raw_citation_marker_count"] == 1
    assert counts["metadata_status"] == ("metadata_present" if has_sources else "metadata_absent")
    assert counts_fn(normalized)["unresolved_citation_count"] == 1
    logs = [record.getMessage() for record in caplog.records if record.name == "agent.retrieval_sources"]
    if enabled:
        assert all(any("boundary=" + boundary in log for log in logs)
                   for boundary in ("receive.item", "receive.terminal", "assembly", "normalize"))
        assert all("source_count=" in log and "metadata_status=" in log for log in logs)
        assert not any(word in " ".join(logs) for word in (
            "private", "credential", "secret.example", "turn7search0", "provider omitted", "Retrieved"))
    else:
        assert logs == []


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/source", "https://user:password@example.org/",
                                 "https:///missing-host", "https://example.org/\n", "https://example.org/\x00"])
def test_helper_revalidates_persisted_carrier_sources_and_strips_untrusted_fields(url):
    from agent.retrieval_sources import extract_retrieval_sources

    sources = [dict(url=url, provenance="web_search_call.action.sources", request_id="req"),
               dict(url=URL, provenance="web_search_call.action.sources", request_id="req", title="Safe",
                    ref_id="turn7search0", query="private query", unexpected=object()),
               dict(url=URL, provenance="guessed_from_prose", request_id="req")]
    carrier = dict(type="hermes_retrieval_sources", version=1, sources=sources)
    result = extract_retrieval_sources([dict(role="assistant", codex_message_items=[carrier])])
    assert result == [dict(url=URL, provenance="web_search_call.action.sources", request_id="req", title="Safe")]
    json.dumps(result)


def test_legacy_typed_annotations_are_extracted_but_plain_prose_links_are_not():
    from agent.retrieval_sources import extract_retrieval_sources

    annotation = dict(type="url_citation", url=URL, title="Cited", start_index=0, end_index=5)
    sidecar = [vars(answer(text="Claim", annotations=[annotation]))]
    # Convert the legacy JSON sidecar as persistence would, without new registry metadata.
    sidecar[0]["content"] = [vars(part) for part in sidecar[0]["content"]]
    history = [dict(role="assistant", content="[Invented](https://wrong.example/)", codex_message_items=sidecar),
               dict(role="tool", content="https://not-retained.example/"),
               dict(role="assistant", content="https://also-not-metadata.example/")]
    sources = extract_retrieval_sources(history)
    assert len(sources) == 1 and sources[0]["url"] == URL
    assert sources[0]["provenance"] == "url_citation"
    assert "request_id" not in sources[0] and "response_id" not in sources[0]


@pytest.mark.parametrize("response_id", [None, "real_response_id"])
def test_real_sdk_source_items_normalize_with_independent_request_scope(response_id):
    from openai.types.responses import ResponseFunctionWebSearch
    from agent.retrieval_sources import extract_retrieval_sources

    item = ResponseFunctionWebSearch.model_validate(dict(
        type="web_search_call", id="ws_sdk", status="completed",
        action=dict(type="search", query="private query", sources=[dict(type="url", url=URL)])))
    response = NS(status="completed", id=response_id, output=[item, answer()])
    normalized = ResponsesApiTransport().normalize_response(response)
    source = extract_retrieval_sources(normalized)[0]
    assert source.get("request_id")
    assert source["item_id"] == "ws_sdk"
    assert source.get("response_id") == response_id
    json.dumps(normalized.codex_message_items)
    other = ResponsesApiTransport().normalize_response(NS(status="completed", output=[item, answer()]))
    assert extract_retrieval_sources(other)[0]["request_id"] != source["request_id"]


@pytest.mark.parametrize("mismatch", ["item", "text"])
def test_terminal_url_retained_without_attaching_offsets_to_different_authoritative_text(mismatch):
    from agent.retrieval_sources import extract_retrieval_sources

    annotated = answer(annotations=[dict(type="url_citation", url=URL, title="Received",
                                       start_index=6, end_index=6 + len(MARKER))])
    done = answer(item_id="different_item" if mismatch == "item" else "msg_1",
                  text="Other " + MARKER if mismatch == "text" else "Claim " + MARKER)
    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=done),
        NS(type="response.completed", response=NS(id="resp_1", status="completed", output=[annotated])),
    ], model="test")
    source = extract_retrieval_sources(response)[0]
    assert source["item_id"] == "msg_1"
    assert source["location_verified"] is False
    normalized = ResponsesApiTransport().normalize_response(response)
    assert "https://" not in normalized.content
    assert "[unresolved citation:" in normalized.content
    assert extract_retrieval_sources(normalized) == [source]


def test_delta_annotations_keep_real_item_and_output_part_indexes_without_done_items():
    from agent.retrieval_sources import extract_retrieval_sources

    annotation = dict(type="url_citation", url=URL, title="Cited", start_index=0, end_index=5)
    response = _consume_codex_event_stream([
        NS(type="response.output_text.delta", item_id="msg_delta", output_index=4, content_index=2, delta="Claim"),
        NS(type="response.output_text.annotation.added", item_id="msg_delta", output_index=4,
           content_index=2, annotation_index=0, annotation=annotation),
        NS(type="response.completed", response=NS(id="resp_delta", status="completed", output=None)),
    ], model="test")
    source = extract_retrieval_sources(response)[0]
    assert source["output_index"] == 4
    assert source["content_index"] == 2
    assert source["item_id"] == "msg_delta"
    assert source["location_verified"] is True


def test_foreign_response_metadata_cannot_cross_stream_response_identity():
    from agent.retrieval_sources import extract_retrieval_sources

    response = _consume_codex_event_stream([
        NS(type="response.created", response=NS(id="resp_owned", status="in_progress")),
        NS(type="response.output_item.done", response_id="resp_foreign", output_index=0,
           item=search_item(url="https://wrong.example/")),
        NS(type="response.output_item.done", response_id="resp_owned", output_index=0,
           item=search_item()),
        NS(type="response.completed", response=NS(id="resp_foreign", status="completed",
                                                  output=[search_item(url="https://wrong.example/terminal")])),
        NS(type="response.completed", response=NS(id="resp_owned", status="completed", output=None)),
    ], model="test", request_id="req_owned")
    sources = extract_retrieval_sources(response)
    assert len(sources) == 1
    assert sources[0]["url"] == URL
    assert sources[0]["response_id"] == "resp_owned"
    assert response.id == "resp_owned"


def test_unattached_annotation_event_url_is_kept_as_unlocated_metadata_not_guessed_link():
    from agent.retrieval_sources import extract_retrieval_sources

    annotation = dict(type="url_citation", url=URL, title="Received", start_index=0, end_index=5)
    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=search_item()),
        NS(type="response.output_text.delta", item_id="msg_missing", output_index=1, content_index=0,
           delta="Claim " + MARKER),
        NS(type="response.output_text.annotation.added", item_id="msg_missing", output_index=1, content_index=0,
           annotation_index=2, annotation=annotation),
        NS(type="response.completed", response=NS(id="resp_1", status="completed", output=None)),
    ], model="test")
    normalized = ResponsesApiTransport().normalize_response(response)
    citations = [source for source in extract_retrieval_sources(normalized) if source["provenance"] == "url_citation"]
    assert len(citations) == 1
    assert citations[0]["item_id"] == "msg_missing"
    assert citations[0]["annotation_index"] == 2
    assert citations[0]["location_verified"] is False
    assert normalized.content == "Claim [unresolved citation: turn7search0]"


def test_leaked_tool_text_is_not_replayed_but_verified_sources_still_persist():
    from agent.retrieval_sources import extract_retrieval_sources

    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=search_item()),
        NS(type="response.output_item.done", output_index=1, item=answer(text='to=functions.terminal {"command":"pwd"}')),
        NS(type="response.completed", response=NS(id="resp_leak", status="completed", output=None)),
    ], model="test")
    normalized = ResponsesApiTransport().normalize_response(response)
    assert normalized.finish_reason == "incomplete" and normalized.content == ""
    assert extract_retrieval_sources(normalized) == extract_retrieval_sources(response)
    assert len(normalized.codex_message_items) == 1
    assert normalized.codex_message_items[0]["type"] == "hermes_retrieval_sources"
    assert _chat_messages_to_responses_input([dict(role="assistant", content="", codex_message_items=normalized.codex_message_items)]) == []


def test_runtime_finalizer_keeps_original_request_identity_after_agent_advances(monkeypatch):
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    from agent.retrieval_sources import extract_retrieval_sources
    from tests.agent.test_codex_retrieval_source_include import agent

    runtime = agent()
    runtime._current_api_request_id = "req_original"
    finalizers, chunk_hooks = [], []
    real_stream = relay_llm.stream

    def traced_stream(*args, **kwargs):
        finalizers.append(kwargs["finalizer"])
        chunk_hooks.append(kwargs["on_chunk"])
        return real_stream(*args, **kwargs)

    def events():
        event = NS(type="response.output_item.done", output_index=0, item=search_item())
        chunk_hooks[0](event)  # emulate the managed Relay's chunk interception
        yield event
        runtime._current_api_request_id = "req_newer"
        event = NS(type="response.completed", response=NS(id="resp_original", status="completed", output=None))
        chunk_hooks[0](event)
        yield event

    monkeypatch.setattr(relay_llm, "stream", traced_stream)
    response = run_codex_stream(runtime, {"model": "test"}, client=NS(responses=NS(create=lambda **_: events())))
    assert extract_retrieval_sources(response)[0]["request_id"] == "req_original"
    assert extract_retrieval_sources(finalizers[0]())[0]["request_id"] == "req_original"


def test_settled_pending_calls_do_not_invent_dense_indexes_for_anonymous_citations():
    from agent.retrieval_sources import extract_retrieval_sources

    annotation = dict(type="url_citation", url=URL, title="Cited", start_index=0, end_index=5)
    response = _consume_codex_event_stream([
        NS(type="response.output_item.added", output_index=0,
           item=NS(type="function_call", id="fc_pending", call_id="call_pending", name="terminal", arguments="{}")),
        NS(type="response.output_item.done", output_index=4, item=answer(text="Claim", item_id=None, annotations=[annotation])),
        NS(type="response.completed", response=NS(id="resp_sparse", status="completed", output=None)),
    ], model="test")
    sources = extract_retrieval_sources(response)
    assert len(sources) == 1
    assert sources[0]["output_index"] == 4
    assert "item_id" not in sources[0]


@pytest.mark.parametrize("invalid_type", [[], {}, 42])
def test_malformed_metadata_types_fail_closed_without_breaking_a_response(invalid_type):
    from agent.retrieval_sources import extract_retrieval_sources

    carrier = dict(type="hermes_retrieval_sources", version=1,
                   sources=[dict(url=URL, provenance=invalid_type)])
    assert extract_retrieval_sources([dict(role="assistant", codex_message_items=[carrier])]) == []
    item = NS(type="web_search_call", id="ws_bad", action=NS(sources=[dict(type=invalid_type, url=URL)]))
    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=item),
        NS(type="response.output_item.done", output_index=1, item=answer(text="Plain answer")),
        NS(type="response.completed", response=NS(id="resp_bad", status="completed", output=None)),
    ], model="test")
    assert extract_retrieval_sources(response) == []
    assert ResponsesApiTransport().normalize_response(response).content == "Plain answer"


def test_native_compaction_wire_pruning_does_not_mutate_local_source_sidecars(monkeypatch):
    from agent.history_commentary import project_history_commentary
    from agent.retrieval_sources import extract_retrieval_sources

    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=search_item()),
        NS(type="response.output_item.done", output_index=1, item=answer()),
        NS(type="response.completed", response=NS(id="resp_history", status="completed", output=None)),
    ], model="test")
    normalized = ResponsesApiTransport().normalize_response(response)
    history = [dict(role="assistant", content=normalized.content, codex_message_items=normalized.codex_message_items),
               dict(role="assistant", content="retained tail", codex_reasoning_items=[dict(
                   type="compaction", encrypted_content="opaque-checkpoint", _issuer_kind="codex_backend")])]
    before = json.dumps(history, sort_keys=True)
    sources = extract_retrieval_sources(history)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"display": {"show_commentary": False}})
    projected = project_history_commentary(history)
    assert extract_retrieval_sources(projected) == sources
    wire = _chat_messages_to_responses_input(history, current_issuer_kind="codex_backend", native_compaction_eligible=True)
    assert any(item["type"] == "compaction" for item in wire)
    assert "hermes_retrieval_sources" not in json.dumps(wire)
    assert URL not in json.dumps(wire)
    assert json.dumps(history, sort_keys=True) == before
    assert extract_retrieval_sources(history) == sources
