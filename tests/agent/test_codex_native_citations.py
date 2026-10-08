"""Native Responses citations must reach every chat surface as portable Markdown."""
from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.transports.codex import ResponsesApiTransport


MARKER = "\ue200cite\ue202turn261832view6\ue202turn261832view8\ue201"


def response_with(text, annotations=()):
    return NS(status="completed", output=[NS(
        type="message", role="assistant", id="msg_cited", status="completed", phase="final_answer",
        content=[NS(type="output_text", text=text, annotations=list(annotations))],
    )])


def citation(text, url="https://example.org/source", title="Source") -> dict:
    return dict(type="url_citation", start_index=text.index(MARKER),
                end_index=text.index(MARKER) + len(MARKER), url=url, title=title)


def test_transport_renders_native_annotation_urls_without_inventing_ref_mapping():
    text = "容量は十分です。" + MARKER
    annotations = [citation(text), NS(**citation(text, "https://example.org/other", "Other"))]
    result = ResponsesApiTransport().normalize_response(response_with(text, annotations))
    assert result.content == "容量は十分です。[Source](https://example.org/source) [Other](https://example.org/other)"
    # Wire replay retains original text and offsets, not rendered text with stale offsets.
    assert result.provider_data is not None
    part = result.provider_data["codex_message_items"][0]["content"][0]
    assert part["text"] == text
    assert part["annotations"] == [annotations[0], vars(annotations[1])]


@pytest.mark.parametrize("output", [[], [NS(type="web_search_call", status="completed")]],
                         ids=["empty_output", "server_tool_only"])
def test_unknown_native_refs_are_explicit_not_dropped_or_linked_to_guessed_urls(output):
    expected = "Claim [unresolved citation: turn261832view6, turn261832view8]"
    message, _ = _normalize_codex_response(response_with("Claim " + MARKER))
    assert message.content == expected
    # Also cover the response.output_text fallback without output items.
    fallback, _ = _normalize_codex_response(NS(status="completed", output=output, output_text="Claim " + MARKER))
    assert fallback.content == expected


@pytest.mark.parametrize("done_item", [False, True], ids=["deltas_only", "item_done"])
def test_sse_annotation_events_survive_assembly_at_their_content_part(done_item):
    from agent.codex_runtime import _consume_codex_event_stream

    text = "Claim " + MARKER
    annotation = citation(text)
    events = [
        NS(type="response.output_item.added", output_index=0, item=NS(
            type="message", id="msg_cited", phase="final_answer")),
        NS(type="response.output_text.delta", item_id="msg_cited", output_index=0, content_index=0, delta=text),
        NS(type="response.output_text.annotation.added", item_id="msg_cited", output_index=0,
           content_index=0, annotation_index=0, annotation=annotation),
    ]
    if done_item:
        events.append(NS(type="response.output_item.done", output_index=0, item=response_with(text).output[0]))
    events.append(NS(type="response.completed", response=NS(status="completed", output=None)))
    response = _consume_codex_event_stream(events, model="gpt-test")
    message, _ = _normalize_codex_response(response)
    assert message.content == "Claim [Source](https://example.org/source)"
    assert message.codex_message_items[0]["content"][0]["annotations"] == [annotation]


def test_annotated_delta_only_stream_preserves_refusal_text():
    from agent.codex_runtime import _consume_codex_event_stream

    text = "Claim " + MARKER
    refusal = " I cannot help with that."
    events = [
        NS(type="response.output_text.delta", item_id="msg_cited", output_index=0,
           content_index=0, delta=text),
        NS(type="response.output_text.annotation.added", item_id="msg_cited", output_index=0,
           content_index=0, annotation_index=0, annotation=citation(text)),
        NS(type="response.refusal.delta", item_id="msg_cited", output_index=0,
           content_index=1, delta=refusal),
        NS(type="response.completed", response=NS(status="completed", output=None)),
    ]
    message, _ = _normalize_codex_response(_consume_codex_event_stream(events, model="gpt-test"))
    assert message.content == "Claim [Source](https://example.org/source)" + refusal
    assert message.codex_message_items[0]["content"][0]["annotations"] == [citation(text)]
    assert message.codex_message_items[0]["content"][1]["text"] == refusal


def test_annotation_span_over_prose_does_not_erase_the_claim():
    annotation = dict(type="url_citation", start_index=0, end_index=5,
                      title="Source", url="https://example.org/source")
    message, _ = _normalize_codex_response(response_with("Claim remains.", [annotation]))
    assert message.content == "Claim [Source](https://example.org/source) remains."


def test_citation_metadata_survives_history_conversion_and_preflight():
    from agent.codex_responses_adapter import _chat_messages_to_responses_input, _preflight_codex_input_items

    text = "Claim " + MARKER
    message, _ = _normalize_codex_response(response_with(text, [citation(text)]))
    history = [{"role": "assistant", "content": message.content,
                "codex_message_items": message.codex_message_items}]
    items = _preflight_codex_input_items(_chat_messages_to_responses_input(history))
    assert items[0]["content"][0] == dict(type="output_text", text=text, annotations=[citation(text)])


def test_preflight_omits_stale_annotations_when_real_sanitizer_changes_text():
    from agent.codex_responses_adapter import _chat_messages_to_responses_input, _preflight_codex_input_items

    text = "<|\u200bend|>Claim " + MARKER
    annotation = citation(text)
    message, _ = _normalize_codex_response(response_with(text, [annotation]))
    history = [{"role": "assistant", "content": message.content,
                "codex_message_items": message.codex_message_items}]
    items = _preflight_codex_input_items(
        _chat_messages_to_responses_input(history), sanitize_harmony_tokens=True,
    )
    assert items[0]["content"][0] == dict(type="output_text", text="<｜end｜>Claim " + MARKER)
    # Sanitizing replay must not rewrite the original persisted sidecar.
    assert message.codex_message_items[0]["content"][0] == dict(
        type="output_text", text=text, annotations=[annotation],
    )


@pytest.mark.parametrize("offsets", [{}, {"start_index": -1, "end_index": 2},
                                    {"start_index": 0, "end_index": 999}], ids=["missing", "negative", "out_of_range"])
def test_unlocated_source_urls_are_retained_without_claiming_they_resolve_a_ref(offsets):
    annotation = dict(type="url_citation", title="Source", url="https://example.org/source", **offsets)
    message, _ = _normalize_codex_response(response_with("Claim " + MARKER, [annotation]))
    assert message.content == (
        "Claim [unresolved citation: turn261832view6, turn261832view8]"
        "\n\nSources (citation location unavailable): [Source](https://example.org/source)"
    )


@pytest.mark.parametrize("done_item", [False, True], ids=["deltas_only", "item_done"])
def test_terminal_annotations_supplement_missing_sse_annotation_events(done_item):
    from agent.codex_runtime import _consume_codex_event_stream

    text = "Claim " + MARKER
    annotated = response_with(text, [citation(text)])
    events = [NS(type="response.output_text.delta", item_id="msg_cited", output_index=0,
                 content_index=0, delta=text)]
    if done_item:
        events.append(NS(type="response.output_item.done", output_index=0, item=response_with(text).output[0]))
    events.append(NS(type="response.completed", response=annotated))
    response = _consume_codex_event_stream(events, model="gpt-test")
    message, _ = _normalize_codex_response(response)
    assert message.content == "Claim [Source](https://example.org/source)"


@pytest.mark.parametrize("annotation_source", ["terminal", "event"])
def test_annotations_are_not_attached_to_a_different_done_text(annotation_source):
    from agent.codex_runtime import _consume_codex_event_stream

    text = "Claim " + MARKER
    changed = "Other " + MARKER
    events = []
    if annotation_source == "event":
        events = [NS(type="response.output_text.delta", item_id="msg_cited", output_index=0,
                     content_index=0, delta=text),
                  NS(type="response.output_text.annotation.added", item_id="msg_cited", output_index=0,
                     content_index=0, annotation_index=0, annotation=citation(text))]
    events.append(NS(type="response.output_item.done", output_index=0, item=response_with(changed).output[0]))
    events.append(NS(type="response.completed", response=response_with(text, [citation(text)])))
    message, _ = _normalize_codex_response(_consume_codex_event_stream(events, model="gpt-test"))
    assert message.content == "Other [unresolved citation: turn261832view6, turn261832view8]"


def test_anonymous_sse_items_do_not_borrow_another_output_items_citation():
    from agent.codex_runtime import _consume_codex_event_stream

    text = "Claim " + MARKER
    items = [response_with(text).output[0], response_with(text).output[0]]
    for item in items:
        item.id = None
    events = [NS(type="response.output_text.annotation.added", output_index=1,
                 content_index=0, annotation_index=0, annotation=citation(text))]
    events += [NS(type="response.output_item.done", output_index=index, item=item)
               for index, item in enumerate(items)]
    events.append(NS(type="response.completed", response=NS(status="completed", output=None)))
    message, _ = _normalize_codex_response(_consume_codex_event_stream(events, model="gpt-test"))
    assert message.content == ("Claim [unresolved citation: turn261832view6, turn261832view8]\n"
                               "Claim [Source](https://example.org/source)")


def test_overlapping_prose_spans_do_not_corrupt_links_or_original_text():
    text = "Claim remains true."
    annotations = [dict(type="url_citation", start_index=0, end_index=len(text),
                        title="Source", url="https://example.org/source"),
                   dict(type="url_citation", start_index=6, end_index=13,
                        title="Other", url="https://example.org/other")]
    message, _ = _normalize_codex_response(response_with(text, annotations))
    assert message.content == (
        "Claim remains [Other](https://example.org/other) true. [Source](https://example.org/source)"
    )


def test_partial_native_span_does_not_change_the_unresolved_ref_identifiers():
    text = "Claim " + MARKER
    annotation = citation(text)
    annotation["end_index"] -= 2
    message, _ = _normalize_codex_response(response_with(text, [annotation]))
    assert message.content == (
        "Claim [unresolved citation: turn261832view6, turn261832view8]"
        "\n\nSources (citation location unavailable): [Source](https://example.org/source)"
    )


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://example.org/source\n"])
def test_unsafe_or_malformed_annotation_urls_are_not_rendered_as_links(url):
    text = "Claim " + MARKER
    message, _ = _normalize_codex_response(response_with(text, [citation(text, url)]))
    assert message.content == "Claim [unresolved citation: turn261832view6, turn261832view8]"


def test_annotation_url_backslash_cannot_escape_markdown_destination_end():
    text = "Claim " + MARKER
    message, _ = _normalize_codex_response(response_with(
        text, [citation(text, "https://example.org/source\\")],
    ))
    assert message.content == "Claim [Source](https://example.org/source%5C)"


def test_real_sdk_annotations_are_serializable_with_markdown_safe_labels_and_urls():
    from openai.types.responses import ResponseOutputMessage
    from agent.codex_runtime import _consume_codex_event_stream
    import json

    text = "Claim " + MARKER
    item = ResponseOutputMessage.model_validate(dict(
        type="message", role="assistant", id="msg_cited", status="completed",
        content=[dict(type="output_text", text=text, annotations=[citation(
            text, "https://example.org/source(revision)", "Guide [Docs]")])],
    ))
    response = _consume_codex_event_stream([
        NS(type="response.output_item.done", output_index=0, item=item),
        NS(type="response.completed", response=NS(status="completed", output=None)),
    ], model="gpt-test")
    message, _ = _normalize_codex_response(response)
    assert message.content == r"Claim [Guide \[Docs\]](https://example.org/source%28revision%29)"
    json.dumps(message.codex_message_items)  # persisted annotation sidecar contains no SDK objects
