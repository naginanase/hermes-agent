"""Slack delivery must not expose internal citation diagnostics as source links."""
from types import SimpleNamespace as NS
import json
import pytest
from copy import deepcopy

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def test_slack_formatter_replaces_unresolved_ids_with_visible_attribution_warning():
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True)
    text = "比較には実機検証が必要です。[unresolved citation: turn0view4]"
    shown = adapter.format_message(text)
    assert "unresolved citation" not in shown
    assert "turn0view4" not in shown
    assert "比較には実機検証が必要です。" in shown
    assert "出典" in shown
    assert "確認" in shown


def test_retrieved_sources_are_clickable_without_claiming_an_opaque_ref_mapping():
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    messages = [{"role": "user", "content": "比較して"}, {"role": "assistant", "codex_message_items": [
        {"type": "hermes_retrieval_sources", "request_id": "resp_sources", "sources": [
            {"url": "https://example.org/source", "provenance": "web_search_call.action.sources", "title": "Retrieved source"},
        ]},
    ]}]
    original = deepcopy(messages)
    result = prepare_slack_citation_delivery(
        "比較が必要です。[unresolved citation: turn0view4]", messages=messages)
    assert "https://example.org/source" in result.text
    assert "対応" in result.text
    assert "turn0view4" not in result.text
    assert result.diagnostic_refs == ("turn0view4",)
    assert result.source_count == 1
    assert messages == original


def test_gateway_repairs_citations_before_native_stream_finalization():
    from unittest.mock import Mock
    from gateway.run_turn_runner import TurnRunner
    from gateway.config import Platform
    ctx = NS(source=NS(platform=Platform.SLACK), result_holder=[None])
    runner = TurnRunner(None, ctx)
    result = {"completed": True, "final_response": "比較が必要です。[unresolved citation: turn0view4]",
              "messages": [{"role": "user", "content": "比較して"}, {"role": "assistant", "codex_message_items": [
                  {"type": "hermes_retrieval_sources", "request_id": "resp_gate", "sources": [
                      {"url": "https://example.org/source", "provenance": "web_search_call.action.sources", "title": "Source"}]}]}]}
    consumer = Mock()
    runner._finish_stream_consumer(result, [], consumer)
    final_text = consumer.finish.call_args.args[0]
    assert "unresolved citation" not in final_text
    assert "turn0view4" not in final_text
    assert "https://example.org/source" in final_text
    assert result["response_transformed"] is True
    assert ctx.result_holder[0] is result


def test_citation_preview_buffers_split_native_markers_until_authoritative_final():
    from agent.slack_citation_delivery import SlackCitationPreviewBuffer
    preview = SlackCitationPreviewBuffer()
    deltas = ["比較です。\ue200", "cite\ue202turn0", "view4\ue201続きです。"]
    visible = "".join(preview.feed(part) for part in deltas)
    assert visible == "比較です。"
    assert "turn0" not in visible
    assert "\ue200" not in visible
    assert preview.feed("さらに続き") == ""


def test_gateway_stream_callback_buffers_citations_only_for_slack_text_preview():
    from unittest.mock import Mock, patch
    from gateway.run_turn_runner import TurnRunner
    from gateway.config import Platform, StreamingConfig
    consumer, speech = Mock(), Mock()
    ctx = NS(mute_notification_reply=False, streaming_tts_consumer_holder=[speech],
             user_config={}, scheduled_heartbeat=False, interim_assistant_messages_enabled=False,
             resolve_display_setting=lambda *args: None, source=NS(platform=Platform.SLACK, chat_id="C1"),
             _status_thread_metadata={}, progress_queue=None, event_message_id="111.000",
             _run_still_current=lambda: True, stream_consumer_holder=[None])
    outer = Mock(config=NS(streaming=StreamingConfig(enabled=True)))
    outer._build_stream_consumer_config.return_value = (NS(), None)
    with patch("gateway.stream_consumer.GatewayStreamConsumer", return_value=consumer):
        _, cb, _, _ = TurnRunner(outer, ctx)._setup_stream_consumer("slack")
    for delta in ("Claim.\ue200", "cite\ue202turn0view0\ue201Tail"):
        cb(delta)
    visible = "".join(call.args[0] for call in consumer.on_delta.call_args_list)
    assert visible == "Claim."
    assert speech.on_delta.call_count == 2


@pytest.mark.parametrize("extra", [{"markdown_blocks": True}, {"rich_blocks": True}])
def test_block_payload_hides_internal_refs_with_visible_uncertainty(extra):
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra=extra))
    blocks = adapter._maybe_blocks("比較です。[unresolved citation: turn0view4]")
    payload = json.dumps(blocks, ensure_ascii=False)
    assert blocks
    assert "turn0view4" not in payload
    assert "unresolved citation" not in payload
    assert "引用" in payload


@pytest.mark.asyncio
async def test_native_finalization_repairs_direct_send_citations_without_a_duplicate():
    from unittest.mock import AsyncMock, Mock
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter._app = Mock()
    adapter._running = True
    client = AsyncMock()
    client.chat_startStream.return_value = {"ok": True, "ts": "123.456"}
    client.chat_stopStream.return_value = {"ok": True}
    adapter._get_client = Mock(return_value=client)
    adapter.stop_typing = AsyncMock()
    metadata = {"thread_id": "111.000", "user_id": "U123"}
    await adapter.send_draft("C1", 7, "比較です。", metadata=metadata)
    result = await adapter.send("C1", "比較です。[unresolved citation: turn0view4]", metadata=metadata)
    assert result.success
    final_payload = json.dumps(client.chat_stopStream.await_args.kwargs, ensure_ascii=False)
    assert "turn0view4" not in final_payload
    assert "引用" in final_payload
    client.chat_postMessage.assert_not_awaited()


@pytest.mark.parametrize("marker", ["[unresolved citation: turn0view4]", "\ue200cite\ue202turn0view4\ue201"])
@pytest.mark.parametrize("tail", ["", "続きです。"])
@pytest.mark.asyncio
async def test_citation_containing_native_draft_is_repaired_on_its_original_message(marker, tail):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    draft = "比較です。" + marker
    started = await adapter.send_draft("C1", 7, draft, metadata=META)
    assert started.success
    assert client.chat_startStream.await_args.kwargs["markdown_text"] == draft

    result = await adapter.send("C1", draft + tail, metadata=META)

    assert result.success
    assert result.message_id == started.message_id == "123.456"
    client.chat_postMessage.assert_not_awaited()
    client.chat_stopStream.assert_awaited_once()
    assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
    client.chat_update.assert_awaited_once()
    final = client.chat_update.await_args.kwargs
    assert final["ts"] == started.message_id
    assert "比較です。" in final["text"]
    assert tail in final["text"]
    payload = json.dumps(final, ensure_ascii=False)
    assert "turn0view4" not in payload
    assert "unresolved citation" not in payload
    assert "\ue200" not in payload
    assert "引用" in payload
    assert not adapter._active_streams


@pytest.mark.parametrize("prefix, marker", [
    ("[unresolved citation: turn0", "[unresolved citation: turn0view4]"),
    ("\ue200cite\ue202turn0", "\ue200cite\ue202turn0view4\ue201"),
])
@pytest.mark.asyncio
async def test_native_stream_with_a_partial_citation_prefix_is_repaired_in_place(prefix, marker):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    started = await adapter.send_draft("C1", 7, "比較です。" + prefix, metadata=META)

    result = await adapter.send("C1", "比較です。" + marker, metadata=META)

    assert result.success
    assert result.message_id == started.message_id
    client.chat_postMessage.assert_not_awaited()
    client.chat_update.assert_awaited_once()
    payload = json.dumps(client.chat_update.await_args.kwargs, ensure_ascii=False)
    assert "turn0" not in payload
    assert "unresolved citation" not in payload
    assert "\ue200" not in payload
    assert "引用" in payload
    assert not adapter._active_streams


@pytest.mark.parametrize("tail", ["", "続きです。"])
@pytest.mark.asyncio
async def test_raw_native_stream_accepts_an_already_repaired_source_aware_final(tail):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter({"rich_blocks": True})
    draft = "比較です。[unresolved citation: turn0view4]"
    started = await adapter.send_draft("C1", 7, draft, metadata=META)
    final = prepare_slack_citation_delivery(
        draft + tail, messages=[_retrieval_message("https://example.org/current")]).text

    result = await adapter.send("C1", final, metadata=META)

    assert result.success
    assert result.message_id == started.message_id
    client.chat_postMessage.assert_not_awaited()
    client.chat_stopStream.assert_awaited_once()
    assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
    client.chat_update.assert_awaited_once()
    payload = client.chat_update.await_args.kwargs
    assert payload["ts"] == started.message_id
    assert payload["text"] == adapter.format_message(final)
    assert payload["blocks"]
    assert "https://example.org/current" in json.dumps(payload)
    assert "turn0view4" not in json.dumps(payload)
    assert not adapter._active_streams


@pytest.mark.parametrize("prefix, marker", [
    ("[unresolved citation: turn0", "[unresolved citation: turn0view4]"),
    ("\ue200cite\ue202turn0", "\ue200cite\ue202turn0view4\ue201"),
])
@pytest.mark.parametrize("tail", ["", "続きです。"])
@pytest.mark.asyncio
async def test_partial_raw_draft_accepts_source_aware_final_on_original_message(prefix, marker, tail):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter({"rich_blocks": True})
    metadata = dict(META, team_id="T1")
    draft = "比較です。" + prefix
    started = await adapter.send_draft("C1", 7, draft, metadata=metadata)
    messages = [_retrieval_message("https://example.org/current")]
    original = deepcopy(messages)
    final = prepare_slack_citation_delivery("比較です。" + marker + tail, messages=messages).text

    result = await adapter.send("C1", final, metadata=metadata)

    assert result.success
    assert result.message_id == started.message_id
    client.chat_postMessage.assert_not_awaited()
    client.chat_stopStream.assert_awaited_once()
    assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
    client.chat_update.assert_awaited_once()
    payload = client.chat_update.await_args.kwargs
    assert payload["ts"] == started.message_id
    assert payload["text"] == adapter.format_message(final)
    assert payload["blocks"]
    assert "https://example.org/current" in json.dumps(payload)
    assert "turn0" not in json.dumps(payload)
    assert "unresolved citation" not in json.dumps(payload)
    assert "\ue200" not in json.dumps(payload, ensure_ascii=False)
    assert messages == original
    assert draft == "比較です。" + prefix
    assert not adapter._active_streams


@pytest.mark.parametrize("prefix, marker", [
    ("[unresolved citation: turn0view4]", "[unresolved citation: turn0view4]"),
    ("[unresolved citation: turn0", "[unresolved citation: turn0view4]"),
    ("\ue200cite\ue202turn0view4\ue201", "\ue200cite\ue202turn0view4\ue201"),
    ("\ue200cite\ue202turn0", "\ue200cite\ue202turn0view4\ue201"),
])
@pytest.mark.parametrize("source_aware", [False, True])
@pytest.mark.asyncio
async def test_oversized_citation_replacement_delivers_all_bounded_chunks(prefix, marker, source_aware):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    from tests.gateway.test_slack_native_streaming import _make_adapter, META, META_B
    adapter, client = _make_adapter({"rich_blocks": True, "reply_broadcast": True})
    metadata = dict(META, team_id="T1")
    started = await adapter.send_draft("C1", 7, "Claim." + prefix, metadata=metadata)
    # Neither another thread nor the same thread in another workspace is ours to close.
    others = [("T1", "C1", META_B["thread_id"]), ("T2", "C1", META["thread_id"])]
    for key in others:
        adapter._active_streams[key] = dict(ts="other", sent="Other answer")
    raw = "Claim." + marker + "\n\n" + "A" * (adapter.MAX_MESSAGE_LENGTH + 64) + "\nEND_REQUIRED"
    sources = [_retrieval_message("https://example.org/current")] if source_aware else []
    visible = prepare_slack_citation_delivery(raw, messages=sources).text
    final = visible if source_aware else raw
    expected = adapter._format_chunks(visible)
    assert len(expected) > 1

    result = await adapter.send("C1", final, metadata=metadata)

    assert result.success
    assert result.message_id == started.message_id
    client.chat_stopStream.assert_awaited_once()
    assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
    client.chat_update.assert_awaited_once()
    update = client.chat_update.await_args.kwargs
    assert update["ts"] == started.message_id
    delivered = [update["text"]] + [call.kwargs["text"] for call in client.chat_postMessage.await_args_list]
    assert delivered == expected  # No prefix repetition and no tail lost to chat.update truncation.
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in delivered)
    payload = json.dumps([update] + [call.kwargs for call in client.chat_postMessage.await_args_list])
    assert "END_REQUIRED" in payload
    assert "turn0" not in payload
    assert "unresolved citation" not in payload
    assert "https://example.org/current" in payload if source_aware else "unavailable" in payload
    for call in client.chat_postMessage.await_args_list:
        assert call.kwargs["channel"] == "C1"
        assert call.kwargs["thread_ts"] == META["thread_id"]
        assert not call.kwargs.get("reply_broadcast")
    assert set(adapter._active_streams) == set(others)
    client_requests = getattr(adapter._get_client, "call_args_list")
    assert all(call.kwargs.get("team_id") == "T1" for call in client_requests)


@pytest.mark.parametrize("failed_continuation", [0, 1])
@pytest.mark.asyncio
async def test_oversized_citation_replacement_does_not_report_partial_delivery_success(failed_continuation):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    draft = "Claim.[unresolved citation: turn0view4]"
    await adapter.send_draft("C1", 7, draft, metadata=META)
    final = draft + "\n\n" + "A" * (2 * adapter.MAX_MESSAGE_LENGTH + 64) + "\nEND_REQUIRED"
    client.chat_postMessage.side_effect = (
        [{"ts": "continuation"}] * failed_continuation + [RuntimeError("continuation failed")]
    )

    result = await adapter.send("C1", final, metadata=META)

    assert not result.success
    assert result.error == "continuation failed"
    assert client.chat_postMessage.await_count == failed_continuation + 1
    client.chat_update.assert_awaited_once()
    assert not adapter._active_streams


@pytest.mark.parametrize("stop_fails", [False, True])
@pytest.mark.parametrize("update_fails", [False, True])
@pytest.mark.asyncio
async def test_oversized_citation_replacement_preserves_stop_update_fallbacks(stop_fails, update_fails):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    draft = "Claim.[unresolved citation: turn0view4]"
    await adapter.send_draft("C1", 7, draft, metadata=META)
    final = draft + "\n\n" + "A" * (adapter.MAX_MESSAGE_LENGTH + 64) + "\nEND_REQUIRED"
    expected = adapter._format_chunks(prepare_slack_citation_delivery(final).text)
    if stop_fails:
        client.chat_stopStream.side_effect = RuntimeError("stop failed")
    if update_fails:
        client.chat_update.side_effect = RuntimeError("update failed")

    result = await adapter.send("C1", final, metadata=META)

    assert result.success
    assert not adapter._active_streams
    assert client.chat_stopStream.await_count == (2 if stop_fails else 1)
    assert all("markdown_text" not in call.kwargs for call in client.chat_stopStream.await_args_list)
    client.chat_update.assert_awaited_once()
    posted = [call.kwargs["text"] for call in client.chat_postMessage.await_args_list]
    if update_fails:
        assert posted == expected
        assert result.message_id == "999.111"
    else:
        assert [client.chat_update.await_args.kwargs["text"]] + posted == expected
        assert result.message_id == "123.456"
    assert "END_REQUIRED" in "".join(posted)
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in posted)


@pytest.mark.parametrize("text", [
    "Plain draft without a marker",
    "`[unresolved citation: turn0view4]`",
    "```text\n[unresolved citation: turn0view4]\n```",
    "```text\n[unresolved citation: turn0",
    "```text\n\ue200cite\ue202turn0",
])
def test_native_citation_body_matching_preserves_code_and_non_citation_text(text):
    from agent.slack_citation_delivery import slack_citation_draft_body
    assert slack_citation_draft_body(text) is None


@pytest.mark.parametrize("send_kind", ["unrelated", "interim", "editable"])
@pytest.mark.parametrize("prefix", ["[unresolved citation: turn0view4]", "[unresolved citation: turn0", "\ue200cite\ue202turn0"])
@pytest.mark.asyncio
async def test_raw_citation_stream_is_not_consumed_by_unrelated_or_interim_sends(send_kind, prefix):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    draft = "比較です。" + prefix
    await adapter.send_draft("C1", 7, draft, metadata=META)
    metadata: dict = dict(META)
    if send_kind == "interim":
        metadata["_interim_send"] = True
    elif send_kind == "editable":
        metadata["expect_edits"] = True
    text = ("Unrelated notice [unresolved citation: turn0view4]" if send_kind == "unrelated"
            else "比較です。[unresolved citation: turn0view4]")

    result = await adapter.send("C1", text, metadata=metadata)

    assert result.success
    client.chat_postMessage.assert_awaited_once()
    client.chat_stopStream.assert_not_awaited()
    client.chat_update.assert_not_awaited()
    assert "turn0view4" not in json.dumps(client.chat_postMessage.await_args.kwargs)
    key = adapter._stream_key("C1", META)
    assert key is not None
    stream = adapter._active_streams[key]
    assert stream["ts"] == "123.456"
    assert stream["sent"] == draft


@pytest.mark.parametrize("stop_fails", [False, True])
@pytest.mark.parametrize("update_fails", [False, True])
@pytest.mark.asyncio
async def test_raw_citation_repair_preserves_native_commit_failure_fallbacks(stop_fails, update_fails):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter()
    draft = "比較です。[unresolved citation: turn0view4]"
    await adapter.send_draft("C1", 7, draft, metadata=META)
    if stop_fails:
        client.chat_stopStream.side_effect = RuntimeError("stop failed")
    if update_fails:
        client.chat_update.side_effect = RuntimeError("update failed")

    result = await adapter.send("C1", draft, metadata=META)

    assert result.success
    assert not adapter._active_streams
    assert client.chat_stopStream.await_count == (2 if stop_fails else 1)
    assert all("markdown_text" not in call.kwargs for call in client.chat_stopStream.await_args_list)
    client.chat_update.assert_awaited_once()
    if update_fails:
        assert result.message_id == "999.111"
        client.chat_postMessage.assert_awaited_once()
        payload = client.chat_postMessage.await_args.kwargs
    else:
        assert result.message_id == "123.456"
        client.chat_postMessage.assert_not_awaited()
        payload = client.chat_update.await_args.kwargs
    shown = json.dumps(payload, ensure_ascii=False)
    assert "turn0view4" not in shown
    assert "unresolved citation" not in shown
    assert "引用" in shown


def _retrieval_message(url):
    return {"role": "assistant", "codex_message_items": [{
        "type": "hermes_retrieval_sources", "request_id": "resp_boundary", "sources": [{
            "url": url, "provenance": "web_search_call.action.sources",
        }],
    }]}


@pytest.mark.parametrize("transition", ["in-place-compaction", "session-split"])
def test_gateway_retains_current_sources_after_transcript_rebaseline(transition):
    from unittest.mock import Mock
    from gateway.run_turn_runner import TurnRunner
    from gateway.config import Platform
    history = [{"role": "assistant", "content": "old history"}] * 20
    agent = NS(_last_compaction_in_place=transition == "in-place-compaction",
               session_id="child" if transition == "session-split" else "original")
    ctx = NS(source=NS(platform=Platform.SLACK), result_holder=[None],
             agent_holder=[agent], session_id="original", session_key="slack:C1")
    messages = [{"role": "user", "content": "current question"},
                _retrieval_message("https://example.org/current")]
    if transition == "session-split":
        messages[:0] = [{"role": "system", "content": "compacted history"},
                        _retrieval_message("https://example.org/previous")]
    original = deepcopy(messages)
    result = {"completed": True, "final_response": "Claim [unresolved citation: turn0view4]",
              "messages": messages}
    consumer = Mock()

    # Finish precedes session sync, so it still receives the old history length.
    TurnRunner(None, ctx)._finish_stream_consumer(result, history, consumer)

    final = consumer.finish.call_args.args[0]
    assert "https://example.org/current" in final
    assert "https://example.org/previous" not in final
    assert "turn0view4" not in final
    assert result["citation_delivery_diagnostics"]["source_count"] == 1
    assert messages == original


@pytest.mark.parametrize("history_offset", [0, 1, 20])
@pytest.mark.parametrize("has_current_sources", [False, True])
def test_latest_user_boundary_isolates_sources_even_with_a_stale_history_offset(history_offset, has_current_sources):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    messages = [{"role": "user", "content": "previous question"},
                _retrieval_message("https://example.org/previous"),
                {"role": "user", "content": "current question"}]
    if has_current_sources:
        messages.append(_retrieval_message("https://example.org/current"))
    result = prepare_slack_citation_delivery(
        "Claim [unresolved citation: turn0view4]", messages=messages,
        history_offset=history_offset)
    assert result.source_count == int(has_current_sources)
    assert "https://example.org/previous" not in result.text
    assert ("https://example.org/current" in result.text) is has_current_sources


def test_without_a_user_boundary_history_offset_still_excludes_old_sources():
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    result = prepare_slack_citation_delivery(
        "Claim [unresolved citation: turn0view4]",
        messages=[_retrieval_message("https://example.org/previous")], history_offset=1)
    assert result.source_count == 0
    assert "https://example.org/previous" not in result.text


def test_unrecognized_registry_provenance_is_not_a_retrieved_source():
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    result = prepare_slack_citation_delivery(
        "比較です。[unresolved citation: turn0view4]",
        messages=[dict(role="assistant", codex_message_items=[dict(
            type="hermes_retrieval_sources", sources=[dict(url="https://example.org/unverified", provenance="terminal_stdout")])])])
    assert result.source_count == 0
    assert "https://example.org/unverified" not in result.text


def test_malformed_sidecars_fail_closed_without_breaking_delivery():
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    for items in (7, [dict(type="hermes_retrieval_sources", sources=7)],
                  [dict(type="message", content=7)],
                  [dict(type="message", content=[dict(annotations=7)])]):
        result = prepare_slack_citation_delivery(
            "比較です。[unresolved citation: turn0view4]",
            messages=[dict(role="assistant", codex_message_items=items)])
        assert result.source_count == 0
        assert "turn0view4" not in result.text
        assert "取得できず" in result.text


@pytest.mark.parametrize("marker", ["[unresolved citation: literal_ref]", "\ue200cite\ue202literal_ref\ue201"])
@pytest.mark.parametrize("mixed", [False, True])
def test_gateway_final_preserves_original_double_ticks_and_literal_sidecars(marker, mixed):
    from unittest.mock import Mock
    from gateway.run_turn_runner import TurnRunner
    from gateway.config import Platform
    code = "``" + marker + "``"
    raw = code + ("\nClaim [unresolved citation: prose_ref]" if mixed else "")
    messages = [{"role": "user", "content": "examples"}, _retrieval_message("https://example.org/current"),
                {"role": "assistant", "content": raw, "codex_message_items": [{
                    "type": "message", "id": "immutable_literal", "content": [{"type": "output_text", "text": raw}]}]}]
    original = deepcopy(messages)
    ctx = NS(source=NS(platform=Platform.SLACK), result_holder=[None])
    result = {"completed": True, "final_response": raw, "messages": messages}
    consumer = Mock()
    TurnRunner(None, ctx)._finish_stream_consumer(result, [], consumer)
    final = consumer.finish.call_args.args[0]
    assert final.startswith(code)
    if mixed:
        assert "Claim ※" in final
        assert "prose_ref" not in final
        assert "https://example.org/current" in final
        assert result["citation_delivery_diagnostics"] == {"unresolved_count": 1, "source_count": 1}
        assert result["response_transformed"] is True
    else:
        assert final == raw
        assert "citation_delivery_diagnostics" not in result
        assert not result.get("response_transformed", False)
    assert messages == original
    assert messages[-1]["content"] == raw


@pytest.mark.parametrize("marker", ["[unresolved citation: literal_ref]", "\ue200cite\ue202literal_ref\ue201"])
@pytest.mark.parametrize("template", ["``{marker}``", "> ~~~~text\r\n> {marker}\r\n> ~~~~~\r\n"])
def test_native_markdown_block_payload_keeps_exact_original_literal_code(marker, template):
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={"markdown_blocks": True}))
    raw = template.format(marker=marker)
    assert adapter._maybe_blocks(raw) == [{"type": "markdown", "text": raw}]


@pytest.mark.parametrize("marker", ["[unresolved citation: literal_ref]", "\ue200cite\ue202literal_ref\ue201"])
@pytest.mark.parametrize("markdown_blocks", [False, True])
@pytest.mark.asyncio
async def test_code_only_native_draft_is_not_falsely_replaced_or_duplicated(marker, markdown_blocks):
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter({"markdown_blocks": markdown_blocks})
    raw = "``" + marker + "``"
    started = await adapter.send_draft("C1", 7, raw, metadata=META)
    result = await adapter.send("C1", raw + " Tail", metadata=META)
    assert result.success
    assert result.message_id == started.message_id
    assert client.chat_startStream.await_args.kwargs["markdown_text"] == raw
    assert client.chat_stopStream.await_args.kwargs["markdown_text"] == " Tail"
    client.chat_postMessage.assert_not_awaited()
    if markdown_blocks:
        assert client.chat_update.await_args.kwargs["blocks"] == [{"type": "markdown", "text": raw + " Tail"}]
    else:
        client.chat_update.assert_not_awaited()
    assert not adapter._active_streams


@pytest.mark.parametrize("literal", ["[unresolved citation: literal_ref]", "\ue200cite\ue202literal_ref\ue201"])
@pytest.mark.parametrize("partial", ["[unresolved citation: prose", "\ue200cite\ue202prose"])
@pytest.mark.asyncio
async def test_native_partial_draft_matches_repaired_final_without_normalizing_literal_context(literal, partial):
    from agent.slack_citation_delivery import prepare_slack_citation_delivery
    from tests.gateway.test_slack_native_streaming import _make_adapter, META
    adapter, client = _make_adapter({"markdown_blocks": True})
    code = "Example ``" + literal + "``\n\n"
    draft = code + "Claim " + partial
    complete = partial + ("_ref]" if partial.startswith("[") else "_ref\ue201")
    messages = [_retrieval_message("https://example.org/current")]
    original = deepcopy(messages)
    final = prepare_slack_citation_delivery(code + "Claim " + complete, messages=messages).text
    started = await adapter.send_draft("C1", 7, draft, metadata=META)
    result = await adapter.send("C1", final, metadata=META)
    assert result.success
    assert result.message_id == started.message_id
    client.chat_postMessage.assert_not_awaited()
    client.chat_stopStream.assert_awaited_once()
    assert "markdown_text" not in client.chat_stopStream.await_args.kwargs
    client.chat_update.assert_awaited_once()
    assert client.chat_update.await_args.kwargs["blocks"] == [{"type": "markdown", "text": final}]
    assert final.startswith(code + "Claim ※")
    assert "prose_ref" not in final
    assert "https://example.org/current" in final
    assert messages == original
    assert not adapter._active_streams
