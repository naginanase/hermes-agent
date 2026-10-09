"""Completed TurnRunner finals preserve text and close the owning native draft.

Offline integration: real runner final caller, consumer and Slack adapter; only
Slack's append/update transport is represented by the shared fake client.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from tests.gateway.test_slack_stream_answer_only import FULL_PREFIX, _push_through_tool
from tests.gateway.test_slack_stream_consumer_prefix import (
    CURSOR,
    assert_one_sealed_answer,
    running_stream,
)


async def _finish_completed(harness, final):
    ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.SLACK), result_holder=[None])
    result = {"completed": True, "final_response": final, "messages": []}
    TurnRunner(None, ctx)._finish_stream_consumer(result, [], harness.consumer)
    await asyncio.wait_for(harness.task, timeout=3)
    assert ctx.result_holder[0] is result
    assert result["final_response"] == final


@pytest.mark.parametrize("final", [
    "Please wait…", "Literal glyph: ▌", "Three periods...", "Literal cursor:" + CURSOR,
    "Trailing whitespace…  \n", "Code: `▌…`", "Code:\n```\n▌…  \n```", "▌", CURSOR,
])
@pytest.mark.asyncio
async def test_completed_answer_only_preserves_literal_cursor_characters(monkeypatch, final):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        await _finish_completed(harness, final)
        assert_one_sealed_answer(harness, final, original_id)
        assert [p["ts"] for p in harness.client.payloads("chat_stopStream")] == [original_id]
        assert harness.client.payloads("chat_update")[0]["text"] == final


@pytest.mark.asyncio
async def test_completed_answer_only_overflow_replaces_and_closes_original_draft(monkeypatch):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = "complete answer line\n" * 70 + "END_REQUIRED"
        await _finish_completed(harness, final)
        assert harness.client.messages[original_id]["sealed"] is True
        assert not harness.adapter._active_streams
        assert [p["ts"] for p in harness.client.payloads("chat_stopStream")] == [original_id]
        assert harness.client.payloads("chat_update")[0]["ts"] == original_id
        assert harness.consumer.message_id == original_id
        chunks = harness.adapter._format_chunks(final)
        assert [m["text"] for m in harness.client.messages.values()] == chunks
        assert all(len(m["text"]) <= 600 for m in harness.client.messages.values())
        assert len(harness.client.payloads("chat_postMessage")) == len(chunks) - 1
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("suffix", ["…", " ▌", CURSOR, "  \n", "\nCode: `▌…`"])
@pytest.mark.asyncio
async def test_completed_full_prefix_preserves_exact_authoritative_suffix(monkeypatch, suffix):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        final = FULL_PREFIX + suffix
        await _finish_completed(harness, final)
        assert_one_sealed_answer(harness, final, original_id)


@pytest.mark.asyncio
async def test_completed_full_prefix_overflow_replaces_instead_of_reposting_prefix(monkeypatch):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = FULL_PREFIX + "\n" + "complete answer line\n" * 70 + "END_REQUIRED…"
        await _finish_completed(harness, final)
        assert all("markdown_text" not in p for p in harness.client.payloads("chat_stopStream"))
        assert harness.consumer.message_id == original_id
        assert [m["text"] for m in harness.client.messages.values()] == harness.adapter._format_chunks(final)
        assert all(len(m["text"]) <= 600 and m["sealed"] for m in harness.client.messages.values())
        assert not harness.adapter._active_streams
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("full_prefix", [False, True], ids=["answer-only", "full-prefix"])
@pytest.mark.parametrize("lines,limit,safe", [
    (20, 600, 500), (25, 600, 500), (30, 600, 500), (100, 600, 500),
    (2100, 39000, 38800),
], ids=["below-consumer", "consumer-only-overflow", "adapter-overflow", "many-chunks", "native-limit"])
@pytest.mark.asyncio
async def test_completed_whole_final_is_bounded_at_each_length(monkeypatch, full_prefix, lines, limit, safe):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = limit
        harness.consumer._safe_limit = safe
        body = "complete answer line\n" * lines + "END_REQUIRED…"
        final = FULL_PREFIX + "\n" + body if full_prefix else body
        await _finish_completed(harness, final)
        expected = ([final] if full_prefix and len(final) <= limit
                    else harness.adapter._format_chunks(final))
        assert [m["text"] for m in harness.client.messages.values()] == expected
        assert all(len(m["text"]) <= limit and m["sealed"] for m in harness.client.messages.values())
        assert not harness.adapter._active_streams
        assert harness.consumer.message_id == original_id
        assert len(harness.sends) == 1
        assert harness.sends[0]["content"] == final
        assert harness.sends[0]["metadata"]["_slack_final_draft_id"] == harness.consumer._draft_id
        assert all("_slack_final_draft_id" not in p for _, p in harness.client.calls)
        delivered = "".join(m["text"] for m in harness.client.messages.values())
        assert delivered.count("complete answer line") == lines
        assert "END_REQUIRED…" in delivered
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("failure", ["stop", "update", "both", "continuation-first", "continuation-later", "all"])
@pytest.mark.asyncio
async def test_completed_overflow_failure_flags_and_no_consumer_retry(monkeypatch, failure):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = "complete answer line\n" * 100 + "END_REQUIRED…"
        if failure in ("stop", "both", "all"):
            harness.client.failures["chat_stopStream"] = "stop unavailable"
        if failure in ("update", "both", "all"):
            harness.client.failures["chat_update"] = "update unavailable"
        if failure == "all":
            harness.client.failures["chat_postMessage"] = "post unavailable"
        if failure.startswith("continuation"):
            post = harness.client.chat_postMessage
            count = 0
            fail_at = 1 if failure == "continuation-first" else 2

            async def fail_continuation(**kwargs):
                nonlocal count
                count += 1
                if count == fail_at:
                    harness.client.failures["chat_postMessage"] = "continuation unavailable"
                return await post(**kwargs)

            monkeypatch.setattr(harness.client, "chat_postMessage", fail_continuation)
        await _finish_completed(harness, final)
        assert not harness.adapter._active_streams
        stops = harness.client.payloads("chat_stopStream")
        assert len(stops) == (2 if failure in ("stop", "both", "all") else 1)
        assert all("markdown_text" not in p for p in stops)
        assert len(harness.sends) == 1  # No second full body after a partially ACKed replacement.
        delivered = "".join(m["text"] for m in harness.client.messages.values())
        success = failure in ("stop", "update", "both")
        assert harness.consumer.final_response_sent is success
        assert harness.consumer.final_content_delivered is success
        assert harness.consumer.delivered_final_matches(final) is success
        if success:
            assert "END_REQUIRED…" in delivered
            assert delivered.count("complete answer line") == 100
        else:
            assert "END_REQUIRED…" not in delivered
        if failure.startswith("continuation"):
            expected = harness.adapter._format_chunks(final)[:fail_at]
            assert [m["text"] for m in harness.client.messages.values()] == expected
            assert len(harness.client.payloads("chat_postMessage")) == fail_at
        if failure == "stop":
            assert harness.consumer.message_id == original_id
            assert harness.client.messages[original_id]["sealed"] is True


@pytest.mark.parametrize("final", ["Please wait…", "Literal glyph: ▌", "Literal cursor:" + CURSOR])
@pytest.mark.asyncio
async def test_missing_ownership_hook_keeps_glyph_send_ordinary_and_draft_open(monkeypatch, final):
    from plugins.platforms.slack.adapter import SlackAdapter

    monkeypatch.setattr(SlackAdapter, "draft_final_metadata", None)
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        await _finish_completed(harness, final)
        assert harness.consumer.message_id != original_id
        assert harness.client.messages[harness.consumer.message_id] == {"text": final, "sealed": True}
        assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}
        assert not harness.client.payloads("chat_stopStream")
        assert not harness.client.payloads("chat_update")
        assert all("_slack_final_draft_id" not in s["metadata"] for s in harness.sends)


@pytest.mark.asyncio
async def test_short_raw_final_with_formatted_overflow_does_not_retry_failed_continuation(monkeypatch):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = "&" * 150 + "END_REQUIRED…"
        expected = harness.adapter._format_chunks(final)
        assert len(final) < harness.consumer._safe_limit
        assert len(expected) > 1
        post = harness.client.chat_postMessage
        calls = 0

        async def fail_once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                harness.client.failures["chat_postMessage"] = "transient continuation failure"
            try:
                return await post(**kwargs)
            finally:
                harness.client.failures.pop("chat_postMessage", None)

        monkeypatch.setattr(harness.client, "chat_postMessage", fail_once)
        await _finish_completed(harness, final)
        assert len(harness.sends) == 1
        assert harness.client.messages == {original_id: {"text": expected[0], "sealed": True}}
        assert not harness.adapter._active_streams
        assert harness.consumer.final_response_sent is False
        assert harness.consumer.final_content_delivered is False
        assert harness.consumer.delivered_final_matches(final) is False


@pytest.mark.parametrize("size", [500, 501, 600, 601])
@pytest.mark.asyncio
async def test_whole_final_limit_boundaries_keep_original_identity(monkeypatch, size):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        tail = "END_REQUIRED…"
        final = "x" * (size - len(tail)) + tail
        await _finish_completed(harness, final)
        assert harness.consumer.message_id == original_id
        assert [m["text"] for m in harness.client.messages.values()] == harness.adapter._format_chunks(final)
        assert all(len(m["text"]) <= 600 and m["sealed"] for m in harness.client.messages.values())
        assert not harness.adapter._active_streams
        assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("denial", ["missing-hook", "interim", "preview"])
@pytest.mark.asyncio
async def test_overflow_contract_requires_final_ownership_not_just_class_opt_in(monkeypatch, denial):
    from plugins.platforms.slack.adapter import SlackAdapter
    from tests.gateway.test_slack_stream_consumer_prefix import META

    metadata = dict(META)
    if denial == "missing-hook":
        monkeypatch.setattr(SlackAdapter, "draft_final_metadata", None)
    else:
        metadata[{"interim": "_interim_send", "preview": "expect_edits"}[denial]] = True

        def forbidden_hook(*args, **kwargs):
            pytest.fail("interim/preview must be refused before the final ownership hook")

        monkeypatch.setattr(SlackAdapter, "draft_final_metadata", forbidden_hook)
    async with running_stream(monkeypatch, metadata=metadata) as harness:
        original_id = await _push_through_tool(harness)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = "complete answer line\n" * 70 + "END_REQUIRED…"
        await _finish_completed(harness, final)
        assert len(harness.sends) > 1
        assert all("_slack_final_draft_id" not in s["metadata"] for s in harness.sends)
        assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}
        assert not harness.client.payloads("chat_stopStream")


@pytest.mark.asyncio
async def test_overflow_owned_by_stale_consumer_cannot_close_newer_draft(monkeypatch):
    from tests.gateway.test_slack_stream_consumer_prefix import META

    async with running_stream(monkeypatch) as harness:
        await _push_through_tool(harness)
        draft_id = harness.consumer._draft_id
        newer = await harness.adapter.send_draft("CFAKE", draft_id + 1, "Newer turn", metadata=META)
        stops = len(harness.client.payloads("chat_stopStream"))
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = "complete answer line\n" * 70 + "END_REQUIRED…"
        await _finish_completed(harness, final)
        assert harness.client.messages[newer.message_id] == {"text": "Newer turn", "sealed": False}
        assert len(harness.client.payloads("chat_stopStream")) == stops
        assert next(iter(harness.adapter._active_streams.values()))["draft_id"] == draft_id + 1
        assert harness.consumer.message_id != newer.message_id
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(final) is True
        delivered = "".join(m["text"] for m in harness.client.messages.values())
        assert delivered.count("complete answer line") == 70
        assert "END_REQUIRED…" in delivered


@pytest.mark.asyncio
async def test_unowned_native_draft_cursor_cleanup_remains_legacy(monkeypatch):
    from tests.gateway.test_slack_stream_consumer_prefix import META

    async with running_stream(monkeypatch) as harness:
        draft = await harness.adapter.send_draft("CFAKE", 123, "Draft body ▌", metadata=META)
        assert harness.client.messages[draft.message_id]["text"] == "Draft body"
        result = await harness.adapter.send("CFAKE", "Draft body ▌", metadata=META)
        assert result.success and result.message_id == draft.message_id
        assert harness.client.messages[draft.message_id] == {"text": "Draft body", "sealed": True}
        assert not harness.adapter._active_streams
        await harness.finish(None)
