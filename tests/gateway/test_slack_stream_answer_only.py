"""Authoritative answer-only finals must reconcile the owning persistent draft.

Sanitized, offline regressions using the real consumer and Slack adapter. The
shared FakeSlackClient models only Slack's append-only transport and edits.
"""

from __future__ import annotations

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.test_slack_stream_consumer_prefix import (
    FakeSlackClient,
    META,
    assert_one_sealed_answer,
    running_stream,
)


PREAMBLE = "Checking the prerequisite."
ANSWER = "The result is ready."
FULL_PREFIX = PREAMBLE + "\n\n" + ANSWER


async def _push_through_tool(harness):
    await harness.push(PREAMBLE)
    original_id = next(iter(harness.client.messages))
    harness.consumer.on_delta(None)
    await harness.wait("_end_segment")
    await harness.push("\n\n" + ANSWER)
    return original_id


@pytest.mark.parametrize("persistent", [False, True], ids=["segmented-control", "persistent-draft"])
@pytest.mark.asyncio
async def test_answer_only_final_seals_owning_draft_without_duplicate(monkeypatch, persistent):
    monkeypatch.setattr(SlackAdapter, "draft_stream_is_message", persistent)
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        await harness.finish(ANSWER)

        final_id = harness.consumer.message_id
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(ANSWER) is True
        # Legacy segmented delivery retains the raw paragraph separator; its
        # established equality contract ignores only surrounding whitespace.
        final_visible = ANSWER if persistent else "\n\n" + ANSWER
        assert harness.client.messages[final_id] == {"text": final_visible, "sealed": True}
        if not persistent:
            assert final_id != original_id
        actual = {
            "messages": harness.client.messages,
            "fresh_posts": len(harness.client.payloads("chat_postMessage")),
            "stopped_ids": [p["ts"] for p in harness.client.payloads("chat_stopStream")],
            "active_stream_keys": list(harness.adapter._active_streams),
            "final_message_id": final_id,
        }
        expected_messages = {original_id: {"text": ANSWER, "sealed": True}} if persistent else {
            original_id: {"text": PREAMBLE, "sealed": True},
            final_id: {"text": final_visible, "sealed": True},
        }
        assert actual == {
            "messages": expected_messages,
            "fresh_posts": 0,
            "stopped_ids": [original_id] if persistent else [original_id, final_id],
            "active_stream_keys": [],
            "final_message_id": original_id if persistent else final_id,
        }, harness.client.calls


@pytest.mark.asyncio
async def test_full_prefix_final_still_seals_original_persistent_draft(monkeypatch):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        await harness.finish(FULL_PREFIX)
        assert_one_sealed_answer(harness, FULL_PREFIX, original_id)


@pytest.mark.parametrize("notify", [False, True], ids=["ordinary-send", "notify-send"])
@pytest.mark.asyncio
async def test_answer_suffix_without_final_provenance_cannot_capture_live_draft(monkeypatch, notify):
    # Text overlap and notify=True are not proof that a send owns this turn.
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        result = await harness.adapter.send("CFAKE", ANSWER, metadata={**META, "notify": notify})
        assert result.success
        assert result.message_id != original_id
        assert not harness.client.payloads("chat_stopStream")
        assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}
        assert harness.adapter._active_streams
        await harness.finish(FULL_PREFIX)
        assert_one_sealed_answer(harness, FULL_PREFIX, original_id, side_messages=(ANSWER,))


@pytest.mark.parametrize("identity", ["stale", "none", "string", "float", "bool", "mapping", "zero", "negative"])
@pytest.mark.parametrize("content", [ANSWER, FULL_PREFIX], ids=["rewritten", "matching-prefix"])
@pytest.mark.asyncio
async def test_invalid_explicit_final_identity_cannot_fall_back_to_text_guessing(
    monkeypatch, identity, content,
):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        draft_id = harness.consumer._draft_id
        assert type(draft_id) is int
        invalid = {
            "stale": draft_id + 1, "none": None, "string": str(draft_id),
            "float": float(draft_id), "bool": True, "mapping": {},
            "zero": 0, "negative": -1,
        }[identity]
        result = await harness.adapter.send(
            "CFAKE", content,
            metadata={**META, "notify": True, "_slack_final_draft_id": invalid},
        )
        assert result.success
        assert result.message_id != original_id
        assert not harness.client.payloads("chat_stopStream")
        assert not harness.client.payloads("chat_update")
        assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}
        await harness.finish(ANSWER)
        assert_one_sealed_answer(harness, ANSWER, original_id, side_messages=(content,))


@pytest.mark.asyncio
async def test_stale_consumer_final_cannot_seal_newer_draft_with_identical_answer(monkeypatch):
    async with running_stream(monkeypatch) as harness:
        await _push_through_tool(harness)
        draft_id = harness.consumer._draft_id
        assert type(draft_id) is int
        newer_id = draft_id + 1
        newer = await harness.adapter.send_draft("CFAKE", newer_id, ANSWER, metadata=META)
        stop_count = len(harness.client.payloads("chat_stopStream"))
        await harness.finish(ANSWER)
        assert harness.consumer.message_id != newer.message_id
        assert harness.client.messages[newer.message_id] == {"text": ANSWER, "sealed": False}
        assert len(harness.client.payloads("chat_stopStream")) == stop_count
        assert next(iter(harness.adapter._active_streams.values()))["draft_id"] == newer_id
        assert harness.consumer.delivered_final_matches(ANSWER) is True


@pytest.mark.asyncio
async def test_split_final_chunks_never_claim_full_authoritative_rewrite_intent(monkeypatch):
    # Opt-out adapters still use the legacy consumer split path; no partial chunk
    # may inherit whole-body ownership merely because finish() supplied a final.
    monkeypatch.setattr(SlackAdapter, "draft_final_handles_overflow", False)
    async with running_stream(monkeypatch) as harness:
        await harness.push(PREAMBLE)
        harness.adapter.MAX_MESSAGE_LENGTH = 600
        harness.consumer._safe_limit = 500
        final = ("complete answer line\n" * 70) + "END_REQUIRED"
        await harness.finish(final)
        assert len(harness.sends) > 1
        assert all("_slack_final_draft_id" not in (send["metadata"] or {}) for send in harness.sends)
        delivered = "".join(message["text"] for message in harness.client.messages.values())
        assert delivered.count("complete answer line") == 70
        assert "END_REQUIRED" in delivered
        assert harness.consumer.final_content_delivered is True
        assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("identity", ["missing", "string", "float", "bool"])
@pytest.mark.asyncio
async def test_malformed_active_draft_identity_denies_owned_final(monkeypatch, identity):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        stream = next(iter(harness.adapter._active_streams.values()))
        draft_id = stream.pop("draft_id")
        if identity != "missing":
            stream["draft_id"] = {"string": str(draft_id), "float": float(draft_id), "bool": True}[identity]
        result = await harness.adapter.send(
            "CFAKE", FULL_PREFIX,
            metadata={**META, "_slack_final_draft_id": draft_id},
        )
        assert result.success and result.message_id != original_id
        assert not harness.client.payloads("chat_stopStream")
        stream["draft_id"] = draft_id
        await harness.finish(ANSWER)
        assert_one_sealed_answer(harness, ANSWER, original_id, side_messages=(FULL_PREFIX,))


@pytest.mark.parametrize("route", ["thread", "team", "chat"])
@pytest.mark.asyncio
async def test_owned_final_never_captures_stream_in_different_route(monkeypatch, route):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        metadata = dict(META)
        chat_id = "CFAKE"
        if route == "chat":
            chat_id = "COTHER"
        else:
            metadata[{"thread": "thread_id", "team": "team_id"}[route]] = {
                "thread": "100.999", "team": "TOTHER",
            }[route]
        draft_id = harness.consumer._draft_id
        assert type(draft_id) is int
        other = await harness.adapter.send_draft(chat_id, draft_id + 1, ANSWER, metadata=metadata)
        result = await harness.adapter.send(
            chat_id, ANSWER, metadata={**metadata, "_slack_final_draft_id": draft_id},
        )
        assert result.success and result.message_id not in (original_id, other.message_id)
        assert not harness.client.payloads("chat_stopStream")
        assert len(harness.adapter._active_streams) == 2
        await harness.finish(ANSWER)
        assert harness.consumer.message_id == original_id
        assert harness.client.messages[original_id] == {"text": ANSWER, "sealed": True}
        assert harness.client.messages[other.message_id] == {"text": ANSWER, "sealed": False}
        assert list(harness.adapter._active_streams) == [harness.adapter._stream_key(chat_id, metadata)]


@pytest.mark.parametrize("marker", ["_interim_send", "expect_edits"])
@pytest.mark.asyncio
async def test_interim_or_preview_denies_even_valid_owned_final(monkeypatch, marker):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        result = await harness.adapter.send(
            "CFAKE", ANSWER, metadata={
                **META, marker: True, "notify": True,
                "_slack_final_draft_id": harness.consumer._draft_id,
            },
        )
        assert result.success and result.message_id != original_id
        assert not harness.client.payloads("chat_stopStream")
        await harness.finish(ANSWER)
        assert_one_sealed_answer(harness, ANSWER, original_id, side_messages=(ANSWER,))


@pytest.mark.parametrize("marker", ["_interim_send", "expect_edits"])
@pytest.mark.asyncio
async def test_consumer_denies_interim_or_preview_before_optional_hook(monkeypatch, marker):
    def forbidden_hook(*args, **kwargs):
        pytest.fail("interim/preview must be denied before the ownership hook")

    monkeypatch.setattr(SlackAdapter, "draft_final_metadata", forbidden_hook)
    async with running_stream(monkeypatch, metadata={**META, marker: True}) as harness:
        original_id = await _push_through_tool(harness)
        await harness.finish(ANSWER)
        assert "_slack_final_draft_id" not in harness.sends[-1]["metadata"]
        assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}


@pytest.mark.parametrize("final", [ANSWER, FULL_PREFIX], ids=["answer-only", "legacy-prefix"])
@pytest.mark.asyncio
async def test_missing_optional_hook_preserves_legacy_text_relation(monkeypatch, final):
    monkeypatch.setattr(SlackAdapter, "draft_final_metadata", None)
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        await harness.finish(final)
        assert harness.sends[-1]["metadata"] == {**META, "notify": True}
        if final == FULL_PREFIX:
            assert_one_sealed_answer(harness, final, original_id)
        else:
            assert harness.consumer.message_id != original_id
            assert not harness.client.payloads("chat_stopStream")
            assert harness.client.messages[original_id] == {"text": FULL_PREFIX, "sealed": False}


@pytest.mark.parametrize("final_kind", ["answer-only", "full-prefix"])
@pytest.mark.asyncio
async def test_multiple_tool_boundaries_keep_one_draft_for_authoritative_final(monkeypatch, final_kind):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        harness.consumer.on_delta(None)
        await harness.wait("_end_segment")
        await harness.push("\n\nSecond check completed.")
        harness.consumer.on_delta(None)
        await harness.wait("_end_segment")
        await harness.push("\n\n" + ANSWER)
        final = ANSWER if final_kind == "answer-only" else harness.consumer._accumulated
        await harness.finish(final)
        assert_one_sealed_answer(harness, final, original_id)
        assert len({frame["draft_id"] for frame in harness.frames}) == 1
        assert all(later["content"].startswith(earlier["content"])
                   for earlier, later in zip(harness.frames, harness.frames[1:]))
        assert harness.consumer.metadata == META
        assert all("_slack_final_draft_id" not in payload for _, payload in harness.client.calls)


@pytest.mark.parametrize("failure", ["stop", "edit", "both", "all"])
@pytest.mark.asyncio
async def test_owned_rewrite_preserves_existing_loss_safe_failure_fallback(monkeypatch, failure):
    async with running_stream(monkeypatch) as harness:
        original_id = await _push_through_tool(harness)
        if failure in ("stop", "both", "all"):
            harness.client.failures["chat_stopStream"] = "seal unavailable"
        if failure in ("edit", "both", "all"):
            harness.client.failures["chat_update"] = "edit unavailable"
        if failure == "all":
            harness.client.failures["chat_postMessage"] = "post unavailable"
        await harness.finish(ANSWER)
        stops = harness.client.payloads("chat_stopStream")
        assert stops and all("markdown_text" not in payload for payload in stops)
        assert not harness.adapter._active_streams
        if failure == "stop":
            assert len(stops) == 2  # Empty-tail sealing alone may be retried.
            assert_one_sealed_answer(harness, ANSWER, original_id)
        elif failure == "all":
            assert harness.consumer.final_content_delivered is False
            assert harness.consumer.final_response_sent is False
            assert harness.consumer.delivered_final_matches(ANSWER) is False
        else:
            assert harness.consumer.message_id != original_id
            assert harness.client.messages[harness.consumer.message_id] == {"text": ANSWER, "sealed": True}
            assert len(harness.client.payloads("chat_postMessage")) == 1
            assert harness.consumer.final_content_delivered is True
            assert harness.consumer.delivered_final_matches(ANSWER) is True


@pytest.mark.parametrize("ending", ["bare-finish", "cancel", "stale", "error"])
@pytest.mark.asyncio
async def test_non_authoritative_turn_death_never_emits_rewrite_intent(monkeypatch, ending):
    def forbidden_hook(*args, **kwargs):
        pytest.fail("turn death has no authoritative-final rewrite ownership")

    monkeypatch.setattr(SlackAdapter, "draft_final_metadata", forbidden_hook)
    async with running_stream(monkeypatch) as harness:
        await _push_through_tool(harness)
        if ending == "cancel":
            harness.task.cancel()
            await harness.task
        else:
            if ending == "stale":
                harness.consumer._run_still_current = lambda: False
            elif ending == "error":
                def fail_tick(tick):
                    raise RuntimeError("offline consumer failure")
                monkeypatch.setattr(harness.consumer, "_should_edit", fail_tick)
            await harness.finish(None)
        assert all("_slack_final_draft_id" not in (send["metadata"] or {}) for send in harness.sends)


@pytest.mark.parametrize("finalize,is_turn_final", [(False, True), (True, False)])
@pytest.mark.asyncio
async def test_preview_and_segment_or_cancel_finalize_deny_adopted_intent(
    monkeypatch, finalize, is_turn_final,
):
    async with running_stream(monkeypatch) as harness:
        await _push_through_tool(harness)
        harness.consumer._adopt_final_text(ANSWER)
        metadata = harness.consumer._metadata_for_send(final=finalize, is_turn_final=is_turn_final)
        assert "_slack_final_draft_id" not in (metadata or {})
        await harness.finish(ANSWER)


@pytest.mark.parametrize("platform", ["slack", "telegram"])
@pytest.mark.asyncio
async def test_relay_authoritative_final_metadata_does_not_gain_native_slack_keys(platform):
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter
    from tests.gateway.relay.test_relay_stream_semantics_gating import RecordingTransport

    adapter, _ = _connected_adapter(
        platform=platform, supported_ops=("send", "edit", "typing", "draft"),
    )
    transport = RecordingTransport()
    adapter._transport = transport
    metadata = {"thread_id": "thread.1", "reply_to_message_id": "event.1"}
    consumer = GatewayStreamConsumer(adapter, "C1", metadata=metadata)
    consumer._use_draft_streaming = True
    consumer._bump_draft_id()
    await consumer._send_draft_frame(PREAMBLE)
    consumer.finish(ANSWER)
    assert consumer._drain_queue().got_done
    await consumer._send_or_edit(ANSWER, finalize=True, is_turn_final=True)
    assert consumer._metadata_for_send(final=True, is_turn_final=True) == {**metadata, "notify": True}
    assert all("_slack_final_draft_id" not in (frame.get("metadata") or {}) for frame in transport.ops)
    assert consumer.metadata == metadata


@pytest.mark.asyncio
async def test_instance_mock_attribute_cannot_opt_adapter_into_final_ownership():
    from unittest.mock import AsyncMock, MagicMock
    from gateway.platforms.base import SendResult

    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    adapter.draft_stream_is_message = True
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""), metadata=META)
    consumer._use_draft_streaming = True
    consumer._bump_draft_id()
    consumer.on_delta(PREAMBLE)
    consumer.finish(ANSWER)
    assert consumer._drain_queue().got_done
    assert await consumer._first_send(ANSWER, finalize=True, is_turn_final=True)
    assert adapter.send.await_args.kwargs["metadata"] == {**META, "notify": True}
    adapter.draft_final_metadata.assert_not_called()


@pytest.mark.parametrize("fail_continuation", [False, True], ids=["full-tail", "failed-tail"])
@pytest.mark.asyncio
async def test_owned_oversized_replacement_is_bounded_and_never_claims_lost_tail(
    monkeypatch, fail_continuation,
):
    from types import SimpleNamespace
    from gateway.config import PlatformConfig

    client = FakeSlackClient()
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter._app = SimpleNamespace(client=client)
    monkeypatch.setattr(adapter, "MAX_MESSAGE_LENGTH", 600)
    consumer = GatewayStreamConsumer(adapter, "CFAKE", metadata=dict(META))
    consumer._use_draft_streaming = True
    consumer._bump_draft_id()
    consumer.on_delta(PREAMBLE)
    consumer._drain_queue()
    original = await adapter.send_draft("CFAKE", consumer._draft_id, PREAMBLE, metadata=META)
    final = "complete answer line\n" * 70 + "END_REQUIRED"
    consumer.finish(final)
    assert consumer._drain_queue().got_done
    if fail_continuation:
        client.failures["chat_postMessage"] = "continuation unavailable"
    result = await adapter.send(
        "CFAKE", final, metadata=consumer._metadata_for_send(final=True, is_turn_final=True),
    )
    assert not adapter._active_streams
    assert all("markdown_text" not in payload for payload in client.payloads("chat_stopStream"))
    assert client.payloads("chat_update")[0]["ts"] == original.message_id
    assert all(len(message["text"]) <= 600 for message in client.messages.values())
    delivered = "".join(message["text"] for message in client.messages.values())
    if fail_continuation:
        assert result.success is False
        assert "END_REQUIRED" not in delivered
    else:
        assert result.success is True and result.message_id == original.message_id
        assert delivered.count("complete answer line") == 70
        assert "END_REQUIRED" in delivered
