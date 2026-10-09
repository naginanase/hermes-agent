"""Slack's persistent native draft must keep the real consumer's frames append-only.

These are sanitized, in-memory snapshots, not production transcripts. Only the Slack
Web API is replaced: the consumer, draft delta computation and final seal are real.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter


META = {"thread_id": "100.001", "team_id": "TFAKE", "user_id": "UFAKE"}
CURSOR = " ▉"
CODE_SNAPSHOTS = [
    pytest.param(
        "Set `sample_option =", " 'on", "Set `sample_option = 'on'` then retry.",
        id="open-inline-code",
    ),
    pytest.param(
        "Run:\n```sh\nprintf 'he", "llo", "Run:\n```sh\nprintf 'hello'\n```\nDone.",
        id="open-fenced-code",
    ),
]


class FakeSlackClient:
    """Model Slack's APPEND-only stream API and ordinary replacement edits."""

    def __init__(self):
        self.calls = []
        self.messages = {}
        self.failures = {}
        self.next_id = 0

    def __getattr__(self, method):
        if not method.startswith(("chat_", "assistant_")):
            raise AttributeError(method)

        async def call(**payload):
            self.calls.append((method, dict(payload)))
            if method in self.failures:
                raise RuntimeError(self.failures[method])
            handler = {
                "chat_startStream": self._start,
                "chat_postMessage": self._start,
                "chat_appendStream": self._append,
                "chat_stopStream": self._append,
                "chat_update": self._update,
                "chat_delete": self._delete,
            }.get(method)
            ts = handler(method, payload) if handler else payload.get("ts")
            return {"ok": True, "ts": ts, "channel": payload.get("channel")}

        return call

    def _start(self, method, payload):
        self.next_id += 1
        ts = f"200.{self.next_id:03d}"
        self.messages[ts] = {
            "text": payload.get("markdown_text", payload.get("text", "")),
            "sealed": method == "chat_postMessage",
        }
        return ts

    def _append(self, method, payload):
        ts = payload["ts"]
        self.messages[ts]["text"] += payload.get("markdown_text", "")
        if method == "chat_stopStream":
            self.messages[ts]["sealed"] = True
        return ts

    def _update(self, method, payload):
        ts = payload["ts"]
        self.messages[ts].update(text=payload["text"], sealed=True)
        return ts

    def _delete(self, method, payload):
        ts = payload["ts"]
        self.messages.pop(ts, None)
        return ts

    def payloads(self, method):
        return [payload for name, payload in self.calls if name == method]


class StreamHarness:
    """Event-based driver; spies delegate to the actual consumer and Slack adapter."""

    def __init__(self, monkeypatch, *, extra=None, transport="auto", metadata=None):
        self.client = FakeSlackClient()
        self.adapter = SlackAdapter(PlatformConfig(enabled=True, extra=extra or {}))
        self.adapter._app = SimpleNamespace(client=self.client)
        self.frames = []
        self.sends = []
        self.events = asyncio.Queue()
        self.consumer = GatewayStreamConsumer(
            self.adapter, "CFAKE",
            StreamConsumerConfig(
                edit_interval=0, buffer_threshold=1, cursor=CURSOR,
                transport=transport, chat_type="channel",
            ),
            metadata=dict(META if metadata is None else metadata),
        )
        send_draft = self.adapter.send_draft
        send = self.adapter.send

        async def draft_spy(*args, **kwargs):
            self.frames.append(dict(kwargs))
            return await send_draft(*args, **kwargs)

        async def send_spy(*args, **kwargs):
            self.sends.append({"args": args, **kwargs})
            return await send(*args, **kwargs)

        monkeypatch.setattr(self.adapter, "send_draft", draft_spy)
        monkeypatch.setattr(self.adapter, "send", send_spy)
        for name in ("_push_update", "_end_segment", "_deliver_commentary"):
            self._observe(monkeypatch, name)
        self.task = asyncio.create_task(self.consumer.run())

    def _observe(self, monkeypatch, name):
        original = getattr(self.consumer, name)

        async def observed(*args, **kwargs):
            result = await original(*args, **kwargs)
            self.events.put_nowait((name, self.consumer._accumulated))
            return result

        monkeypatch.setattr(self.consumer, name, observed)

    async def wait(self, name, accumulated=None):
        async def reached():
            while True:
                event_name, text = await self.events.get()
                if event_name == name and (accumulated is None or text == accumulated):
                    return

        await asyncio.wait_for(reached(), timeout=3)

    async def push(self, text):
        expected = self.consumer._accumulated + text
        self.consumer.on_delta(text)
        await self.wait("_push_update", accumulated=expected)

    async def finish(self, final):
        self.consumer.finish(final)
        await asyncio.wait_for(self.task, timeout=3)


@asynccontextmanager
async def running_stream(monkeypatch, **kwargs):
    harness = StreamHarness(monkeypatch, **kwargs)
    try:
        yield harness
    finally:
        if not harness.task.done():
            harness.consumer.finish()
            await asyncio.wait_for(harness.task, timeout=3)


def assert_one_sealed_answer(harness, final, original_id, *, side_messages=()):
    assert len(harness.client.messages) == 1 + len(side_messages), (
        "a persistent draft must not leave a preview beside a fresh duplicate final",
        harness.client.calls,
    )
    assert harness.client.messages[original_id] == {"text": final, "sealed": True}
    assert [message["text"] for ts, message in harness.client.messages.items()
            if ts != original_id] == list(side_messages)
    assert len(harness.client.payloads("chat_postMessage")) == len(side_messages)
    assert not harness.adapter._active_streams
    assert harness.consumer.message_id == original_id
    assert harness.consumer.final_response_sent is True
    assert harness.consumer.final_content_delivered is True
    assert harness.consumer.delivered_final_matches(final) is True


@pytest.mark.parametrize("prefix,extension,final", CODE_SNAPSHOTS)
@pytest.mark.parametrize("next_partial_frame", [False, True], ids=["direct-final", "next-frame"])
@pytest.mark.asyncio
async def test_open_code_draft_finalizes_original_answer(
    monkeypatch, prefix, extension, final, next_partial_frame,
):
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        snapshots = [prefix]
        if next_partial_frame:
            await harness.push(extension)
            snapshots.append(prefix + extension)
        await harness.finish(final)

        assert_one_sealed_answer(harness, final, original_id)
        frames = [frame["content"] for frame in harness.frames]
        assert frames == snapshots  # No artificial closing ticks or configured cursor.
        assert all(later.startswith(earlier) for earlier, later in zip(frames, frames[1:]))
        assert all(final.startswith(frame) for frame in frames)
        assert len({frame["draft_id"] for frame in harness.frames}) == 1
        native_text = "".join(
            payload.get("markdown_text", "") for method, payload in harness.client.calls
            if method in ("chat_startStream", "chat_appendStream", "chat_stopStream")
        )
        assert native_text == final
        assert CURSOR not in native_text
        final_metadata = harness.sends[-1]["metadata"]
        assert final_metadata["notify"] is True
        assert not final_metadata.get("_interim_send")
        assert not final_metadata.get("expect_edits")
        assert all(frame["metadata"] == META for frame in harness.frames)


@pytest.mark.parametrize("prefix,extension,final", CODE_SNAPSHOTS)
@pytest.mark.parametrize("boundary", ["segment", "tool", "commentary", "flush"])
@pytest.mark.asyncio
async def test_boundaries_preserve_open_code_stream(
    monkeypatch, prefix, extension, final, boundary,
):
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        draft_id = harness.frames[0]["draft_id"]
        commentary = "Checking prerequisites."
        side_messages = ()
        if boundary == "commentary":
            harness.consumer.on_commentary(commentary)
            await harness.wait("_deliver_commentary")
            side_messages = (commentary,)
            assert harness.sends[-1]["metadata"]["_interim_send"] is True
            assert not harness.sends[-1]["metadata"].get("notify")
        elif boundary == "flush":
            assert await asyncio.to_thread(harness.consumer.flush_pending_sync, 3)
            await harness.wait("_end_segment")
        else:
            if boundary == "tool":
                # Slack drafts are not WeCom's mutable tool-progress overlay.
                assert harness.consumer.accepts_tool_progress is False
                harness.consumer.on_tool_progress("Tool: checking")
                harness.consumer.on_delta(None)
            else:
                harness.consumer.on_segment_break()
            await harness.wait("_end_segment")

        assert not harness.client.payloads("chat_stopStream")
        assert harness.client.messages[original_id] == {"text": prefix, "sealed": False}
        assert harness.consumer.final_response_sent is False
        await harness.push(extension)
        await harness.finish(final)

        assert_one_sealed_answer(harness, final, original_id, side_messages=side_messages)
        frames = [frame["content"] for frame in harness.frames]
        assert frames == [prefix, prefix + extension]
        assert {frame["draft_id"] for frame in harness.frames} == {draft_id}
        assert all(later.startswith(earlier) for earlier, later in zip(frames, frames[1:]))
        assert all(CURSOR not in frame for frame in frames)


@pytest.mark.parametrize("marker", ["_interim_send", "expect_edits"])
@pytest.mark.asyncio
async def test_matching_interim_or_preview_never_seals_persistent_draft(monkeypatch, marker):
    prefix = "An ordinary answer"
    final = prefix + " completed."
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        result = await harness.adapter.send(
            "CFAKE", prefix, metadata={**META, marker: True, "notify": True},
        )
        assert result.success
        assert result.message_id != original_id
        assert not harness.client.payloads("chat_stopStream")
        assert harness.client.messages[original_id] == {"text": prefix, "sealed": False}
        assert harness.adapter._active_streams
        await harness.finish(final)
        assert_one_sealed_answer(harness, final, original_id, side_messages=(prefix,))


@pytest.mark.parametrize("notify", [False, True], ids=["ordinary-notice", "notify-status"])
@pytest.mark.asyncio
async def test_unrelated_send_leaves_answer_for_consumer_final(monkeypatch, notify):
    prefix = "An ordinary answer"
    final = prefix + " completed."
    notice = "Status: running."
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        result = await harness.adapter.send("CFAKE", notice, metadata={**META, "notify": notify})
        assert result.success
        assert not harness.client.payloads("chat_stopStream")
        assert harness.adapter._active_streams
        await harness.finish(final)
        assert_one_sealed_answer(harness, final, original_id, side_messages=(notice,))
        assert harness.sends[-1]["metadata"]["notify"] is True


@pytest.mark.parametrize("route", ["feature-gate", "cached-unsupported", "unfurl", "edit", "no-thread"])
@pytest.mark.parametrize("prefix,extension,final", CODE_SNAPSHOTS)
@pytest.mark.asyncio
async def test_unavailable_native_stream_still_delivers_through_ordinary_edit(
    monkeypatch, route, prefix, extension, final,
):
    extra = {"unfurl_links": False} if route == "unfurl" else None
    metadata = {"team_id": "TFAKE", "user_id": "UFAKE"} if route == "no-thread" else META
    transport = "edit" if route == "edit" else "auto"
    async with running_stream(
        monkeypatch, extra=extra, metadata=metadata, transport=transport,
    ) as harness:
        if route == "feature-gate":
            harness.client.failures["chat_startStream"] = "feature_not_enabled"
        elif route == "cached-unsupported":
            harness.adapter._native_stream_unsupported = True
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        assert harness.sends[0]["metadata"]["expect_edits"] is True
        assert not harness.sends[0]["metadata"].get("notify")
        await harness.push(extension)
        await harness.finish(final)

        assert list(harness.client.messages) == [original_id]
        assert harness.client.messages[original_id] == {
            "text": harness.adapter.format_message(final), "sealed": True,
        }
        assert len(harness.client.payloads("chat_postMessage")) == 1
        assert harness.client.payloads("chat_update")
        assert not harness.client.payloads("chat_appendStream")
        assert not harness.adapter._active_streams
        assert harness.consumer.message_id == original_id
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.delivered_final_matches(final) is True
        assert CURSOR not in harness.client.messages[original_id]["text"]
        if route == "feature-gate":
            assert harness.adapter._native_stream_unsupported is True
        if route == "unfurl":
            assert harness.client.payloads("chat_postMessage")[0]["unfurl_links"] is False


@pytest.mark.asyncio
async def test_failed_seal_updates_original_answer_instead_of_duplicating(monkeypatch):
    prefix = "Set `sample_option ="
    final = "Set `sample_option = 'on'` then retry."
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        original_id = next(iter(harness.client.messages))
        harness.client.failures["chat_stopStream"] = "seal failed"
        await harness.finish(final)
        assert_one_sealed_answer(harness, final, original_id)
        assert len(harness.client.payloads("chat_stopStream")) == 1  # Never retry an appending tail.
        assert harness.client.payloads("chat_update")[0]["ts"] == original_id


@pytest.mark.asyncio
async def test_uncommittable_stream_falls_back_so_completed_answer_is_not_lost(monkeypatch):
    prefix = "Set `sample_option ="
    final = "Set `sample_option = 'on'` then retry."
    async with running_stream(monkeypatch) as harness:
        await harness.push(prefix)
        harness.client.failures.update(chat_stopStream="seal failed", chat_update="edit failed")
        await harness.finish(final)
        # A fresh post is loss-safe ONLY after both in-place commit routes failed.
        assert len(harness.client.payloads("chat_postMessage")) == 1
        assert harness.client.messages[harness.consumer.message_id] == {"text": final, "sealed": True}
        assert not harness.adapter._active_streams
        assert harness.consumer.final_response_sent is True
        assert harness.consumer.delivered_final_matches(final) is True
