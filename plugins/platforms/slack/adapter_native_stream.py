"""Native Slack turn-final reconciliation and bounded in-place replacements.

SlackAdapter inherits this phase; draft/state ownership stays on the adapter.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import SendResult

logger = logging.getLogger(f"{__package__}.adapter" if __package__ else "adapter")


class SlackNativeFinalizationMixin:
    """Finalize native streams without duplicating a successfully delivered answer."""

    # Only this opt-in transport can bound a whole owned final before consumer splitting.
    draft_final_handles_overflow = True

    def draft_final_metadata(self, *, draft_id: int) -> Dict[str, Any]:
        """Private provenance for this consumer's authoritative turn-final send."""
        return {"_slack_final_draft_id": draft_id}

    async def _try_finalize_stream(
        self, chat_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SendResult]:
        """Seal the active native stream if ``content`` is its final text: SendResult when the
        stream IS the final message; None when unrelated (interim commentary), leaving it open.

        Invariant (duplicate-final class): a successfully streamed answer is NEVER posted a
        second time. A fresh post happens only when the streamed message cannot hold the final
        (a failed seal or required rewrite cannot be repaired in place), or when the content
        is unrelated to the stream."""
        md = metadata or {}
        # Streaming contract: interim sends (commentary, segment-tail flushes) and editable
        # previews are never the turn final, whatever their text looks like.
        if md.get("_interim_send") or md.get("expect_edits"):
            return None
        key = self._stream_key(chat_id, md)
        if key is None:
            return None
        stream = self._active_streams.get(key)
        if stream is None:
            return None
        owned_final = "_slack_final_draft_id" in md
        if owned_final:
            draft_id = md["_slack_final_draft_id"]
            # Explicit but invalid/stale ownership must not fall through to the
            # legacy text classifier and accidentally seal a different turn.
            if (type(draft_id) is not int or draft_id <= 0
                    or type(stream.get("draft_id")) is not int
                    or draft_id != stream["draft_id"]):
                return None
        # Ownership validates an authoritative body, not a cursor-suffixed draft.
        text = content if owned_final else self._strip_stream_cursor(content)
        sent = stream.get("sent", "")
        from agent.slack_citation_delivery import (
            prepare_slack_citation_delivery, slack_citation_draft_body,
        )
        visible = prepare_slack_citation_delivery(text).text
        if owned_final:
            # An owned body must match exactly, not merely after draft/whitespace
            # normalization. Only a bounded, nonblank exact suffix is safe to append.
            delta = visible[len(sent):] if visible.startswith(sent) else None
            replace = (delta is None or len(visible) > self.MAX_MESSAGE_LENGTH
                       or bool(delta) and not delta.strip())
            return await self._commit_stream(
                key, stream, visible, metadata, delta="" if replace else delta or "", replace=replace)
        kind, delta = self._stream_relation(sent, text)
        sent_body = slack_citation_draft_body(sent)
        if kind == "unrelated" and sent_body is not None and visible == text:
            # The gateway may already have repaired the final with verified sources.
            # Normalize complete AND trailing partial markers, not a diagnostic footer.
            kind, delta = self._stream_relation(sent_body, text)
        if kind == "unrelated":
            # A turn-final that only restyled the streamed answer (mrkdwn conversion turned
            # ``*Done:*`` into ``_Done:_``) replaces it in place so the thread holds ONE answer.
            # Anything else — including mid-turn notify replies (/status, /approve, clarify
            # answers) that share the thread key — is not this stream's final: leave the stream
            # open and let send() post it fresh.
            if not (md.get("notify") and text.strip() and sent.strip()
                    and self._mrkdwn_insensitive(text) == self._mrkdwn_insensitive(sent)):
                return None
            return await self._commit_stream(key, stream, visible, metadata, replace=True)
        # Match the original draft/final before rendering. stopStream only appends:
        # a marker already sent in a native frame must be replaced via chat.update.
        if sent_body is not None:
            return await self._commit_stream(key, stream, visible, metadata, replace=True)
        if visible != text:
            text = visible
            kind, delta = self._stream_relation(sent, text)
            if kind == "unrelated":
                # A partial marker was already streamed; rendering removed that prefix.
                return await self._commit_stream(key, stream, text, metadata, replace=True)
        if kind == "extends" and len(delta) > self.MAX_MESSAGE_LENGTH:
            # Tail too large for one append: close the stream on what is visible and let the
            # normal split path deliver the full final.
            self._active_streams.pop(key, None)
            await self._seal_stream(key, stream)
            return None
        return await self._commit_stream(key, stream, text, metadata, delta=delta)

    async def _commit_stream(
        self, key: Tuple[str, str, str], stream: Dict[str, Any], text: str,
        metadata: Optional[Dict[str, Any]], delta: str = "", replace: bool = False,
    ) -> Optional[SendResult]:
        """Close ``stream`` as the turn's final message holding ``text``: SendResult on success,
        None when send() must post the final fresh (a duplicate beats a lost answer).

        Bounded: stopStream (appending ``delta``), one retry ONLY when nothing is appended
        (``markdown_text`` APPENDS: a first attempt that landed server-side but raised here would
        repeat the tail), then one idempotent in-place ``chat.update`` carrying the full text —
        always when ``replace`` (the content changed) or the seal failed, otherwise only to apply
        a rich Block Kit layout (non-fatal: the streamed markdown stands). Oversized replacements
        update the first bounded chunk and post every continuation before reporting success."""
        chat_id, ts = key[1], stream["ts"]
        self._active_streams.pop(key, None)
        sealed = await self._seal_stream(key, stream, delta=delta)
        if not sealed and not (delta and delta.strip()):
            sealed = await self._seal_stream(key, stream)
        if replace:
            chunks = self._format_chunks(text)
            if len(chunks) > 1:
                return await self._replace_stream_chunks(key, stream, chunks, metadata)
        if sealed and not replace and not self._maybe_blocks(text):
            committed = True
        else:
            # A reopened stream's message starts at ``base``; the sealed one before it already
            # shows the prefix. (A ``replace`` rewrite is not prefix-aligned, so use it whole.)
            shown = text if replace else text[stream.get("base", 0):]
            edited = await self.edit_message(chat_id, ts, shown, finalize=True, metadata=metadata)
            committed = edited.success or (sealed and not replace)
            if not sealed:
                logger.warning(
                    "[Slack] chat.stopStream failed for %s/%s; %s", chat_id, ts,
                    "final committed via chat.update instead (no duplicate post)" if committed
                    else "in-place update failed too, delivering the final as a fresh post")
        if not committed:
            return None
        await self.stop_typing(chat_id, metadata)
        return SendResult(success=True, message_id=ts)

    async def _replace_stream_chunks(
        self, key: Tuple[str, str, str], stream: Dict[str, Any], chunks: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Replace a raw stream with bounded, already-formatted chunks, without losing its tail.

        Update failure falls back to the normal full-final post. Once the update lands,
        continuation failures propagate to send() as failures, never as full success
        or a second full answer. Do not reformat chunks or derive oversized blocks.
        """
        try:  # package and flat plugin-dir imports share the adapter's helper seam
            from .adapter import _slack_unfurl_kwargs
        except ImportError:  # pragma: no cover - flat plugin-dir import
            from adapter import _slack_unfurl_kwargs

        team_id, chat_id, thread_ts = key
        ts = stream["ts"]
        client_fn = lambda: self._get_client(chat_id, team_id=team_id or None)  # noqa: E731
        try:
            await self._call_with_block_fallback(
                client_fn, "chat_update",
                {"channel": chat_id, "ts": ts, "text": chunks[0], "blocks": []}, "edit")
        except Exception as e:
            logger.warning(
                "[Slack] Chunked native replacement failed for %s/%s; "
                "delivering the final as a fresh post: %s", chat_id, ts, e, exc_info=True)
            return None
        for chunk in chunks[1:]:
            response = await self._call_with_block_fallback(
                client_fn, "chat_postMessage", {
                    "channel": chat_id, "thread_ts": thread_ts, "text": chunk,
                    "mrkdwn": True, **_slack_unfurl_kwargs(self.config.extra),
                }, "send")
            continuation_ts = response.get("ts") if response else None
            if continuation_ts:
                self._bot_message_ts.add(self._workspace_message_marker(team_id, continuation_ts))
        self._bot_message_ts.add(self._workspace_message_marker(team_id, thread_ts))
        self._trim_bot_message_timestamps()
        await self.stop_typing(chat_id, metadata)
        return SendResult(success=True, message_id=ts)
