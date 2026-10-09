"""Speech-delta and citation-preview callback wiring for a gateway turn.

TurnRunner inherits this phase; stream creation and context ownership stay there.
"""
from typing import Optional

from gateway.config import Platform


class TurnStreamMixin:
    """Wire existing stream sinks and completed interim assistant commentary."""

    def _wire_stream_callbacks(self, stream_consumer, stts, want_interim_messages):
        ctx = self._ctx
        # Deltas tee to the stream consumer (when text streaming is on) and to streaming TTS.
        delta_sinks = [
            sc for sc in (
                stream_consumer if stream_consumer and stream_consumer.stream_deltas_enabled else None,
                stts,
            ) if sc is not None
        ]
        stream_delta_cb = None
        citation_preview = None
        if stream_consumer is not None and ctx.source.platform == Platform.SLACK:
            from agent.slack_citation_delivery import SlackCitationPreviewBuffer
            citation_preview = SlackCitationPreviewBuffer()
        if delta_sinks:
            def stream_delta_cb(text: Optional[str]) -> None:
                if ctx._run_still_current():
                    for sink in delta_sinks:
                        if sink is stream_consumer and citation_preview is not None and isinstance(text, str):
                            visible = citation_preview.feed(text)
                            if visible:
                                sink.on_delta(visible)
                        else:
                            sink.on_delta(text)

        def interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            if stts is not None:
                # Flush accepted deltas; completed commentary is a separate speech segment.
                stts.on_delta(None)
                if not already_streamed:
                    stts.on_delta(text)
                    stts.on_delta(None)
            if stream_consumer is not None:
                stream_consumer.on_segment_break() if already_streamed else stream_consumer.on_commentary(text)
            elif not already_streamed and ctx._status_adapter and str(text or "").strip():
                self._send_status_text(text, ctx._status_thread_metadata, "interim_assistant_callback scheduling error")

        return stream_consumer, stream_delta_cb, interim_assistant_cb, want_interim_messages
