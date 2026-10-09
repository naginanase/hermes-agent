"""A failed native replacement retains the I/O traceback and full-final fallback."""
import logging

import pytest

from tests.gateway.test_slack_native_streaming import META, _make_adapter


@pytest.mark.asyncio
async def test_failed_native_replacement_records_traceback_and_posts_full_final(caplog):
    adapter, client = _make_adapter()
    error = RuntimeError("replacement transport unavailable")
    client.chat_update.side_effect = error
    adapter.MAX_MESSAGE_LENGTH = 120
    raw = "Claim [unresolved citation: turn0view4] " + "tail " * 90
    started = await adapter.send_draft("C1", 7, raw, metadata=META)
    assert started.success
    with caplog.at_level(logging.WARNING):
        result = await adapter.send("C1", raw, metadata=META)
    assert result.success
    assert not adapter._active_streams
    client.chat_update.assert_awaited_once()
    posts = client.chat_postMessage.await_args_list
    assert posts
    delivered = "".join(call.kwargs["text"] for call in posts)
    assert delivered.count("tail") == 90
    assert "turn0view4" not in delivered
    records = [record for record in caplog.records if "Chunked native replacement failed" in record.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is error
    assert records[0].exc_info[2] is not None
