"""Slack literal-formatting contracts, including tool previews and escape handling."""
from types import SimpleNamespace

import pytest

from tests.gateway import test_slack

adapter = test_slack.adapter


class TestLiteralFormatting:
    """Markdown escapes and literal preview text keep their original meaning."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target,pattern", [
        ("files", "*test*.py"),
        ("content", r"test_.*_.*\.py"),
    ])
    async def test_tool_progress_preserves_search_pattern(self, adapter, target, pattern):
        from gateway.run_turn_runner import TurnRunner

        args = {"target": target, "pattern": pattern}
        ctx = SimpleNamespace(source=None, progress_mode="all", last_was_terminal_block=[False])
        runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
        message = TurnRunner(runner, ctx)._progress_build_message("search_files", pattern, args)
        client = adapter._app.client
        client.chat_postMessage.return_value = {"ok": True, "ts": "123.456"}
        client.chat_update.return_value = {"ok": True, "ts": "123.456"}

        assert (await adapter.send("C123", message)).success
        assert (await adapter.edit_message("C123", "123.456", message)).success
        for method in (client.chat_postMessage, client.chat_update):
            assert method.call_args.kwargs["text"].endswith(f"`{pattern}`")
        assert args == {"target": target, "pattern": pattern}

    def test_tool_preview_backticks_do_not_break_code_span(self, adapter):
        from agent.display import ToolPreview

        preview = ToolPreview(text="`*test*`.py")
        rendered = adapter.format_message(adapter.format_tool_preview(preview))
        assert rendered == "`ˋ*test*ˋ.py`"
        assert preview.text == "`*test*`.py"

    def test_markdown_backslash_escapes_render_as_literal_text(self, adapter):
        text = r"\*CPU・GPU共用メモリ\*" + "\n" + r"1\. 項目" + "\n" + r"\- 項目"
        assert adapter.format_message(text) == "`*`CPU・GPU共用メモリ `*`\n1. 項目\n- 項目"

    @pytest.mark.parametrize("delimiter", ["*", "_", "~"])
    def test_escaped_emphasis_is_literal_in_slack_mrkdwn(self, adapter, delimiter):
        text = f"\\{delimiter}hello\\{delimiter}"
        # Bare punctuation would re-enable Slack bold/italic/strike formatting.
        assert adapter.format_message(text) == f"`{delimiter}`hello`{delimiter}`"

    def test_markdown_hard_break_does_not_leak_backslash(self, adapter):
        assert adapter.format_message("first\\\nsecond\\\r\nthird") == "first\nsecond\r\nthird"

    @pytest.mark.parametrize("text", [r"`\*raw\*`", "```text\n" + r"\*raw\*" + "\n```"])
    def test_backslash_escapes_inside_code_are_preserved(self, adapter, text):
        assert adapter.format_message(text) == text.replace("```text\n", "```\n")

    def test_escaped_slack_control_chars_cannot_create_mentions(self, adapter):
        assert adapter.format_message(r"\<@U123\> \&") == "&lt;@U123&gt; &amp;"

    @pytest.mark.parametrize("text", [
        r"<https://example.com/a\?b>",
        r"<https://example.com/a\?b|label\*>",
        r"<mailto:user@example.com|label\_>",
        r"<@U123|label\*>",
        r"<#C123|label\~>",
    ])
    def test_native_entities_preserve_backslashes(self, adapter, text):
        assert adapter.format_message(text) == text

    def test_escaped_native_entity_opening_is_not_a_mention(self, adapter):
        assert adapter.format_message(r"\<@U123>") == "&lt;@U123&gt;"

    @pytest.mark.parametrize("destination,expected", [
        ("https://example.com", "https://example.com"),
        (r"https://example.com/a\?b", "https://example.com/a?b"),
    ])
    def test_angle_wrapped_markdown_link_destination(self, adapter, destination, expected):
        assert adapter.format_message(f"[link](<{destination}>)") == f"<{expected}|link>"

    @pytest.mark.parametrize("delimiter", ["*", "_", "~"])
    def test_ordinary_markdown_link_destination_unescapes_emphasis(self, adapter, delimiter):
        text = f"[link](https://example.com/a\\{delimiter}b)"
        assert adapter.format_message(text) == f"<https://example.com/a{delimiter}b|link>"

    def test_broadcast_label_entities_are_not_double_escaped(self, adapter):
        assert adapter.format_message("<!here|AT&amp;T>") == "&lt;!here|AT&amp;T&gt;"

    @pytest.mark.parametrize("text,expected", [
        (r"\<!channel>", "&lt;!channel&gt;"),
        (r"\<!everyone\>", "&lt;!everyone&gt;"),
        (r"\<!here|here>", "&lt;!here|here&gt;"),
    ])
    def test_escaped_broadcast_mentions_are_not_double_escaped(self, adapter, text, expected):
        assert adapter.format_message(text) == expected

    def test_non_markdown_backslashes_are_preserved(self, adapter):
        assert adapter.format_message(r"C:\Users\name") == r"C:\Users\name"
