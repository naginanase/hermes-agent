"""Flat mrkdwn boundaries from the live Japanese A/B/C rendering repro."""

import pytest

from plugins.platforms.slack.adapter import SlackAdapter, _standalone_format_mrkdwn


@pytest.fixture
def adapter():
    return SlackAdapter.__new__(SlackAdapter)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("- 強調： **太字**", "- 強調： *太字*"),  # A: already renders
        ("- 強調：**太字**", "- 強調： *太字*"),  # B: missing boundary
        ("- Bold: **Bold text**", "- Bold: *Bold text*"),  # C: already renders
        ("- 強調：**この部分だけ太字**", "- 強調： *この部分だけ太字*"),
    ],
    ids=["live-A", "live-B", "live-C", "original-thread"],
)
def test_live_bold_repro_adds_only_missing_cjk_boundary(adapter, text, expected):
    assert adapter.format_message(text) == expected
    assert _standalone_format_mrkdwn(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("- コード： `rich_blocks`", "- コード： `rich_blocks`"),
        ("- コード：`rich_blocks`", "- コード： `rich_blocks`"),
        ("- Code: `rich_blocks`", "- Code: `rich_blocks`"),
        ("`rich_blocks`です", "`rich_blocks` です"),
        ("前`rich_blocks`後", "前 `rich_blocks` 後"),
    ],
    ids=["live-A", "live-B", "live-C", "trailing-CJK", "both-CJK"],
)
def test_inline_code_adds_boundaries_outside_delimiters(adapter, text, expected):
    assert adapter.format_message(text) == expected
    assert _standalone_format_mrkdwn(text) == expected


@pytest.mark.parametrize(
    "body",
    [r"\*raw\*", " **太字** *italic* ~~strike~~ ", r"C:\Users\名前", " <@U123>& ", "https://example.org/字**a**_b_"],
)
def test_cjk_adjacent_inline_code_content_is_byte_exact(adapter, body):
    shown = adapter.format_message(f"前`{body}`後")
    assert shown == f"前 `{body}` 後"
    assert shown.split("`")[1].encode("utf-8") == body.encode("utf-8")


@pytest.mark.parametrize("delimiter", ["*", "_", "~"])
def test_escaped_literal_emphasis_code_fallback_has_cjk_boundaries(adapter, delimiter):
    text = f"前\\{delimiter}太字\\{delimiter}後"
    assert adapter.format_message(text) == f"前 `{delimiter}` 太字 `{delimiter}` 後"


@pytest.mark.parametrize(
    "source, span",
    [("**太字**", "*太字*"), ("*斜体*", "_斜体_"), ("~~削除~~", "~削除~"), ("***両方***", "*_両方_*")],
)
@pytest.mark.parametrize("before, after", [("前", "後"), ("", "です"), ("日本語：", ""), ("中文", "。"), ("한글", "후")])
def test_inline_styles_share_cjk_boundary_rule(adapter, source, span, before, after):
    expected = f"{before}{' ' if before else ''}{span}{' ' if after else ''}{after}"
    assert adapter.format_message(f"{before}{source}{after}") == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("**太字**", "*太字*"),
        ("English**bold**tail", "English*bold*tail"),
        ("English*italic*tail", "English_italic_tail"),
        ("English~~strike~~tail", "English~strike~tail"),
        ("English`code`tail", "English`code`tail"),
        ("前 **太字** 後", "前 *太字* 後"),
        ("前\t**太字**\n後", "前\t*太字*\n後"),
        ("前\u3000**太字**\u3000後", "前\u3000*太字*\u3000後"),
        ("前\u3000`code`\u3000後", "前\u3000`code`\u3000後"),
        ("**末尾。**", "*末尾。\u200b*"),
        ("前**末尾。**後", "前 *末尾。\u200b* 後"),
        (r"\*CPU・GPU共用メモリ\*" + "\n" + r"1\. 項目" + "\n" + r"\- 項目",
         "`*`CPU・GPU共用メモリ `*`\n1. 項目\n- 項目"),
    ],
)
def test_existing_boundaries_and_unicode_punctuation_workaround_survive(adapter, text, expected):
    assert adapter.format_message(text) == expected


@pytest.mark.parametrize(
    "entity",
    [
        r"<https://example.org/字\*字\_字\~字|表示>",
        "<https://example.org/字**太字**字|表示>",
        "<https://example.org/字`raw`字|表示>",
        "<mailto:user@example.org|字`raw`字>",
        r"<@U123|字\*字>",
        "<#C123|字`raw`字>",
        "<!date^1700000000^{date_short}|字`raw`字>",
    ],
)
def test_native_entities_and_autolink_data_are_unchanged(adapter, entity):
    text = f"前{entity}後"
    assert adapter.format_message(text) == text


@pytest.mark.parametrize(
    "label, destination, expected_url",
    [
        ("表示", "https://example.org/字**太字**字", "https://example.org/字**太字**字"),
        ("表示", r"https://example.org/字\*字\_字\~字", "https://example.org/字*字_字~字"),
        ("字`raw`字", "https://example.org/path", "https://example.org/path"),
        ("表示", "<https://example.org/字`raw`字>", "https://example.org/字`raw`字"),
    ],
)
def test_markdown_links_do_not_receive_span_padding(adapter, label, destination, expected_url):
    assert adapter.format_message(f"前[{label}]({destination})後") == f"前<{expected_url}|{label}>後"


@pytest.mark.parametrize(
    "fence",
    ["```\n前**太字**後\n字`raw`字\n```", "前```字**太字**字```後"],
)
def test_code_fences_remain_byte_exact(adapter, fence):
    assert adapter.format_message(fence) == fence
