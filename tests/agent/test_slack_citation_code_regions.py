"""Citation delivery protects original Markdown code, never rendered token text."""
from copy import deepcopy

import pytest

from agent.slack_citation_delivery import (
    prepare_slack_citation_delivery,
    slack_citation_draft_body,
)

NATIVE = "\ue200cite\ue202turn0view4\ue201"
DIAGNOSTIC = "[unresolved citation: turn0view4]"


@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
def test_double_backtick_code_is_exact_without_false_diagnostics(marker):
    raw = "``" + marker + "``"
    result = prepare_slack_citation_delivery(raw)
    assert result.text == raw
    assert result.unresolved_count == 0
    assert result.diagnostic_refs == ()
    assert result.source_count == 0


@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
def test_double_backtick_code_is_not_a_citation_draft(marker):
    raw = "``" + marker + "``"
    assert slack_citation_draft_body(raw) is None


@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
@pytest.mark.parametrize("template", [
    "~~~text\n{marker}\n~~~",
    "````text\n{marker}\n```\n{marker}\n`````",
    "~~~text\n{marker}\n```\n{marker}\n~~~~",
    "```text\n{marker}",
    "~~~text\r\n{marker}\r\n~~~~\r\n",
    "    {marker}\n",
    "\t{marker}\r\n",
    "- ```text\n  {marker}\n  ```",
    "> ~~~text\n> {marker}\n> ~~~",
    "- item\n\n      {marker}\n",
    ">     {marker}\n",
])
def test_block_code_is_exact_without_false_citations(template, marker):
    raw = template.format(marker=marker)
    result = prepare_slack_citation_delivery(raw)
    assert result.text == raw
    assert result.unresolved_count == 0
    assert result.diagnostic_refs == ()
    assert slack_citation_draft_body(raw) is None


@pytest.mark.parametrize("code", [
    "~~~text\r\n[unresolved citation: literal]\r\n~~~~\r\n",
    "```text\n[unresolved citation: literal]  \t",
    "    [unresolved citation: literal]  \t\r\n",
])
def test_warning_does_not_strip_trailing_original_code_bytes(code):
    raw = "Claim " + NATIVE + "\n\n" + code
    result = prepare_slack_citation_delivery(raw)
    assert result.text.startswith("Claim ※\n\n" + code + "\n\n")
    assert result.unresolved_count == 1
    assert result.diagnostic_refs == ("turn0view4",)
    assert slack_citation_draft_body(raw) == "Claim ※\n\n" + code


@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 9, 32])
def test_inline_delimiters_use_the_complete_matching_run(marker, length):
    ticks = "`" * length
    shorter = "`" * (length - 1) if length > 1 else ""
    raw = "Example: " + ticks + "  \x00" + marker + " " + shorter + "\r\n" + marker + "  " + ticks
    result = prepare_slack_citation_delivery(raw)
    assert result.text == raw
    assert result.unresolved_count == 0
    assert result.diagnostic_refs == ()
    assert slack_citation_draft_body(raw) is None


@pytest.mark.parametrize("template,count", [
    (r"\`{marker}\`", 1),
    ("`{marker}", 1),
    ("``{marker}`", 1),
    ("`{marker}\n\n{marker}`", 2),
    (r"`example\` {marker}", 1),
    ("prose\n    {marker}", 1),
    ("~~\n{marker}\n~~", 1),
])
@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
def test_escaped_unmatched_delimiters_and_noncode_indent_remain_prose(template, count, marker):
    raw = template.format(marker=marker)
    result = prepare_slack_citation_delivery(raw)
    assert result.unresolved_count == count
    assert result.diagnostic_refs == ("turn0view4",) * count
    assert marker not in result.text
    assert slack_citation_draft_body(raw) == raw.replace(marker, "※")


@pytest.mark.parametrize("template", [
    "  # head `` {marker} `` ###\r\n",
    "- first ``{marker}``\r\n  next ``{marker}``\r\n",
    "> first ``{marker}\r\n> second {marker}``\r\n",
    "> - first ``{marker}\r\n>   second {marker}``\r\n",
    "- first ``{marker}\r\n  second {marker}``\r\n",
    "-\tfirst ``{marker}\r\n\tsecond {marker}``\r\n",
    ">\tfirst ``{marker}\r\n>\tsecond {marker}``\r\n",
    ">  title ``{marker}``\r\n>  ===\r\n",
    "[``{marker}``](https://example.org/) ``{marker}``",
    "![``{marker}``](https://example.org/) ``{marker}``",
])
@pytest.mark.parametrize("marker", [NATIVE, DIAGNOSTIC])
def test_inline_source_mapping_survives_whitespace_and_container_removal(template, marker):
    raw = template.format(marker=marker)
    result = prepare_slack_citation_delivery(raw)
    assert result.text == raw
    assert result.unresolved_count == 0
    assert slack_citation_draft_body(raw) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_identical_code_and_prose_examples_keep_offsets_sources_and_history_immutable(newline):
    code = "``" + NATIVE + "\x00 " + DIAGNOSTIC + "``"
    fenced = newline.join(["> ~~~~text", "> " + NATIVE, "> ~~~", "> " + DIAGNOSTIC, "> ~~~~~"])
    raw = newline.join(["Before " + NATIVE, code, "Between " + DIAGNOSTIC, "", fenced, "", code, "After " + NATIVE])
    expected = newline.join(["Before ※", code, "Between ※", "", fenced, "", code, "After ※"])
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "codex_message_items": [{"type": "hermes_retrieval_sources", "sources": [
            {"url": "https://example.org/old", "provenance": "web_search_call.action.sources"}]}]},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": raw, "codex_message_items": [
            {"type": "message", "id": "immutable_message", "content": [{"type": "output_text", "text": raw}]},
            {"type": "hermes_retrieval_sources", "request_id": "immutable_response", "sources": [
                {"url": "https://example.org/current", "source_id": "turn0view4",
                 "provenance": "web_search_call.action.sources"}]}]},
    ]
    history = deepcopy(messages)
    result = prepare_slack_citation_delivery(raw, messages=messages, history_offset=20)
    assert result.text.startswith(expected + "\n\n")
    assert result.unresolved_count == 3
    assert result.diagnostic_refs == ("turn0view4",) * 3
    assert result.source_count == 1
    assert "https://example.org/current" in result.text
    assert "https://example.org/old" not in result.text
    assert slack_citation_draft_body(raw) == expected
    assert messages == history
    assert messages[-1]["content"] == raw


@pytest.mark.parametrize("template", [
    "- ~~~\n  {marker}\n\nOutside {marker}",
    "> ```\n> {marker}\n\nOutside {marker}",
    "    {marker}\n\nOutside {marker}",
    "~~~~\n{marker}\n~~~ wrong\n{marker}\n~~~~~\nOutside {marker}",
])
def test_block_boundary_does_not_hide_prose_outside_the_actual_container(template):
    raw = template.format(marker=DIAGNOSTIC)
    result = prepare_slack_citation_delivery(raw)
    assert result.unresolved_count == 1
    assert result.diagnostic_refs == ("turn0view4",)
    assert "Outside ※" in result.text
    assert "Outside " + DIAGNOSTIC not in result.text
    assert slack_citation_draft_body(raw) == raw.rsplit(DIAGNOSTIC, 1)[0] + "※"


@pytest.mark.parametrize("partial", ["\ue200cite\ue202turn0", "[unresolved citation: turn0"])
@pytest.mark.parametrize("template", ["~~~\n{partial}", "> ```\n> {partial}", "    {partial}"])
def test_partial_citation_at_end_of_block_code_is_not_a_draft_match(template, partial):
    raw = template.format(partial=partial)
    assert prepare_slack_citation_delivery(raw).text == raw
    assert slack_citation_draft_body(raw) is None


def test_empty_body_has_no_code_or_draft():
    assert prepare_slack_citation_delivery("").text == ""
    assert slack_citation_draft_body("") is None


@pytest.mark.parametrize("prefix,code,suffix", [
    ("# prose " + DIAGNOSTIC + " ", "``" + NATIVE + "``", " " + DIAGNOSTIC + " ###\r\n"),
    ("> - prose ", "``\x00 " + NATIVE + "\r\n>   " + DIAGNOSTIC + "``", " after " + NATIVE),
    ("-\tprose ", "``" + NATIVE + "\r\n\t" + DIAGNOSTIC + "``", " after " + NATIVE),
    ("before " + NATIVE + "\r\n\r\n", "> ~~~~text\r\n> " + DIAGNOSTIC + "\r\n> ~~~~~\r\n", "\r\nafter " + NATIVE),
    ("before " + NATIVE + "\r\n\r\n", "    " + DIAGNOSTIC + "  \t\r\n", "\r\nafter " + NATIVE),
])
def test_region_boundaries_are_exact_original_offsets_not_whole_inline_blocks(prefix, code, suffix):
    from agent.markdown_code_regions import markdown_code_parts, markdown_code_regions
    raw = prefix + code + suffix
    assert markdown_code_regions(raw) == [(len(prefix), len(prefix + code))]
    assert list(markdown_code_parts(raw)) == [(False, prefix), (True, code), (False, suffix)]
    assert "".join(part for _, part in markdown_code_parts(raw)) == raw


def test_identical_inline_examples_get_distinct_exact_offsets():
    from agent.markdown_code_regions import markdown_code_regions
    prose = NATIVE + " "
    code = "``" + NATIVE + "``"
    raw = prose + code + prose + code + prose
    assert markdown_code_regions(raw) == [
        (len(prose), len(prose + code)),
        (len(prose + code + prose), len(prose + code + prose + code)),
    ]
