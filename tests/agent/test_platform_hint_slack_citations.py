"""Slack research replies need portable links, not renderer-only citation IDs."""
from types import SimpleNamespace

from agent.prompt_builder import PLATFORM_HINTS
from agent.system_prompt import platform_hint


def test_slack_hint_requests_clickable_verified_source_links():
    hint = platform_hint(SimpleNamespace(platform="slack", _platform_hint_overrides={}))
    assert "clickable standard Markdown source links" in hint
    assert "URLs actually returned by retrieval" in hint
    assert "provider-native citation markers alone are not usable source links" in hint


def test_slack_hint_keeps_research_replies_scannable():
    hint = PLATFORM_HINTS["slack"]
    assert "conclusion first" in hint
    assert "short labeled bullets" in hint
    assert "Avoid whole-paragraph bold" in hint


def test_slack_portable_source_guidance_does_not_change_other_surfaces():
    for platform in ("cli", "telegram", "discord"):
        assert "provider-native citation markers alone" not in PLATFORM_HINTS[platform]
