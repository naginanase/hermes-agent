"""Cache-safe Slack hint upgrades through real prompt restore and request assembly."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import platform_hint
from agent.turn_context import build_api_messages, build_turn_context
from hermes_state import SessionDB
from tools.mcp_tool_agent import tool_pin_version


OLD_SLACK_HINT = (
    "You are in a Slack workspace communicating with your user. Standard markdown is "
    "auto-converted to Slack formatting (bold, headers, links, code); tables are NOT "
    "supported — use bullet lists or labeled lines."
)
NOTE_PREFIX = "[System: Slack formatting guidance upgrade v1"


def _prompt(hint=OLD_SLACK_HINT, platform="slack"):
    return (
        f"Frozen system prefix\n\n{hint}\n\nConversation started: Friday, October 09, 2026\n"
        f"Model: test/model\nProvider: openrouter\nPlatform: {platform}"
    )


class _Agent:
    """Only external effects are stubbed; restore/prologue/wire builders are real."""

    def __init__(self, db, platform="slack"):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.api_mode = "chat_completions"
        self.platform = platform
        self.quiet_mode = True
        self.max_iterations = 10
        self.tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        self.valid_tool_names = {"read_file"}
        self.enabled_toolsets = self.disabled_toolsets = None
        self._skip_mcp_refresh = True
        self._session_db = db
        self._session_db_created = True
        self._cached_system_prompt: str | None = None
        self._surface_switch_note = ""
        self._gateway_turn_context_notes = ""
        self._use_prompt_caching = False
        self._bot_mode_protocol = False
        self._platform_hint_overrides = None
        self._build_system_prompt = MagicMock(side_effect=lambda _: _prompt(platform_hint(self), self.platform))
        self.compression_enabled = False
        self.context_compressor = SimpleNamespace(protect_first_n=2, protect_last_n=2)
        self._memory_store = self._memory_manager = None
        self._memory_nudge_interval = self._turns_since_memory = self._user_turn_count = 0
        self._todo_store = SimpleNamespace(has_items=lambda: True)
        self._tool_guardrails = SimpleNamespace(reset_for_turn=lambda: None)
        self._compression_warning = None
        self._interrupt_requested = False
        self.ephemeral_system_prompt = ""
        self._memory_write_origin = "assistant_tool"
        self._restore_primary_runtime = lambda: None
        self._cleanup_dead_connections = lambda: False
        self._ensure_db_session = lambda: None
        self._copy_reasoning_content_for_api = lambda *a: None
        self._should_sanitize_tool_calls = lambda: False
        self.persisted: list[dict[str, Any]] | None = None

    def _persist_session(self, messages, _history):
        # Snapshot the exact durable bytes BEFORE the request is built.
        self.persisted = deepcopy(messages)
        current = messages[-1]
        self._session_db.append_message(
            self.session_id, current["role"], content=current["content"],
            api_content=current.get("api_content"), timestamp=current.get("timestamp"),
        )


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **k: [])
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)


@pytest.fixture
def db(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as store:
        store.create_session("sess-1", source="slack")
        yield store


def _seed(db, prompt):
    db.update_system_prompt("sess-1", prompt)
    db.update_session_tool_names("sess-1", {"version": tool_pin_version(), "tools": _Agent(db).tools})


def _turn(agent, history, content: Any = "hello"):
    ctx = build_turn_context(
        agent, content, None, history, None, None, None,
        restore_or_build_system_prompt=_restore_or_build_system_prompt,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=str,
        set_session_context=lambda *a: None,
        set_current_write_origin=lambda *a: None,
        ra=lambda: SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    wire, system = build_api_messages(
        agent, ctx.messages, current_turn_user_idx=ctx.current_turn_user_idx,
        ext_prefetch_cache=ctx.ext_prefetch_cache, plugin_user_context=ctx.plugin_user_context,
        moa_config=None, active_system_prompt=ctx.active_system_prompt,
    )
    from agent.agent_runtime_helpers import note_turn_persisted
    note_turn_persisted(agent)
    return ctx, wire, system


def test_old_same_platform_slack_restores_exact_bytes_with_versioned_tail(db):
    stored = _prompt()
    _seed(db, stored)
    agent = _Agent(db)
    history = [{"role": "user", "content": "old", "api_content": "old\n\nFROZEN CONTEXT"},
               {"role": "assistant", "content": "previous answer"}]
    original_history = deepcopy(history)
    tools_before = deepcopy(agent.tools)
    row_before = db.get_session("sess-1")

    ctx, wire, system = _turn(agent, history)

    current = ctx.messages[-1]
    assert agent.persisted is not None
    assert agent._cached_system_prompt is not None
    assert NOTE_PREFIX in current.get("api_content", "")
    assert platform_hint(agent) in current["api_content"]
    assert current["content"] == "hello"
    assert wire[-1]["content"] == current["api_content"] == agent.persisted[-1]["api_content"]
    assert system.encode() == stored.encode() == agent._cached_system_prompt.encode()
    assert wire[0] == {"role": "system", "content": stored}
    assert wire[1]["content"] == "old\n\nFROZEN CONTEXT"
    assert history == original_history
    assert ctx.messages[:-1] == original_history
    assert agent.tools == tools_before
    assert db.get_session("sess-1")["system_prompt"] == stored
    assert db.get_session("sess-1")["tool_names"] == row_before["tool_names"]
    agent._build_system_prompt.assert_not_called()
    # Exercise deduplicated prompt storage, not only an inline row fake.
    raw = db._conn.execute("SELECT system_prompt, system_prompt_hash FROM sessions WHERE id=?", ("sess-1",)).fetchone()
    assert raw[0] is None and raw[1]


@pytest.mark.parametrize("cached_agent", [False, True], ids=["fresh-gateway-agent", "cached-agent"])
def test_upgrade_is_not_repeated_after_durable_replay(db, cached_agent):
    _seed(db, _prompt())
    first = _Agent(db)
    ctx, first_wire, _ = _turn(first, [{"role": "assistant", "content": "earlier"}])
    db.append_message("sess-1", "assistant", content="answer")
    history = db.get_messages_as_conversation("sess-1")
    assert history[0]["api_content"] == ctx.messages[-1]["api_content"]
    history_before = deepcopy(history)

    next_agent = first if cached_agent else _Agent(db)
    next_ctx, wire, _ = _turn(next_agent, history, "next question")

    assert "api_content" not in next_ctx.messages[-1]
    assert next_agent.persisted is not None
    assert next_agent.persisted[-1].get("api_content") is None
    assert wire[1]["content"] == first_wire[-1]["content"]
    assert wire[-1]["content"] == "next question"
    assert history == history_before
    assert next_agent._surface_switch_note == ""


def test_surface_switch_and_slack_upgrade_share_one_copy_of_current_guidance(db):
    from agent.surface_switch import _SURFACE_SWITCH_NOTE_PREFIX

    stored = _prompt("Desktop formatting instructions", "desktop")
    _seed(db, stored)
    agent = _Agent(db)
    ctx, wire, system = _turn(agent, [{"role": "assistant", "content": "earlier"}])
    tail = ctx.messages[-1]["api_content"]

    assert NOTE_PREFIX in tail
    assert _SURFACE_SWITCH_NOTE_PREFIX + "slack" in tail
    assert tail.count(platform_hint(agent)) == 1
    assert wire[-1]["content"] == tail
    assert system == stored

    db.append_message("sess-1", "assistant", content="answer")
    replay = db.get_messages_as_conversation("sess-1")
    next_ctx, _, _ = _turn(_Agent(db), replay, "next")
    assert "api_content" not in next_ctx.messages[-1]


def test_pending_upgrade_is_not_duplicated_on_prologue_retry(db):
    from agent.surface_switch import stage_slack_hint_upgrade

    _seed(db, _prompt())
    agent = _Agent(db)
    history = [{"role": "assistant", "content": "earlier"}]
    _restore_or_build_system_prompt(agent, None, history)
    assert agent._cached_system_prompt is not None
    assert stage_slack_hint_upgrade(agent, agent._cached_system_prompt, history)

    ctx, _, _ = _turn(agent, history)

    assert ctx.messages[-1]["api_content"].count(NOTE_PREFIX) == 1


def test_multimodal_upgrade_is_durable_and_replayed_without_reinjection(db):
    _seed(db, _prompt())
    agent = _Agent(db)
    image = {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}
    content = [{"type": "text", "text": "describe this"}, image]
    original_parts = deepcopy(content)
    history = [{"role": "user", "content": deepcopy(original_parts)},
               {"role": "assistant", "content": "earlier"}]
    history_before = deepcopy(history)

    ctx, wire, system = _turn(agent, history, content)

    current = ctx.messages[-1]
    assert agent.persisted is not None
    assert "api_content" not in current
    assert current["content"][:-1] == original_parts
    assert NOTE_PREFIX in current["content"][-1]["text"]
    assert platform_hint(agent) in current["content"][-1]["text"]
    assert wire[-1]["content"] == current["content"] == agent.persisted[-1]["content"]
    assert history == history_before
    assert system == _prompt()
    db.append_message("sess-1", "assistant", content="answer")
    replay = db.get_messages_as_conversation("sess-1")
    assert replay[0]["content"] == current["content"]

    next_ctx, next_wire, _ = _turn(_Agent(db), replay, "next")

    assert "api_content" not in next_ctx.messages[-1]
    assert next_wire[1]["content"] == wire[-1]["content"]


def test_fresh_slack_session_needs_no_compatibility_note(db):
    agent = _Agent(db)
    ctx, wire, system = _turn(agent, None)
    assert platform_hint(agent) in system
    assert NOTE_PREFIX not in system
    assert "api_content" not in ctx.messages[-1]
    assert wire[-1]["content"] == "hello"
    agent._build_system_prompt.assert_called_once_with(None)


def test_continuing_slack_with_current_hint_needs_no_note(db):
    agent = _Agent(db)
    current_prompt = _prompt(platform_hint(agent))
    _seed(db, current_prompt)
    ctx, wire, system = _turn(agent, [{"role": "assistant", "content": "earlier"}])
    assert "api_content" not in ctx.messages[-1]
    assert wire[-1]["content"] == "hello"
    assert system == current_prompt
    agent._build_system_prompt.assert_not_called()


@pytest.mark.parametrize("platform", ["cli", "discord", "telegram", "desktop"])
def test_non_slack_continuations_are_unchanged(db, platform):
    stored = _prompt("Other platform's old guidance", platform)
    _seed(db, stored)
    ctx, wire, system = _turn(_Agent(db, platform), [{"role": "assistant", "content": "earlier"}])
    assert "api_content" not in ctx.messages[-1]
    assert wire[-1]["content"] == "hello"
    assert system == stored


@pytest.mark.parametrize("mode", ["moa", "codex_app_server"])
def test_unstamped_request_modes_do_not_stage_an_upgrade(db, mode):
    _seed(db, _prompt())
    agent = _Agent(db)
    if mode == "moa":
        agent.provider = "moa"
        # Keep runtime identity matched so only the tail-upgrade gate is exercised.
        _seed(db, _prompt().replace("Provider: openrouter", "Provider: moa"))
    else:
        agent.api_mode = mode
    ctx, _, _ = _turn(agent, [{"role": "assistant", "content": "earlier"}])
    assert "api_content" not in ctx.messages[-1]
    assert agent._surface_switch_note == ""


def test_one_shot_note_is_still_deduplicated_beyond_surface_switch_scan_tail(db):
    _seed(db, _prompt())
    first_ctx, _, _ = _turn(_Agent(db), [{"role": "assistant", "content": "earlier"}])
    history = deepcopy(first_ctx.messages)
    history.append({"role": "assistant", "content": "answer"})
    for index in range(110):
        history.extend([{"role": "user", "content": f"question {index}"},
                        {"role": "assistant", "content": f"answer {index}"}])
    original = deepcopy(history)
    ctx, _, _ = _turn(_Agent(db), history, "next")
    assert "api_content" not in ctx.messages[-1]
    assert history == original


def test_already_cached_old_slack_prompt_gets_the_upgrade(db):
    _seed(db, _prompt())
    agent = _Agent(db)
    agent._cached_system_prompt = _prompt()
    ctx, _, system = _turn(agent, [{"role": "assistant", "content": "earlier"}])
    assert NOTE_PREFIX in ctx.messages[-1]["api_content"]
    assert system == _prompt()
    agent._build_system_prompt.assert_not_called()


def test_return_to_slack_does_not_point_back_to_the_obsolete_system_hint(db):
    _seed(db, _prompt())
    _, _, _ = _turn(_Agent(db), [{"role": "assistant", "content": "earlier"}])
    db.append_message("sess-1", "assistant", content="first answer")
    _, _, _ = _turn(_Agent(db, "discord"), db.get_messages_as_conversation("sess-1"), "discord question")
    db.append_message("sess-1", "assistant", content="discord answer")

    agent = _Agent(db)
    ctx, _, system = _turn(agent, db.get_messages_as_conversation("sess-1"), "back on Slack")
    tail = ctx.messages[-1]["api_content"]

    assert "the interface section in the system prompt above" not in tail
    assert platform_hint(agent) in tail
    assert NOTE_PREFIX not in tail  # The versioned upgrade remains one-shot.
    assert system == _prompt()


def test_gateway_restart_sends_upgrade_once_on_the_actual_provider_request(db):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from run_agent import AIAgent

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in [({"role": "assistant", "content": "answer"}, None), ({}, "stop")]:
                    chunk = {"id": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            body = json.dumps({
                "id": "chatcmpl-slack-upgrade-test", "object": "chat.completion", "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stored = _prompt().replace("Model: test/model", "Model: test-model").replace(
        "Provider: openrouter", "Provider: openai-compat"
    )
    db.update_system_prompt("sess-1", stored)
    db.append_message("sess-1", "user", content="earlier")
    db.append_message("sess-1", "assistant", content="earlier answer")

    def fresh_agent():
        return AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            provider="openai-compat", model="test-model", max_iterations=3, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="slack", session_db=db, session_id="sess-1",
        )

    try:
        first_agent = fresh_agent()
        result = first_agent.run_conversation(
            "first question", conversation_history=db.get_messages_as_conversation("sess-1"), task_id="slack-first"
        )
        assert result["final_response"] == "answer"
        sent = [request for request in requests if "messages" in request][-1]
        assert sent["messages"][0] == {"role": "system", "content": stored}
        current = [msg for msg in sent["messages"] if msg["role"] == "user"][-1]
        assert NOTE_PREFIX in current["content"]
        assert platform_hint(first_agent) in current["content"]
        assert all("api_content" not in msg for msg in sent["messages"])
        durable = db.get_messages_as_conversation("sess-1")
        first_row = next(msg for msg in durable if msg.get("content") == "first question")
        assert first_row["api_content"] == current["content"]
        assert db.get_session("sess-1")["system_prompt"] == stored

        second_agent = fresh_agent()
        result = second_agent.run_conversation("second question", conversation_history=durable, task_id="slack-second")
        assert result["final_response"] == "answer"
        replay = [request for request in requests if "messages" in request][-1]
        users = [msg for msg in replay["messages"] if msg["role"] == "user"]
        assert users[-2]["content"] == current["content"]
        assert users[-1]["content"] == "second question"
        assert replay["messages"][0] == sent["messages"][0]
        assert replay.get("tools") == sent.get("tools")
        second_row = next(msg for msg in db.get_messages_as_conversation("sess-1")
                          if msg.get("content") == "second question")
        assert "api_content" not in second_row
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
