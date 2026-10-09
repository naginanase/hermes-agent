"""Source include is an explicit Codex request probe, never a public-spec assumption."""
from copy import deepcopy
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from agent.codex_runtime import run_codex_stream
from agent.transports.codex import ResponsesApiTransport

SOURCE_INCLUDE = "web_search_call.action.sources"


def agent(codex=True):
    return NS(session_id="", provider="openai-codex", model="test", _interrupt_requested=False,
              _last_api_first_chunk_at=None, _touch_activity=lambda *_: None,
              _fire_stream_delta=lambda *_: None, _fire_reasoning_delta=lambda *_: None,
              _client_log_context=lambda: "", _is_codex_backend=lambda: codex,
              _current_api_request_id="req_probe")


def rejection(message, param="include"):
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"error": {"message": message, "param": param}})


def completed():
    return iter([NS(type="response.output_text.delta", delta="ok"),
                 NS(type="response.completed", response=NS(id="resp_probe", status="completed", output=None))])


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_source_include_probe_has_one_targeted_pre_stream_fallback(nested):
    calls = []
    includes = ["reasoning.encrypted_content", SOURCE_INCLUDE]
    kwargs = {"model": "test", "input": [{"role": "user", "content": "private prompt"}],
              "tools": [{"type": "web_search"}], "include": includes}
    if nested:
        kwargs["extra_body"] = {"include": list(includes), "other": "preserve"}
    before = deepcopy(kwargs)

    def create(**wire):
        calls.append(deepcopy(wire))
        if len(calls) == 1:
            raise rejection("Unsupported include value: " + SOURCE_INCLUDE)
        return completed()

    response = run_codex_stream(agent(), kwargs, client=NS(responses=NS(create=create)))
    assert response.output_text == "ok"
    assert len(calls) == 2
    assert kwargs == before
    assert calls[0]["include"] == includes
    assert calls[1]["include"] == ["reasoning.encrypted_content"]
    if nested:
        assert calls[1]["extra_body"]["include"] == ["reasoning.encrypted_content"]
        assert calls[1]["extra_body"]["other"] == "preserve"
        assert calls[1]["extra_body"]["input"] == kwargs["input"]
        assert calls[1]["extra_body"]["tools"] == kwargs["tools"]


@pytest.mark.parametrize("location", ["top-level", "extra_body", "both"])
def test_rejected_source_include_stays_omitted_after_fallback_connect_failure(location):
    calls = []
    includes = ["reasoning.encrypted_content", SOURCE_INCLUDE, "message.output_text.logprobs"]
    kwargs = {"model": "test", "input": [{"role": "user", "content": "private prompt"}],
              "tools": [{"type": "web_search"}], "store": False,
              "reasoning": {"effort": "high"}, "extra_body": {"other": "preserve"}}
    if location in ("top-level", "both"):
        kwargs["include"] = list(includes)
    if location in ("extra_body", "both"):
        kwargs["extra_body"]["include"] = list(includes)
    before = deepcopy(kwargs)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")

    def create(**wire):
        calls.append(deepcopy(wire))
        if any(SOURCE_INCLUDE in container.get("include", [])
               for container in (wire, wire.get("extra_body", {}))):
            raise rejection("Unsupported include value: " + SOURCE_INCLUDE)
        if len(calls) == 2:
            raise APIConnectionError(request=request) from httpx.ConnectError("pre-stream failure", request=request)
        return completed()

    response = run_codex_stream(agent(), kwargs, client=NS(responses=NS(create=create)))

    assert response is not None
    assert response.output_text == "ok"
    assert kwargs == before
    assert len(calls) == 3
    expected = deepcopy(calls[0])
    for container in (expected, expected["extra_body"]):
        if "include" in container:
            container["include"] = [entry for entry in includes if entry != SOURCE_INCLUDE]
    assert calls[1:] == [expected, expected]


def test_source_include_is_not_generated_without_explicit_route_request_override():
    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs("gpt-5-codex", [{"role": "user", "content": "look"}],
                                    is_codex_backend=True)
    assert SOURCE_INCLUDE not in kwargs.get("include", [])
    override = transport.build_kwargs("gpt-5-codex", [{"role": "user", "content": "look"}],
                                      is_codex_backend=True, request_overrides={"include": [SOURCE_INCLUDE]})
    assert override["include"] == [SOURCE_INCLUDE]


@pytest.mark.parametrize("codex,message,param", [
    (False, "Unsupported include value: " + SOURCE_INCLUDE, "include"),
    (True, "Unsupported include value: reasoning.encrypted_content", "include"),
    (True, "Quota exceeded while requesting " + SOURCE_INCLUDE, "quota"),
])
def test_optional_include_fallback_does_not_mask_unrelated_errors_or_other_routes(codex, message, param):
    calls = []
    error = rejection(message, param)

    def create(**wire):
        calls.append(wire)
        raise error

    with pytest.raises(BadRequestError) as caught:
        run_codex_stream(agent(codex), {"model": "test", "include": [SOURCE_INCLUDE]}, client=NS(responses=NS(create=create)))
    assert caught.value is error
    assert len(calls) == 1


def test_optional_include_fallback_is_bounded_even_if_fallback_is_also_rejected():
    calls = []
    error = rejection("Unsupported include value: " + SOURCE_INCLUDE)

    def create(**wire):
        calls.append(wire)
        raise error

    with pytest.raises(BadRequestError):
        run_codex_stream(agent(), {"model": "test", "include": [SOURCE_INCLUDE]}, client=NS(responses=NS(create=create)))
    assert len(calls) == 2
    assert calls[1]["include"] == []


def test_accepted_include_is_not_retried_or_treated_as_globally_supported():
    calls = []

    def create(**wire):
        calls.append(wire)
        return completed()

    run_codex_stream(agent(), {"model": "test", "include": [SOURCE_INCLUDE]}, client=NS(responses=NS(create=create)))
    run_codex_stream(agent(), {"model": "test"}, client=NS(responses=NS(create=create)))
    assert len(calls) == 2
    assert calls[0]["include"] == [SOURCE_INCLUDE]
    assert "include" not in calls[1]
