"""Offline transport fixtures only; no provider credentials or patient requests."""
import copy
import json
import urllib.error
import urllib.request

import pytest

from ama.agent import Agent
from ama.decisions import normalize_decision
from ama.model import Model, ModelError
from ama.tools import Tool, ToolResult
from test_model_client import Response

ALIAS = "qwen36-35b-a3b-bailian"
WIRE_MODEL = "qwen3.6-35b-a3b"


def client(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-fixture-key")
    return Model.from_config(ALIAS)


def response(message, model=WIRE_MODEL, **changes):
    return Response(json.dumps({"model": model, "id": "fixture-response",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"total_tokens": 7}, **changes}).encode())


def test_bailian_alias_and_top_level_thinking(monkeypatch):
    existing = json.loads(open("ama.json").read())["models"]["qwen36"]
    assert existing["model"] == "Qwen3.6-27B"
    assert existing["chat_template_kwargs"] == {"enable_thinking": True}
    def respond(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert payload["model"] == WIRE_MODEL
        assert payload["enable_thinking"] is False
        assert payload["temperature"] == 0 and payload["max_tokens"] == 4096
        assert "chat_template_kwargs" not in payload
        assert "require_response_model" not in payload
        return response({"role": "assistant", "content": "fixture"})
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    m = client(monkeypatch)
    assert m.complete([])["content"] == "fixture"
    assert m.last_response_metadata == {"model": WIRE_MODEL, "id": "fixture-response",
        "finish_reason": "stop", "logprobs_available": False}
    assert "offline-fixture-key" not in json.dumps(m.config)


@pytest.mark.parametrize("reported", ["Qwen3.6-27B", None])
def test_wrong_or_missing_response_model_stops(monkeypatch, reported):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw:
        response({"role": "assistant", "content": "fixture"}, model=reported))
    m = client(monkeypatch)
    with pytest.raises(ModelError, match="response model"):
        m.complete([])
    assert m.last_response_metadata["model"] == reported


def test_metadata_is_cleared_after_transport_failure(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw:
        response({"role": "assistant", "content": "fixture"}))
    m = client(monkeypatch)
    m.complete([])
    def fail(*a, **kw):
        raise urllib.error.URLError("fixture failure")
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ModelError):
        m.complete([])
    assert m.last_response_metadata is None and m.last_usage is None


@pytest.mark.parametrize("option", ["enable_thinking", "require_response_model"])
def test_new_flags_require_booleans(monkeypatch, option):
    config = {"base_url": "https://example.test", "model": "fixture", "api_key_env": "TEST_KEY", option: "false"}
    with pytest.raises(ValueError, match="boolean"):
        Model("fixture", config)


def test_conflicting_thinking_locations_rejected():
    with pytest.raises(ValueError, match="one request location"):
        Model("fixture", {"base_url": "https://example.test", "model": "fixture", "api_key_env": "TEST_KEY",
            "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}})


def test_tool_round_trip_preserves_evidence_and_nested_confidence(monkeypatch):
    requests = []
    decision = {"turn_id": "t1", "answer": {"diagnosis": "synthetic diagnosis", "confidence": 0.7},
        "citations": ["lab-1"], "reasoning_summary": "Synthetic findings."}
    tool_call = {"id": "fixture-call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    def respond(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        if len(requests) == 1:
            return response({"role": "assistant", "content": None, "tool_calls": [tool_call]})
        assert payload["messages"][2]["tool_calls"] == [tool_call]
        assert payload["messages"][3]["tool_call_id"] == "fixture-call"
        assert "lab-1" in payload["messages"][3]["content"]
        return response({"role": "assistant", "content": json.dumps(decision)})
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    tool = Tool("lookup", "Synthetic lookup", {}, lambda: ToolResult("Synthetic lab.", evidence_ids=["lab-1"]))
    agent = Agent(client(monkeypatch), system="Synthetic fixture only.", tools=[tool])
    raw = agent.chat("Synthetic observation.")
    history = copy.deepcopy(agent.history)
    parsed, checks = normalize_decision(raw, "t1", agent.evidence_ids)
    assert parsed == decision and checks["strict_compliance"]
    assert len(agent.calls) == 2
    assert all(c["response_metadata"]["model"] == WIRE_MODEL for c in agent.calls)
    assert agent.evidence_releases == [{"tool_call_id": "fixture-call", "name": "lookup", "evidence_ids": ["lab-1"]}]
    assert agent.history == history


@pytest.mark.parametrize("alias,wire", [
    ("qwen36-35b-a3b-bailian-thinking", WIRE_MODEL),
    ("qwen36-27b-bailian-thinking", "qwen3.6-27b"),
])
def test_thinking_alias_round_trip(monkeypatch, alias, wire):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-fixture-key")
    payloads = []
    tool_call = {"id": "bounded-thinking-call", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}}
    def respond(request, timeout):
        payload = json.loads(request.data)
        payloads.append(payload)
        assert payload["model"] == wire
        assert payload["enable_thinking"] is True
        assert payload["thinking_budget"] == 2048
        assert payload["max_tokens"] == 4096
        assert "chat_template_kwargs" not in payload
        if len(payloads) == 1:
            msg = {"role": "assistant", "content": "", "reasoning_content": "Synthetic reasoning.",
                   "tool_calls": [tool_call]}
        else:
            assert payload["messages"][2]["tool_calls"] == [tool_call]
            assert payload["messages"][3]["tool_call_id"] == tool_call["id"]
            assert "SECRET_SYNTH_91" in payload["messages"][3]["content"]
            msg = {"role": "assistant", "content": "SECRET_SYNTH_91"}
        return response(msg, model=wire, usage={"prompt_tokens": 10, "completion_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 8}})
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    tool = Tool("lookup", "Synthetic lookup", {}, lambda: ToolResult("SECRET_SYNTH_91"))
    agent = Agent(Model.from_config(alias), system="Synthetic tool test.", tools=[tool])
    assert agent.chat("Retrieve the hidden marker.") == "SECRET_SYNTH_91"
    assert len(agent.calls) == 2
    assert agent.calls[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 8


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "2048"])
def test_invalid_thinking_budget_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        Model("fixture", {"base_url": "https://example.test", "model": "fixture",
            "api_key_env": "TEST_KEY", "enable_thinking": True, "thinking_budget": budget})


def test_thinking_budget_requires_explicit_mode():
    with pytest.raises(ValueError, match="requires top-level"):
        Model("fixture", {"base_url": "https://example.test", "model": "fixture",
            "api_key_env": "TEST_KEY", "enable_thinking": False, "thinking_budget": 2048})
