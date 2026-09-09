"""Offline unit tests for the OpenAI-compatible client: wire format, tool
definitions, key handling, error mapping, usage capture. No network."""
import io
import json
import urllib.error
import urllib.request

import pytest

from medical_harness.model import (
    Message,
    ModelError,
    ModelTimeoutError,
    OpenAICompatModel,
    make_model_factory,
    tool_definitions,
)
from medical_harness.schemas import ModelConfig, SubmitAnswerAction

KEY = "test-key-123"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client():
    return OpenAICompatModel(
        name="test", base_url="https://fake.example/api/v4/", model="m1",
        api_key=KEY, state_tools_enabled=True, request_timeout=5.0,
    )


def make_reply(tool_call=None, content=None, usage=None):
    message = {}
    if tool_call is not None:
        message["tool_calls"] = [{
            "id": "abc", "type": "function",
            "function": {"name": tool_call[0], "arguments": json.dumps(tool_call[1], ensure_ascii=False)},
        }]
    if content is not None:
        message["content"] = content
    body = {"choices": [{"message": message}], "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    return FakeResponse(json.dumps(body).encode("utf-8"))


def patch_urlopen(monkeypatch, response_or_exc, capture):
    def fake_urlopen(req, timeout=None):
        capture.append(req)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def base_messages():
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="问题"),
        Message(role="assistant", content="{}", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "list_evidence", "arguments": "{}"},
        }]),
        Message(role="tool", tool_call_id="call_1", content='{"ok": true}'),
    ]


def test_wire_format_and_headers(client, monkeypatch):
    cap = []
    patch_urlopen(monkeypatch, make_reply(tool_call=("list_evidence", {})), cap)
    client.next(base_messages())
    req = cap[0]
    assert req.full_url == "https://fake.example/api/v4/chat/completions"
    assert req.get_header("Authorization") == f"Bearer {KEY}"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "m1"
    wire = payload["messages"]
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[2]["tool_calls"][0]["id"] == "call_1"
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["list_evidence", "read_evidence", "get_state", "propose_state_update", "submit_answer"]


def test_tool_definitions_respect_strategy():
    names = [t["function"]["name"] for t in tool_definitions(state_tools_enabled=False)]
    assert names == ["list_evidence", "read_evidence", "submit_answer"]


def test_submit_tool_call_parsed_to_typed_action(client, monkeypatch):
    cap = []
    patch_urlopen(monkeypatch, make_reply(tool_call=("submit_answer", {
        "answer": {"question_id": "q1", "abstain": True, "reason": "无可见证据"},
    })), cap)
    action = client.next(base_messages())
    assert isinstance(action, SubmitAnswerAction)
    assert action.answer.abstain is True
    assert client.last_usage["total_tokens"] == 15


def test_text_only_reply_returned_as_raw_string(client, monkeypatch):
    patch_urlopen(monkeypatch, make_reply(content="我认为答案是 1.8"), [])
    out = client.next(base_messages())
    assert isinstance(out, str) and "1.8" in out  # agent loop turns this into a visible invalid-action error


def test_malformed_tool_args_returned_as_raw(client, monkeypatch):
    bad = FakeResponse(json.dumps({"choices": [{"message": {"tool_calls": [{
        "id": "x", "type": "function",
        "function": {"name": "read_evidence", "arguments": "{\"evidence_id\": 123, \"junk\": true}"},
    }]}}]}).encode())
    patch_urlopen(monkeypatch, bad, [])
    out = client.next(base_messages())
    assert isinstance(out, str)  # not a typed action: schema validation rejects it downstream, visibly


def test_http_error_is_model_error_without_key_leak(client, monkeypatch):
    body = json.dumps({"error": {"message": "bad request"}}).encode()
    patch_urlopen(monkeypatch, urllib.error.HTTPError("url", 400, "Bad Request", {}, io.BytesIO(body)), [])
    with pytest.raises(ModelError) as exc_info:
        client.next(base_messages())
    assert exc_info.value.kind == "http_error"
    assert "400" in str(exc_info.value)
    assert KEY not in str(exc_info.value)


def test_timeout_raises_model_timeout(client, monkeypatch):
    patch_urlopen(monkeypatch, TimeoutError(), [])
    with pytest.raises(ModelTimeoutError):
        client.next(base_messages())


def test_factory_requires_key_and_url(monkeypatch):
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    cfg = ModelConfig(type="openai_compatible", base_url="https://x", model="m", api_key_env="GLM_API_KEY")
    with pytest.raises(ValueError, match="missing API key"):
        make_model_factory(cfg, None)
    monkeypatch.setenv("GLM_API_KEY", "k")
    with pytest.raises(ValueError, match="base_url and model"):
        make_model_factory(ModelConfig(type="openai_compatible", api_key_env="GLM_API_KEY"), None)
    factory = make_model_factory(cfg, None)
    m = factory("case_beta", "q1")
    assert m.model == "m" and m.api_key == "k"


def test_factory_unknown_type_rejected():
    with pytest.raises(ValueError, match="unsupported model type"):
        make_model_factory(ModelConfig(type="mystery"), None)
