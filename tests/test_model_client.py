"""Provider transport does not choose or parse the interaction protocol."""
import json
import urllib.request

from ama.model import Message, OpenAICompatModel, resolve_model_config
from ama.protocols import EVIDENCE_TOOLS


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return self.data


def test_no_tools_by_default_and_raw_response(monkeypatch):
    captured = {}

    def fake(req, timeout=None):
        captured.update(json.loads(req.data))
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "not parsed here"}}],
                                        "usage": {"total_tokens": 4}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    model = OpenAICompatModel("m", "https://example.test/v1", "provider", "secret",
                              chat_template_kwargs={"enable_thinking": False})
    result = model.next([Message(role="user", content="hello")])
    assert result == {"content": "not parsed here"}
    assert "tools" not in captured
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert model.last_usage["total_tokens"] == 4


def test_tool_choice_is_per_request(monkeypatch):
    captured = {}

    def fake(req, timeout=None):
        captured.update(json.loads(req.data))
        return FakeResponse(json.dumps({"choices": [{"message": {"content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "list_evidence", "arguments": "{}"}}]}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    model = OpenAICompatModel("m", "https://example.test/v1", "provider", "secret")
    result = model.next([Message(role="user", content="hello")], tools=EVIDENCE_TOOLS)
    assert captured["tools"] == EVIDENCE_TOOLS
    assert result["tool_calls"][0]["function"]["name"] == "list_evidence"


def test_provider_alias_does_not_encode_protocol():
    cfg = resolve_model_config("qwen36-json")
    assert cfg.model == "Qwen3.6-27B"
    assert cfg.chat_template_kwargs == {"enable_thinking": False}
    assert "response_mode" not in cfg.model_dump()
