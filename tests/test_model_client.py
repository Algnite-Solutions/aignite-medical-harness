"""Provider transport does not choose or parse the interaction protocol."""
import json
import urllib.request

from ama.model import Message, OpenAICompatModel, resolve_model_config


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
    tools = [{"type": "function", "function": {"name": "example_tool",
              "parameters": {"type": "object", "properties": {}}}}]

    def fake(req, timeout=None):
        captured.update(json.loads(req.data))
        return FakeResponse(json.dumps({"choices": [{"message": {"content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "example_tool", "arguments": "{}"}}]}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    model = OpenAICompatModel("m", "https://example.test/v1", "provider", "secret")
    result = model.next([Message(role="user", content="hello")], tools=tools)
    assert captured["tools"] == tools
    assert result["tool_calls"][0]["function"]["name"] == "example_tool"


def test_provider_alias_does_not_encode_protocol():
    cfg = resolve_model_config("qwen36")
    assert cfg.model == "Qwen3.6-27B"
    assert cfg.wire == "openai_compact_image"
    assert "response_mode" not in cfg.model_dump()
