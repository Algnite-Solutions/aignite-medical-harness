import json
import urllib.error
import urllib.request

import pytest

from ama.model import Model, ModelError, image


class Response:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def read(self):
        return self.value


def model(monkeypatch, **kwargs):
    monkeypatch.setenv("TEST_KEY", "secret-not-logged")
    return Model("test", {"base_url": "https://example.test/v1", "model": "provider",
                          "api_key_env": "TEST_KEY", **kwargs})


def test_request_response_timeout_and_no_tools(monkeypatch):
    requests = []
    def respond(request, timeout):
        requests.append((json.loads(request.data), timeout))
        return Response(b'{"choices":[{"message":{"role":"assistant","content":"raw"}}],"usage":{"total_tokens":4}}')
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    client = model(monkeypatch, chat_template_kwargs={"enable_thinking": False})
    assert client.complete([{"role": "user", "content": "hi"}])["content"] == "raw"
    payload, timeout = requests[0]
    assert timeout == 60 and "tools" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.last_usage == {"total_tokens": 4}
    assert "secret-not-logged" not in json.dumps(client.config)


def test_tools_and_images_preserve_roles_and_ids(monkeypatch, tmp_path):
    path = tmp_path / "scan.jpg"
    path.write_bytes(b"fixture")
    history = [{"role": "user", "content": "start"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
               {"role": "tool", "tool_call_id": "x", "content": "scan"},
               {"role": "user", "content": [image(path)]}]
    def respond(request, timeout):
        payload = json.loads(request.data)
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "tool", "user"]
        assert payload["messages"][2]["tool_call_id"] == "x"
        assert "base64" in payload["messages"][3]["content"][0]["image_url"]["url"]
        assert payload["tools"] == [{"type": "function"}]
        return Response(b'{"choices":[{"message":{"content":"done"}}]}')
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    model(monkeypatch).complete(history, tools=[{"type": "function"}])
    assert "base64" not in json.dumps(history)


@pytest.mark.parametrize("body", [b"invalid", b"{}", b"null", b'{"choices":[]}'])
def test_bad_api_responses_raise(monkeypatch, body):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response(body))
    with pytest.raises(ModelError):
        model(monkeypatch).complete([])


def test_http_tool_image_limit_is_explicit_no_rewrite_or_retry(monkeypatch):
    seen = []
    def fail(*a, **kw):
        seen.append(a)
        raise urllib.error.HTTPError("https://example.test", 400, "bad", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ModelError, match="history was not rewritten"):
        model(monkeypatch).complete([{"role": "tool", "content": "image"}, {"role": "user", "content": []}])
    assert len(seen) == 1


def test_registration_rejects_old_wire_and_missing_key(monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        model(monkeypatch, wire="compact")
    monkeypatch.delenv("MEDAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MEDAI_API_KEY"):
        Model.from_config("qwen36")
    with pytest.raises(ValueError, match="unknown model"):
        Model.from_config("not-registered")
