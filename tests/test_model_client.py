"""OpenAI-compatible client: provider-side badness must become visible feedback,
never a crash of the run."""
import io
import json
import urllib.error
import urllib.request

import pytest

from ama.model import Message, ModelError, ModelTimeoutError, OpenAICompatModel


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def patch_urlopen(monkeypatch, body):
    def fake(req, timeout=None):
        return FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def client():
    return OpenAICompatModel("test", base_url="https://fake/api/v4", model="m",
                             api_key="k", request_timeout=5.0)


def _msgs():
    return [Message(role="system", content="s"), Message(role="user", content="q")]


def test_extra_field_tool_payload_feedback_names_the_field(monkeypatch):
    # regression: GLM sent submit_decision with an extra key inside `decision`;
    # the feedback must name the offending field so the model repairs in one retry,
    # and must never kill the process
    patch_urlopen(monkeypatch, {"choices": [{"message": {"tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "submit_decision", "arguments": json.dumps({
            "decision": {"turn_id": "t2", "state": {"x": 1}, "abstain": False,
                         "next_target": "pathology"},   # <-- extra field
        })},
    }]}}], "usage": {"total_tokens": 5}})
    out = client().next(_msgs())
    assert isinstance(out, str)
    from ama.agent import _to_action
    action, err = _to_action(out)
    assert action is None
    assert "Extra inputs" in err and "next_target" in err  # precise, field-level feedback


def test_missing_required_key_feedback_names_the_key(monkeypatch):
    patch_urlopen(monkeypatch, {"choices": [{"message": {"tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "submit_decision", "arguments": json.dumps({
            "decision": {"state": {}, "abstain": False},  # turn_id missing
        })},
    }]}}], "usage": None})
    out = client().next(_msgs())
    from ama.agent import _to_action
    action, err = _to_action(out)
    assert action is None and "turn_id" in err and "required" in err.lower()


def test_malformed_tool_call_shape_returns_raw(monkeypatch):
    patch_urlopen(monkeypatch, {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function", "function": {"arguments": "not json{"}}]}}],
        "usage": None})
    out = client().next(_msgs())
    assert isinstance(out, str)


def test_text_only_reply_is_raw_string(monkeypatch):
    patch_urlopen(monkeypatch, {"choices": [{"message": {"content": "我认为是 t1"}}],
                                "usage": {"total_tokens": 3}})
    c = client()
    out = c.next(_msgs())
    assert out == "我认为是 t1" and c.last_usage["total_tokens"] == 3


def test_http_error_and_timeout_map_to_model_error(monkeypatch):
    def boom_http(req, timeout=None):
        raise urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(b'{"e":1}'))

    monkeypatch.setattr(urllib.request, "urlopen", boom_http)
    with pytest.raises(ModelError) as ei:
        client().next(_msgs())
    assert ei.value.kind == "http_error" and "k" not in str(ei.value)

    def boom_timeout(req, timeout=None):
        raise TimeoutError()

    monkeypatch.setattr(urllib.request, "urlopen", boom_timeout)
    with pytest.raises(ModelTimeoutError):
        client().next(_msgs())
