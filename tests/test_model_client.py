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


def test_multimodal_user_content_is_preserved_on_wire():
    content = [
        {"type": "text", "text": "image evidence e1"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YQ=="}},
    ]
    model = OpenAICompatModel("t", "https://x/v1", "m", "k")  # default wire keeps content verbatim
    wire = model._to_wire([Message(role="user", content=content)])
    assert wire == [{"role": "user", "content": content}]


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


def _image_history_messages():
    from ama.model import Message
    tool_call = [{"id": "c1", "type": "function",
                  "function": {"name": "read_evidence", "arguments": "{}"}}]
    return [
        Message(role="system", content="sys prompt"),
        Message(role="user", content="[turn_id=t1 | 第 1/1 轮 world observation @ x]\nReview the image."),
        Message(role="assistant", content="{}", tool_calls=tool_call),
        Message(role="tool", tool_call_id="c1", content='{"ok": true}'),
        Message(role="user", content=[
            {"type": "text", "text": "[image evidence: evidence_id=e1]"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA", "detail": "auto"}},
        ]),
    ]


def test_wire_compact_image_drops_tool_history_before_image():
    m = OpenAICompatModel("t", "https://x/v1", "m", "k", wire="openai_compact_image")
    wire = m._to_wire(_image_history_messages())
    assert [w["role"] for w in wire] == ["system", "user"]  # no assistant/tool pairs before image
    parts = wire[1]["content"]
    assert parts[0]["type"] == "text"
    assert "successfully read" in parts[0]["text"] and "Review the image." in parts[0]["text"]
    assert parts[-1]["type"] == "image_url" and parts[-1]["image_url"]["url"].startswith("data:image/jpeg")


def test_wire_compact_keeps_messages_after_image():
    from ama.model import Message
    m = OpenAICompatModel("t", "https://x/v1", "m", "k", wire="openai_compact_image")
    msgs = _image_history_messages() + [
        Message(role="assistant", content="{}", tool_calls=[{"id": "c2", "type": "function",
                                                              "function": {"name": "submit_decision", "arguments": "{}"}}]),
        Message(role="tool", tool_call_id="c2", content='{"ok": true}'),
    ]
    wire = m._to_wire(msgs)
    assert [w["role"] for w in wire] == ["system", "user", "assistant", "tool"]


def test_default_wire_keeps_tool_pairs_and_no_image_passthrough():
    m = OpenAICompatModel("t", "https://x/v1", "m", "k")  # wire defaults to "openai"
    wire = m._to_wire(_image_history_messages())
    assert [w["role"] for w in wire] == ["system", "user", "assistant", "tool", "user"]
    m2 = OpenAICompatModel("t", "https://x/v1", "m", "k", wire="openai_compact_image")
    plain = [type("Msg", (), {})()]  # not needed; just ensure no image -> unchanged
    from ama.model import Message
    no_img = [Message(role="system", content="s"), Message(role="user", content="plain"),
              Message(role="assistant", content="{}", tool_calls=[{"id": "c", "type": "function",
                                                                   "function": {"name": "list_evidence", "arguments": "{}"}}]),
              Message(role="tool", tool_call_id="c", content="{}")]
    assert [w["role"] for w in wire] == ["system", "user", "assistant", "tool", "user"]
    assert len(m2._to_wire(no_img)) == 4  # untouched when no image present


def test_unknown_wire_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="unknown wire format"):
        OpenAICompatModel("t", "https://x/v1", "m", "k", wire="grpc")
