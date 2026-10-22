import hashlib
import json
from pathlib import Path

import pytest

from ama import image_analysis_tools as vision
from ama.agent import Agent
from ama.data import Turn
from ama.model import ModelError
from ama.runner import execute, render
from ama.tools import load_tools
from fakes import FakeModel, call, calls, decision
from test_data import write_dataset


def files(tmp_path):
    paths = {}
    for key in ("image_01", "image_02"):
        path = tmp_path / f"{key}.png"
        path.write_bytes(b"image fixture: " + key.encode())
        paths[key] = path
    return paths


def payload(tool, query="Compare the two images.", ids=None):
    return tool.invoke(json.dumps({"query": query, "image_ids": ids or ["image_01", "image_02"]}))


def test_visual_expert_is_stateless_multimodal_and_returns_only_text(tmp_path):
    paths = files(tmp_path)
    expert = FakeModel("image_01: first view. image_02: second view.", "Uncertain finding in image_02.")
    expert.last_usage = {"total_tokens": 123}
    tool = vision.make_image_analysis_tool(model_factory=lambda: expert)
    tool.on_images(paths)
    first = payload(tool)
    result = json.loads(first.text)
    assert first.images == [] and first.evidence_ids == ["image_01", "image_02"]
    assert result["image_ids"] == ["image_01", "image_02"] and result["usage"]["total_tokens"] == 123
    assert result["images"][0]["sha256"] == hashlib.sha256(paths["image_01"].read_bytes()).hexdigest()
    history, definitions = expert.requests[0]
    assert definitions is None and [row["role"] for row in history] == ["system", "user"]
    parts = history[1]["content"]
    assert [part["type"] for part in parts] == ["text", "text", "image", "text", "image"]
    assert parts[1]["text"] == "Image ID: image_01" and parts[3]["text"] == "Image ID: image_02"
    payload(tool, query="Inspect only the second image.", ids=["image_02"])
    assert len(expert.requests[1][0]) == 2
    assert "Compare the two images" not in json.dumps(expert.requests[1][0])
    assert "image_01" not in json.dumps(expert.requests[1][0])


def test_identical_requests_are_cached_but_changed_pixels_are_not(tmp_path):
    paths = files(tmp_path)
    expert = FakeModel("First answer", "Changed-image answer")
    tool = vision.make_image_analysis_tool(model_factory=lambda: expert)
    tool.on_images(paths)
    assert not json.loads(payload(tool).text)["cached"]
    cached = json.loads(payload(tool).text)
    assert cached["cached"] and cached["usage"] is None and cached["duration_ms"] == 0
    assert len(expert.requests) == 1
    paths["image_01"].write_bytes(b"changed pixels")
    assert not json.loads(payload(tool).text)["cached"]
    assert len(expert.requests) == 2


@pytest.mark.parametrize("arguments", [
    {"query": "", "image_ids": ["image_01"]},
    {"query": "x", "image_ids": []}, {"query": "x", "image_ids": "image_01"},
    {"query": "x", "image_ids": ["image_01", "image_01"]},
    {"query": "x", "image_ids": ["future-image"]},
    {"query": "x", "image_ids": ["../another-case.png"]},
    {"query": "x" * 4001, "image_ids": ["image_01"]},
])
def test_invalid_or_unreleased_inputs_never_reach_expert(tmp_path, arguments):
    tool = vision.make_image_analysis_tool(model_factory=lambda: pytest.fail("Unexpected model creation"))
    tool.on_images(files(tmp_path))
    with pytest.raises(ValueError):
        tool.invoke(json.dumps(arguments))


def test_upload_size_and_format_are_checked_before_model_creation(tmp_path, monkeypatch):
    tool = vision.make_image_analysis_tool(model_factory=lambda: pytest.fail("Unexpected model creation"))
    paths = files(tmp_path)
    tool.on_images(paths)
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(ValueError, match="24 MiB"):
        payload(tool)
    unsupported = tmp_path / "animation.gif"
    unsupported.write_bytes(b"gif")
    tool.on_images({"gif": unsupported})
    with pytest.raises(ValueError, match="PNG/JPG"):
        payload(tool, ids=["gif"])


def test_failed_expert_requests_return_errors_and_consume_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "MAX_EXPERT_CALLS", 1)
    expert = FakeModel(ModelError("model endpoint returned HTTP 429", http_status=429))
    tool = vision.make_image_analysis_tool(model_factory=lambda: expert)
    tool.on_images(files(tmp_path))
    result = payload(tool)
    assert "HTTP 429" in json.loads(result.text)["error"] and result.evidence_ids == []
    with pytest.raises(ValueError, match="request limit reached"):
        payload(tool)
    assert len(expert.requests) == 1


@pytest.mark.parametrize("reply", ["", calls(call("unexpected"))])
def test_invalid_expert_replies_do_not_release_evidence(tmp_path, reply):
    tool = vision.make_image_analysis_tool(model_factory=lambda: FakeModel(reply))
    tool.on_images(files(tmp_path))
    result = payload(tool)
    assert "error" in json.loads(result.text) and result.evidence_ids == []


def test_long_analysis_is_bounded(tmp_path):
    tool = vision.make_image_analysis_tool(model_factory=lambda: FakeModel("a" * 7000))
    tool.on_images(files(tmp_path))
    result = json.loads(payload(tool).text)
    assert len(result["analysis"]) == 6000 and result["truncated"]


@pytest.mark.parametrize("tools_path", ["src/ama/image_analysis_tools.py", "src/ama/medical_tools.py"])
def test_factory_binds_fresh_image_state_per_episode(tmp_path, tools_path):
    first = next(t for t in load_tools(tools_path, episode_id="first") if t.name == "analyze_images")
    second = next(t for t in load_tools(tools_path, episode_id="second") if t.name == "analyze_images")
    first.on_images(files(tmp_path))
    with pytest.raises(ValueError, match="not been released"):
        payload(second)


def test_default_reader_uses_deepseek_flash(monkeypatch):
    monkeypatch.delenv("AMA_VISION_MODEL", raising=False)
    monkeypatch.delenv("AMA_VISION_CONFIG", raising=False)
    seen = []
    monkeypatch.setattr(vision.Model, "from_config", lambda *args, **kwargs: seen.append((args, kwargs)))
    vision._default_model()
    assert seen == [(("deepseek-flash", "ama.json"), {"timeout": 60})]


@pytest.mark.parametrize("mode", ["run", "chat"])
def test_runner_delegates_images_hides_future_evidence_and_keeps_main_text_only(tmp_path, monkeypatch, mode):
    paths = files(tmp_path)
    episode = {"id": "ep", "turns": [
        {"id": "t1", "evidence": [{"id": "image_01", "file": paths["image_01"].name}]},
        {"id": "t2", "evidence": [{"id": "image_02", "file": paths["image_02"].name}]},
    ]}
    write_dataset(tmp_path, [episode])
    (tmp_path / "instructions.txt").write_text("Describe the visible images.", encoding="utf-8")
    expert = FakeModel("image_01: observation.", "image_02: observation.")
    tool = vision.make_image_analysis_tool(model_factory=lambda: expert)
    monkeypatch.setattr("ama.runner.load_tools", lambda *args, **kwargs: [tool])
    original = Path.open

    def guard(path, *args, **kwargs):
        assert path.name not in {"targets.jsonl", "eval.json", "provenance.jsonl"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    model = FakeModel(
        calls(call("analyze_images", '{"query":"Inspect future image","image_ids":["image_02"]}', "future")),
        calls(call("analyze_images", '{"query":"Inspect first image","image_ids":["image_01"]}', "first")),
        decision(citations=["image_01"]),
        calls(call("analyze_images", '{"query":"Inspect second image","image_ids":["image_02"]}', "second")),
        decision("t2", citations=["image_02"]),
    )
    inputs = iter(["/next", "/quit"])
    run = execute(tmp_path, model, mode=mode, episode_ids=["ep"], runs_root=tmp_path / "runs",
                  tools_path="src/ama/image_analysis_tools.py", input_fn=lambda _: next(inputs))
    history = json.loads((run / "messages.json").read_text(encoding="utf-8"))["ep"]
    assert len(expert.requests) == 2
    assert "image_02" not in expert.requests[0][0][1]["content"][0]["text"]
    assert all(part.get("type") != "image" for row in history if isinstance(row.get("content"), list)
               for part in row["content"])
    results = [json.loads(row["content"].split("\nCiteable")[0]) for row in history if row["role"] == "tool"]
    assert "error" in results[0] and results[1]["analysis"] == "image_01: observation."
    assert "image_02" not in json.dumps(model.requests[0][0])
    if mode == "run":
        assert "Currently citeable evidence IDs: []" in json.dumps(model.requests[0][0])
        records = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
        assert all(row["termination"] == "completed" for row in records)


def test_default_render_still_delivers_original_image(tmp_path):
    path = files(tmp_path)["image_01"]
    turn = Turn.model_validate({"id": "t1", "evidence": [{"id": "image_01", "file": path.name}]})
    assert any(isinstance(part, dict) and part.get("type") == "image" for part in render(turn, tmp_path))
