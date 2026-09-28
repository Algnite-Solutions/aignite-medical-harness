import json
from pathlib import Path

import pytest

from ama.cli import main, _eval_run
from ama.model import Model, ModelError
from ama.runner import execute, parse_decision
from ama.data import load_dataset
from ama.scorer import score_rocov2
from fakes import FakeModel, calls, call, decision
from test_data import example, write_dataset


def dataset(tmp_path, two=False):
    root = tmp_path / "data"
    root.mkdir()
    episodes = [example()]
    if two:
        other = example()
        other["id"] = "other"
        episodes.append(other)
    write_dataset(root, episodes, targets=[{"id": ep["id"], "turns": {
        t["id"]: {"answer": {"caption": "reference", "cuis": ["C1"]}, "required_evidence": [t["evidence"][0]["id"]]}
        for t in ep["turns"]}} for ep in episodes])
    (root / "instructions.txt").write_text("TASK: describe the observations.")
    (root / "eval.json").write_text('{"scorer":"rocov2"}')
    (root / "provenance.jsonl").write_text("PRIVATE")
    return root


def rows(path, file="decisions.jsonl"):
    return [json.loads(line) for line in (path / file).read_text().splitlines()]


def test_run_context_isolation_and_no_hidden_reads(tmp_path, monkeypatch):
    root = dataset(tmp_path, two=True)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name in {"targets.jsonl", "eval.json", "provenance.jsonl"}:
            raise AssertionError(f"inference opened {path.name}")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    model = FakeModel(decision(), decision("t2"), decision(), decision("t2"))
    path = execute(root, model, runs_root=tmp_path / "runs")
    assert len(rows(path)) == 4
    assert len(model.requests[0][0]) == 2
    assert len(model.requests[1][0]) == 4
    assert len(model.requests[2][0]) == 2
    assert "WBC" not in json.dumps(model.requests[0])
    assert "WBC" in json.dumps(model.requests[1])
    assert all(r["decision"]["answer"] is None for r in rows(path))


def test_chat_followups_next_and_unscorable_without_hidden_reads(tmp_path, monkeypatch, capsys):
    root = dataset(tmp_path)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {"targets.jsonl", "eval.json", "provenance.jsonl"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    inputs = iter(["why?", "/next", "/next", "explain again", "/quit"])
    model = FakeModel("first", "followup", "second", "explanation")
    path = execute(root, model, mode="chat", episode_ids=["ep"], runs_root=tmp_path / "runs",
                   input_fn=lambda _: next(inputs))
    assert len(model.requests) == 4
    assert "WBC" not in json.dumps(model.requests[1])
    assert "WBC" in json.dumps(model.requests[2])
    messages = [r for r in rows(path, "events.jsonl") if r["kind"] == "message" and r["message"]["role"] == "user"]
    assert [r["source"] for r in messages] == ["dataset", "human", "dataset", "human"]
    assert [r["turn_id"] for r in messages] == ["t1", "t1", "t2", "t2"]
    output = capsys.readouterr().out
    assert "没有后续材料" in output
    assert output.count("[dataset] Observation") == 2
    assert output.count("[assistant]") == 4
    with pytest.raises(ValueError, match="探索"):
        _eval_run(path)
    assert not (path / "metrics.json").exists() and not (path / "decisions.jsonl").exists()


@pytest.mark.parametrize("reply", ["bad JSON", decision("wrong")])
def test_invalid_decision_no_repair_then_next_turn(tmp_path, reply):
    model = FakeModel(reply, decision("t2", answer="answer", citations=["hpi", "lab"]))
    path = execute(dataset(tmp_path), model, runs_root=tmp_path / "runs")
    result = rows(path)
    assert result[0]["termination"] == "invalid_decision" and result[0]["reply"] == reply
    assert result[1]["termination"] == "completed"
    assert len(model.requests) == 2


def test_api_error_aborts_episode_but_continues_batch(tmp_path):
    model = FakeModel(ModelError("offline"), decision(), decision("t2"))
    root = dataset(tmp_path, two=True)
    path = execute(root, model, runs_root=tmp_path / "runs")
    assert [r["termination"] for r in rows(path)] == ["api_error", "not_executed", "completed", "completed"]
    assert len(model.requests[1][0]) == 2
    assert any(r["kind"] == "model_call" and r["error"] for r in rows(path, "events.jsonl"))
    assert _eval_run(path) == 0
    metrics = json.loads((path / "metrics.json").read_text())["scored"]
    assert metrics["aggregate"]["caption_token_recall"]["den"] == 4
    assert metrics["per_episode"]["ep"]["completion"]["den"] == 2
    assert (path / "metrics.json").exists()
    assert not (path / "report.md").exists()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), calls(call())])
def test_interrupt_and_call_limit_keep_history(tmp_path, failure):
    model = FakeModel(failure)
    path = execute(dataset(tmp_path), model, runs_root=tmp_path / "runs", max_calls=1)
    assert rows(path)[0]["termination"] in {"interrupted", "call_limit"}
    assert rows(path)[1]["termination"] == "not_executed"
    assert any(r["kind"] == "message" and r["source"] == "dataset" for r in rows(path, "events.jsonl"))
    if isinstance(failure, dict):
        assert any(r.get("message", {}).get("role") == "tool" for r in rows(path, "events.jsonl"))


def test_missing_rows_still_scored(tmp_path):
    ds = load_dataset(dataset(tmp_path), with_targets=True)
    result = score_rocov2(ds, {})
    assert result["aggregate"]["caption_token_recall"] == {"num": 0, "den": 2, "value": 0}
    assert result["per_episode"]["ep"]["completion"]["den"] == 2


def test_cli_instruction_override_selection_and_tools(tmp_path, monkeypatch):
    root = dataset(tmp_path, two=True)
    instruction = tmp_path / "task.txt"
    instruction.write_text("REPLACEMENT")
    model = FakeModel(calls(call()), decision(), decision("t2"))
    monkeypatch.setattr(Model, "from_config", lambda *a, **kw: model)
    assert main(["run", str(root), "--model", "test", "--episode", "other", "--tools", "examples/tools.py",
                 "--instruction-file", str(instruction), "--runs-root", str(tmp_path / "runs")]) == 0
    path = next((tmp_path / "runs").iterdir())
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["episode_ids"] == ["other"]
    assert manifest["instruction"] == "REPLACEMENT" and manifest["tools"]["sha256"]
    assert model.requests[0][1][0]["function"]["name"] == "add"
    assert main(["eval", str(path)]) == 0


def test_chat_cli_requires_episode_and_records_eof(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    model = FakeModel("natural answer")
    monkeypatch.setattr(Model, "from_config", lambda *a, **kw: model)
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(SystemExit):
        main(["chat", str(root), "--model", "test"])
    assert main(["chat", str(root), "--episode", "ep", "--model", "test", "--runs-root", str(tmp_path / "runs")]) == 0
    path = next((tmp_path / "runs").iterdir())
    assert main(["eval", str(path)]) == 1


def test_eval_detects_changed_dataset(tmp_path):
    root = dataset(tmp_path)
    path = execute(root, FakeModel(decision(), decision("t2")), runs_root=tmp_path / "runs")
    with (root / "episodes.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="dataset changed"):
        _eval_run(path)


def test_interrupt_does_not_start_later_episodes(tmp_path):
    model = FakeModel(KeyboardInterrupt())
    path = execute(dataset(tmp_path, two=True), model, runs_root=tmp_path / "runs")
    assert len(model.requests) == 1
    assert len(rows(path)) == 4
    assert [r["termination"] for r in rows(path)] == ["interrupted"] + ["not_executed"] * 3
    assert json.loads((path / "manifest.json").read_text())["termination"] == "interrupted"


def test_chat_interrupt_while_waiting_preserves_answer(tmp_path):
    def stop(_):
        raise KeyboardInterrupt()
    path = execute(dataset(tmp_path), FakeModel("answer"), mode="chat", episode_ids=["ep"],
                   runs_root=tmp_path / "runs", input_fn=stop)
    assert json.loads((path / "manifest.json").read_text())["termination"] == "interrupted"
    assert any(r.get("message", {}).get("content") == "answer" for r in rows(path, "events.jsonl"))


def test_tool_image_source_and_log_has_no_base64(tmp_path, capsys):
    root = dataset(tmp_path)
    scan = tmp_path / "scan.png"
    scan.write_bytes(b"fixture")
    tools = tmp_path / "tools.py"
    tools.write_text('from ama.tools import Tool, ToolResult\n'
                     f'TOOLS = [Tool("scan", "scan", {{}}, lambda: ToolResult("scan", [{str(scan)!r}]))]\n')
    model = FakeModel(calls(call("scan", "{}", "image-call")), "seen")
    path = execute(root, model, mode="chat", episode_ids=["ep"], tools_path=tools,
                   runs_root=tmp_path / "runs", input_fn=lambda _: "/quit")
    messages = [r for r in rows(path, "events.jsonl") if r["kind"] == "message"]
    assert [r["source"] for r in messages] == ["system", "dataset", "model", "tool", "tool", "model"]
    assert messages[3]["message"]["tool_call_id"] == "image-call"
    assert "base64" not in (path / "events.jsonl").read_text()
    output = capsys.readouterr().out
    assert "[tool call] scan {}" in output
    assert "[tool result] scan" in output
    assert output.count("[model] waiting for response...") == 2
    assert output.index("[tool call]") < output.index("[assistant]")


def test_abstain_and_extra_fields():
    assert parse_decision(decision(), "t1", set())["answer"] is None
    with pytest.raises(ValueError, match="abstention"):
        parse_decision(decision(citations=["seen"]), "t1", {"seen"})
    payload = json.loads(decision())
    payload["state"] = "old field"
    with pytest.raises(ValueError):
        parse_decision(json.dumps(payload), "t1", set())


def test_run_accepts_returned_tool_evidence_and_retains_raw_reply(tmp_path):
    root = dataset(tmp_path)
    tools = tmp_path / "tools.py"
    tools.write_text('from ama.tools import Tool, ToolResult\n'
                     'TOOLS = [Tool("lab", "lab", {"type":"object"}, '
                     'lambda: ToolResult("12", evidence_ids=["lab-1"]))]\n')
    raw = "Analysis first.\n" + decision(answer={"diagnosis": "pancreatitis"},
                                          citations=["hpi", "lab-1"])
    model = FakeModel(calls(call("lab", "{}")), raw, decision("t2"))
    run = execute(root, model, tools_path=tools, runs_root=tmp_path / "runs")
    first = rows(run)[0]
    assert first["reply"] == raw
    assert first["decision"]["citations"] == ["hpi", "lab-1"]
    assert first["decision_format"] == "embedded"
    assert first["termination"] == "completed"


def test_bad_visible_data_rejected_before_inference(tmp_path):
    root = dataset(tmp_path)
    (root / "episodes.jsonl").write_text(json.dumps({"id": "ep", "turns": [
        {"id": "t1", "evidence": [{"id": "x", "file": "../outside.txt"}]}]}) + "\n")
    model = FakeModel()
    with pytest.raises(ValueError, match="unsafe"):
        execute(root, model, runs_root=tmp_path / "runs")
    assert model.requests == []
