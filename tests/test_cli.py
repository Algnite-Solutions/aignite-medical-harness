"""CLI run/eval isolation and Chinese experiment reports."""
import json

from ama.cli import main


def test_cli_run_eval_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text('{"schema":"ama-dataset","name":"demo","splits":{"all":["ep"]}}')
    (dataset / "episodes.jsonl").write_text(json.dumps({"id": "ep", "turns": [
        {"id": "t1", "observation": "Question", "evidence": [{"id": "e", "text": "context"}]}]}) + "\n")
    (dataset / "targets.jsonl").write_text('{"id":"ep","turns":{"t1":{"answer":{"caption":"chest image","cuis":["C1"]}}}}\n')
    (dataset / "eval.json").write_text('evaluator file must not be parsed during run')
    (dataset / "provenance.jsonl").write_text('private source file must not be parsed')
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "ep.json").write_text(json.dumps({"episode_id": "ep", "actions": [
        {"turn_id": "t1", "answer": {"caption": "chest image", "cuis": ["C1"]}, "citations": ["e"]}]}))
    (tmp_path / "ama.json").write_text(json.dumps({"models": {"scripted": {
        "type": "scripted", "script_dir": str(script_dir)}}}))
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("Describe the image.")
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(dataset), "--model", "scripted", "--instruction-file", str(instruction),
                 "--runs-root", str(tmp_path / "runs")]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["experiment"]["protocol"] == "direct_decision"
    assert set(manifest["experiment"]) == {"protocol", "instruction", "instruction_sha256"}
    assert manifest["experiment"]["instruction"] == "Describe the image."
    assert len(manifest["experiment"]["instruction_sha256"]) == 64
    assert set(manifest["dataset_hashes"]) == {"dataset.json", "episodes.jsonl"}
    (dataset / "eval.json").write_text('{"scorer":"rocov2"}')
    assert main(["eval", str(run_dir)]) == 0
    report = (run_dir / "report.md").read_text()
    assert "模型原文" in report and "参考原文" in report and "描述词 F1" in report
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["scored"]["aggregate"]["caption_token_f1"]["value"] == 1


def test_cli_rejects_unknown_schema(tmp_path):
    (tmp_path / "dataset.json").write_text('{"schema":"unknown","name":"old"}')
    (tmp_path / "episodes.jsonl").write_text('{"id":"ep","turns":[]}\n')
    assert main(["validate", str(tmp_path)]) == 1
