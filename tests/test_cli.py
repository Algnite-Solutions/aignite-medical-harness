"""CLI run/eval isolation and built-in text importers."""
import json
from pathlib import Path

from ama.cli import main
from ama.data import load_dataset, validate_dataset
from ama.importers.episode_folder import import_episode_folders
from ama.importers.medagentbench import import_medagentbench


def test_cli_run_eval_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text('{"schema":"ama-dataset","name":"demo","splits":{"all":["ep"]}}')
    (dataset / "episodes.jsonl").write_text(json.dumps({"id": "ep", "turns": [
        {"id": "t1", "observation": "Question", "evidence": [{"id": "e", "text": "context"}]}]}) + "\n")
    (dataset / "targets.jsonl").write_text('{"id":"ep","turns":{"t1":{"answers":["yes"]}}}\n')
    (dataset / "eval.json").write_text('evaluator file must not be parsed during run')
    (dataset / "provenance.jsonl").write_text('private source file must not be parsed')
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "ep.json").write_text(json.dumps({"episode_id": "ep", "actions": [
        {"turn_id": "t1", "answer": "yes", "citations": ["e"]}]}))
    (tmp_path / "ama.json").write_text(json.dumps({"models": {"scripted": {
        "type": "scripted", "script_dir": str(script_dir)}}}))
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("Answer yes or no.")
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(dataset), "--model", "scripted", "--instruction-file", str(instruction),
                 "--runs-root", str(tmp_path / "runs")]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["experiment"]["protocol"] == "direct_decision"
    assert set(manifest["experiment"]) == {"protocol", "instruction", "instruction_sha256"}
    assert manifest["experiment"]["instruction"] == "Answer yes or no."
    assert len(manifest["experiment"]["instruction_sha256"]) == 64
    assert set(manifest["dataset_hashes"]) == {"dataset.json", "episodes.jsonl"}
    (dataset / "eval.json").write_text('{"scorer":"exact"}')
    assert main(["eval", str(run_dir)]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["scored"]["aggregate"]["answer_correct"]["num"] == 1


def test_cli_rejects_unknown_schema(tmp_path):
    (tmp_path / "dataset.json").write_text('{"schema":"unknown","name":"old"}')
    (tmp_path / "episodes.jsonl").write_text('{"id":"ep","turns":[]}\n')
    assert main(["validate", str(tmp_path)]) == 1


def test_medagentbench_import(tmp_path, monkeypatch):
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([{"id": "qa1", "eval_MRN": "p1", "instruction": "What diagnosis?",
                                   "context": "History of pain", "sol": ["appendicitis"]}]))
    out = tmp_path / "imported"
    import_medagentbench(source, out)
    assert validate_dataset(out) == []
    ds = load_dataset(out, with_targets=True)
    assert ds.info.schema_name == "ama-dataset"
    assert ds.targets["qa1"].turns["t1"]["answers"] == ["appendicitis"]
    assert "eval_MRN" not in (out / "episodes.jsonl").read_text()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "qa1.json").write_text(json.dumps({"episode_id": "qa1", "actions": [
        {"turn_id": "t1", "answer": "appendicitis", "citations": []}]}))
    (tmp_path / "ama.json").write_text(json.dumps({"models": {"scripted": {
        "type": "scripted", "script_dir": str(scripts)}}}))
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert main(["run", str(out), "--model", "scripted", "--runs-root", str(runs)]) == 0
    run_dir = next(runs.iterdir())
    assert main(["eval", str(run_dir)]) == 0
    assert json.loads((run_dir / "metrics.json").read_text())["scored"]["aggregate"]["answer_correct"]["num"] == 1


def test_episode_folder_import(tmp_path, monkeypatch):
    source = tmp_path / "source"
    folder = source / "case1"
    folder.mkdir(parents=True)
    (folder / "episode.json").write_text(json.dumps({"episode_id": "case1", "patient_id": "secret",
        "turns": [{"turn_id": "t1", "as_of": "2026-01-01T00:00:00Z",
                   "world_message": "New note", "release_evidence_ids": ["e1"]}]}))
    (folder / "evidence.jsonl").write_text(json.dumps({"evidence_id": "e1", "content": "pain",
        "modality": "note", "source_locator": "row 1", "patient_id": "secret"}) + "\n")
    (folder / "workflow.json").write_text(json.dumps({"states": ["initial", "diagnosed"],
        "initial_state": "initial", "transitions": [{"from": "initial", "to": "diagnosed",
        "when": [], "allowed_actions": ["finish"]}]}))
    (folder / "gold.json").write_text(json.dumps({"turns": {"t1": {
        "expected_workflow_state": "diagnosed", "allowed_actions": ["finish"]}}}))
    out = tmp_path / "out"
    import_episode_folders(source, out)
    assert validate_dataset(out) == []
    assert "secret" not in (out / "episodes.jsonl").read_text()
    assert "secret" in (out / "provenance.jsonl").read_text()
    assert load_dataset(out, with_targets=True).eval_config.scorer == "workflow"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "case1.json").write_text(json.dumps({"episode_id": "case1", "actions": [
        {"type": "read_evidence", "id": "e1"},
        {"turn_id": "t1", "answer": {"workflow_state": "diagnosed", "next_step": "finish"},
         "citations": ["e1"]}]}))
    (tmp_path / "ama.json").write_text(json.dumps({"models": {"scripted": {
        "type": "scripted", "script_dir": str(scripts)}}}))
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert main(["run", str(out), "--model", "scripted", "--protocol", "tool_agent",
                 "--instruction-file", str(out / "instructions.txt"), "--runs-root", str(runs)]) == 0
    run_dir = next(runs.iterdir())
    assert main(["eval", str(run_dir)]) == 0
    assert json.loads((run_dir / "metrics.json").read_text())["scored"]["aggregate"]["transition_valid"]["num"] == 1
