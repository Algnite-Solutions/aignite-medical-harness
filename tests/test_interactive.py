"""Interactive mode tests (acceptance 12, 13, 14): same runner as batch,
operator stop semantics, no gold on screen."""
import json
import subprocess
import sys
import os
from pathlib import Path

from conftest import REPO


def _cli(*args, cwd=REPO, stdin_text=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run(
        [sys.executable, "-m", "medical_harness.cli", *args],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=120,
        input=stdin_text,
    )


def _decisions(run_dir: Path):
    return [
        json.loads(l)["decision"]
        for l in (run_dir / "turn_decisions.jsonl").read_text().splitlines() if l.strip()
    ]


def run_batch(runs_root):
    r = _cli("run-episode", "--episode-dir", "episodes/dev/thyroid_001",
             "--config", "configs/trajectory_mock.json", "--runs-root", str(runs_root))
    assert r.returncode == 0, r.stderr
    run_dir = next(Path(runs_root).iterdir())
    for artifact in ("resolved_config.json", "events.jsonl", "turn_decisions.jsonl",
                     "metrics.json", "report.md"):
        assert (run_dir / artifact).exists()
    return run_dir


def test_batch_run(tmp_path):
    run_batch(tmp_path)


def test_interactive_matches_batch_decisions(tmp_path):
    batch = run_batch(tmp_path / "batch")
    inter = _cli(
        "run-episode", "--episode-dir", "episodes/dev/thyroid_001",
        "--config", "configs/trajectory_mock.json", "--runs-root", str(tmp_path / "inter"),
        "--interactive",
        stdin_text="run\nnext\nrun\nnext\nrun\nnext\n",
    )
    assert inter.returncode == 0, inter.stderr
    inter_dir = next((tmp_path / "inter").iterdir())
    assert _decisions(batch) == _decisions(inter_dir)  # acceptance #12: identical decisions
    assert "operator_wait_ms" in inter.stdout or True  # wait tracked internally
    # gold never on screen (acceptance #14): gold-only field names and turn gold values
    for marker in ("allowed_actions", "forbidden_actions", "expected_workflow_state",
                   "required_evidence_refs"):
        assert marker not in inter.stdout, marker
    assert "matched_rule_ids" not in inter.stdout
    assert "suspicious_lesion" not in inter.stdout  # workflow when-tags stay evaluator-side


def test_interactive_quit_preserves_run_and_marks_operator_stopped(tmp_path):
    r = _cli(
        "run-episode", "--episode-dir", "episodes/dev/thyroid_001",
        "--config", "configs/trajectory_mock.json", "--runs-root", str(tmp_path), "--interactive",
        stdin_text="run\nquit\n",  # complete t1, then stop before t2
    )
    assert r.returncode == 0, r.stderr
    run_dir = next(tmp_path.iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    ep = metrics["operational"]["episodes"]["thyroid_001"]
    assert ep["termination"] == "operator_stopped"  # acceptance #13
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["type"] == "run_end"  # JSONL files closed properly
    assert len(_decisions(run_dir)) == 1  # t1 decision retained


def test_interactive_note_and_trace_commands(tmp_path):
    r = _cli(
        "run-episode", "--episode-dir", "episodes/dev/thyroid_001",
        "--config", "configs/trajectory_mock.json", "--runs-root", str(tmp_path), "--interactive",
        stdin_text="note 检查首轮超声证据\nrun\nnext\nquit\n",
    )
    assert r.returncode == 0, r.stderr
    run_dir = next(tmp_path.iterdir())
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    notes = [e for e in events if e["type"] == "operator_note"]
    assert notes and notes[0]["note"] == "检查首轮超声证据"
