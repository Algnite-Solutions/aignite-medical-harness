"""CLI acceptance: exact spec commands (offline), plus evaluate and profile."""
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import REPO


def cli(*args, cwd=REPO):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run(
        [sys.executable, "-m", "medical_harness.cli", *args],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=120,
    )


def test_run_command_produces_complete_artifacts(tmp_path):
    r = cli("run", "--config", "configs/mock.json", "--runs-root", str(tmp_path))
    assert r.returncode == 0, r.stderr
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for artifact in ("resolved_config.json", "events.jsonl", "state_snapshots.jsonl",
                     "answers.jsonl", "metrics.json", "report.md"):
        assert (run_dir / artifact).exists(), f"missing {artifact}"
    rows = [json.loads(l) for l in (run_dir / "answers.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 6
    assert all(r_["termination"] == "completed" for r_ in rows)


def test_evaluate_command_scores_run(tmp_path):
    r = cli("run", "--config", "configs/mock.json", "--runs-root", str(tmp_path))
    assert r.returncode == 0, r.stderr
    run_dir = next(tmp_path.iterdir())
    e = cli("evaluate", "--run-dir", str(run_dir))
    assert e.returncode == 0, e.stderr
    assert "field_correct=5/5" in e.stdout
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["scored"]["field_correct"]["value"] == 1.0
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "scripted" in report and "不代表真实模型效果" in report


def test_profile_command_prints_json():
    r = cli("profile", "--data-dir", "data/sim/inputs")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["n_cases"] == 3


def test_verbose_run_prints_step_trace(tmp_path):
    r = cli("run", "--config", "configs/mock.json", "--runs-root", str(tmp_path), "--verbose")
    assert r.returncode == 0, r.stderr
    assert "step 1 list_evidence -> OK" in r.stdout
    assert "TERMINATION completed" in r.stdout
