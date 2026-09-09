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


def test_validate_episode_ok_and_broken(tmp_path):
    ok = cli("validate-episode", "--episode-dir", "episodes/dev/thyroid_001")
    assert ok.returncode == 0 and ok.stdout.startswith("OK")

    import json as _json
    d = tmp_path / "broken_ep"
    d.mkdir()
    ep = json.loads((REPO / "episodes/dev/thyroid_001/episode.json").read_text(encoding="utf-8"))
    ep["turns"][0]["release_evidence_ids"] = ["ev-missing"]
    (d / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    for name in ("evidence.jsonl", "workflow.json", "gold.json"):
        (d / name).write_text((REPO / "episodes/dev/thyroid_001" / name).read_text(encoding="utf-8"))
    bad = cli("validate-episode", "--episode-dir", str(d))
    assert bad.returncode == 1 and "ev-missing" in bad.stdout


def test_run_trajectory_and_evaluate(tmp_path):
    r = cli("run-trajectory", "--config", "configs/trajectory_mock.json", "--runs-root", str(tmp_path))
    assert r.returncode == 0, r.stderr
    run_dir = next(tmp_path.iterdir())
    e = cli("evaluate-trajectory", "--run-dir", str(run_dir))
    assert e.returncode == 0, e.stderr
    assert "next_action=3/3" in e.stdout
    assert "cumulative=3/3" in e.stdout
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "teacher_forced_replay" in report and "逐轮评分" in report


def test_profile_medagentbench_offline(tmp_path):
    # dead port keeps the test hermetic even when the real FHIR server is up
    r = cli("profile-medagentbench", "--patient-limit", "3",
            "--fhir-base", "http://127.0.0.1:9/fhir", "--out", str(tmp_path / "audit.json"))
    assert r.returncode == 0, r.stderr
    prof = json.loads((tmp_path / "audit.json").read_text())
    assert prof["source"]["n_tasks"] == 300
    assert prof["fhir"]["reachable"] is False
    assert any(p["trajectory_candidate"]["included"] is False for p in prof["patients"])
