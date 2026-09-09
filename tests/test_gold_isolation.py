"""Test 6: gold never enters the model context, tool list, or run artifacts
produced by `run` (gold is evaluator-only)."""
import json

from medical_harness.runner import run as run_harness

from conftest import DATA

import pytest


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ctxcheck")
    cfg = {
        "name": "mock",
        "data_dir": str(DATA / "inputs"),
        "model": {"type": "scripted", "script_dir": str(DATA / "scripts")},
        "strategy": {"name": "versioned_state", "state_tools_enabled": True, "inject_prior_state": True},
        "budget": {"max_steps": 10, "max_model_calls": 10, "deadline_seconds": 60},
        "cases": ["case_gamma", "case_alpha", "case_beta"],
    }
    cfg_path = tmp / "mock.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_dir = run_harness(cfg_path, tmp / "runs")
    return run_dir


def _events(run_dir):
    return [json.loads(l) for l in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_gold_markers_absent_from_model_context(mock_run):
    contexts = [e for e in _events(mock_run) if e["type"] == "question_start"]
    assert contexts, "no question_start events logged"
    blob = json.dumps([e.get("model_context") for e in contexts], ensure_ascii=False)
    for marker in ("allowed_refs", "stale_refs", "forbidden_refs", '"expected"', "gold"):
        assert marker not in blob, f"gold marker '{marker}' leaked into model context"


def test_gold_entries_absent_verbatim(mock_run, gold):
    contexts = [e for e in _events(mock_run) if e["type"] == "question_start"]
    blob = json.dumps([e.get("model_context") for e in contexts], ensure_ascii=False)
    for g in gold.values():
        for a in g.answers:
            payload = json.dumps({"question_id": a.question_id, "expected": a.expected.model_dump(mode="json")}, ensure_ascii=False)
            assert payload not in blob


def test_run_artifacts_never_read_gold(mock_run):
    resolved = json.loads((mock_run / "resolved_config.json").read_text(encoding="utf-8"))
    blob = json.dumps(resolved)
    assert "gold" not in blob
    assert not any("gold" in k for k in resolved["input_hashes"])  # only inputs hashed
    # operational metrics.json written by run has no scored section
    metrics = json.loads((mock_run / "metrics.json").read_text(encoding="utf-8"))
    assert "scored" not in metrics
