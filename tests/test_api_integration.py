"""Opt-in real-API integration test. NEVER runs by default (spec: default
pytest must stay offline and free). Enable with:

    RUN_API_TESTS=1 GLM_API_KEY=... pytest -q tests/test_api_integration.py -m api

It runs one calibration question end-to-end against the configured provider
and asserts the plumbing (tool protocol, validation, logging), not model quality.
"""
import json
import os
from pathlib import Path

import pytest

from conftest import DATA, REPO

pytestmark = [
    pytest.mark.api,
    pytest.mark.skipif(
        os.environ.get("RUN_API_TESTS") != "1" or not os.environ.get(os.environ.get("GLM_API_KEY_ENV", "GLM_API_KEY")),
        reason="opt-in real-API test: set RUN_API_TESTS=1 and GLM_API_KEY",
    ),
]


def _glm_cfg(tmp_path: Path, cases: list[str]) -> Path:
    base = json.loads((REPO / "configs" / "glm.json").read_text(encoding="utf-8"))
    base["cases"] = cases
    base["data_dir"] = str(DATA / "inputs")
    base["model"]["base_url"] = base["model"]["base_url"]
    p = tmp_path / "glm.json"
    p.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    return p


def test_glm_end_to_end_single_question(tmp_path):
    from medical_harness.evaluation import evaluate
    from medical_harness.runner import run as run_harness

    run_dir = run_harness(_glm_cfg(tmp_path, ["case_beta"]), tmp_path / "runs")
    rows = [json.loads(l) for l in (run_dir / "answers.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["termination"] == "completed", row["termination_detail"]
    assert row["answer"] is not None
    # real backend should report token usage (scripted reports "unknown")
    assert isinstance(row["usage"], dict) and row["usage"].get("total_tokens", 0) > 0
    # key never leaks into artifacts
    key = os.environ["GLM_API_KEY"]
    for artifact in ("resolved_config.json", "events.jsonl", "answers.jsonl"):
        assert key not in (run_dir / artifact).read_text(encoding="utf-8")
    result = evaluate(run_dir)
    print("scored:", json.dumps(result["scored"], ensure_ascii=False))
