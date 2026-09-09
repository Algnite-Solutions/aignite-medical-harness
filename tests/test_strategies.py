"""Test 8 (lite): both strategies expose the same visible evidence per
(case, timepoint); full_history runs get an explicit 'disabled' error for
state tools and produce no state snapshots."""
import json

import pytest

from medical_harness.runner import run as run_harness

from conftest import DATA


def _cfg(name, state_tools, tmp_path):
    cfg = {
        "name": name,
        "data_dir": str(DATA / "inputs"),
        "model": {"type": "scripted", "script_dir": str(DATA / "scripts")},
        "strategy": {"name": "versioned_state" if state_tools else "stateless_retrieval",
                     "state_tools_enabled": state_tools, "inject_prior_state": state_tools},
        "budget": {"max_steps": 10, "max_model_calls": 10, "deadline_seconds": 60},
        "cases": ["case_gamma", "case_alpha", "case_beta"],
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _listed_evidence_by_question(run_dir):
    out = {}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "step" and ev.get("action", {}).get("type") == "list_evidence":
            obs = ev["observation"]
            ids = tuple(r["evidence_id"] for r in obs["data"]["evidence"])
            out.setdefault((ev["case_id"], ev["question_id"]), set()).add((ids, obs["data"]["count"]))
    return out


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strategies")
    versioned = run_harness(_cfg("versioned", True, tmp), tmp / "runs")
    full = run_harness(_cfg("full", False, tmp), tmp / "runs")
    return versioned, full


def test_same_visible_evidence_across_strategies(both_runs):
    versioned, full = both_runs
    a, b = _listed_evidence_by_question(versioned), _listed_evidence_by_question(full)
    assert a == b  # identical visible evidence sets for every (case, question)


def test_stateless_retrieval_state_tools_report_disabled(both_runs):
    _, full = both_runs
    steps = []
    for line in (full / "events.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "step" and ev.get("action", {}).get("type") in ("get_state", "propose_state_update"):
            steps.append(ev)
    assert steps, "scripts do use state tools; stateless_retrieval must have hit them"
    assert all("disabled" in (s["observation"].get("error") or "") for s in steps)


def test_full_history_writes_no_state_snapshots(both_runs):
    versioned, full = both_runs
    snaps_versioned = (versioned / "state_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
    snaps_full = (full / "state_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(snaps_versioned) == 5  # gamma q1,q2 + alpha q1,q3 + beta q1 proposes
    assert snaps_full == []


def test_versioned_prior_state_injected_but_full_history_not(both_runs):
    versioned, full = both_runs

    def contexts(run_dir):
        out = []
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("type") == "question_start":
                out.append(ev["model_context"][0]["content"])
        return out

    assert any("此前时间点维护的状态" in c for c in contexts(versioned))
    assert not any("此前时间点维护的状态" in c for c in contexts(full))
