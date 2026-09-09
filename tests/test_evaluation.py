"""Test 7 + 9: correct vs deliberately-wrong scripted runs score differently
and as expected; empty denominators / failed tasks / missing usage are honest."""
import json

import pytest

from medical_harness.evaluation import evaluate, score_run
from medical_harness.runner import run as run_harness

from conftest import DATA


def write_cfg(tmp_path, name, script_dir, cases):
    cfg = {
        "name": name,
        "data_dir": str(DATA / "inputs"),
        "model": {"type": "scripted", "script_dir": str(script_dir)},
        "strategy": {"name": "versioned_state", "state_tools_enabled": True, "inject_prior_state": True},
        "budget": {"max_steps": 10, "max_model_calls": 10, "deadline_seconds": 60},
        "cases": cases,
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def correct_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("correct")
    return run_harness(write_cfg(tmp, "correct", DATA / "scripts", ["case_gamma", "case_alpha", "case_beta"]), tmp / "runs")


@pytest.fixture(scope="module")
def wrong_run(tmp_path_factory):
    # deliberately wrong alpha: q2 answers with the delayed 1.8 value (should abstain),
    # q3 answers with the superseded 1.8 citing e-lab-1 (should be 2.1 / e-lab-2)
    tmp = tmp_path_factory.mktemp("wrong")
    script_dir = tmp / "scripts"
    script_dir.mkdir()
    for name in ("case_beta.json", "case_gamma.json"):
        (script_dir / name).write_text((DATA / "scripts" / name).read_text(encoding="utf-8"), encoding="utf-8")
    wrong = {
        "case_id": "case_alpha",
        "by_question": {
            "q1": json.loads((DATA / "scripts" / "case_alpha.json").read_text(encoding="utf-8"))["by_question"]["q1"],
            "q2": [{"type": "submit_answer", "answer": {
                "question_id": "q2",
                "claims": [{"key": "creatinine", "value": 1.8, "evidence_refs": ["e-lab-1"]}],
                "evidence_refs": ["e-lab-1"], "abstain": False,
                "reason": "故意错误：使用了尚未入库的延迟检验值。"}}],
            "q3": [{"type": "submit_answer", "answer": {
                "question_id": "q3",
                "claims": [{"key": "creatinine", "value": 1.8, "evidence_refs": ["e-lab-1"]}],
                "evidence_refs": ["e-lab-1"], "abstain": False,
                "reason": "故意错误：引用了被更正替代的旧值。"}}],
        },
    }
    (script_dir / "case_alpha.json").write_text(json.dumps(wrong, ensure_ascii=False), encoding="utf-8")
    return run_harness(write_cfg(tmp, "wrong", script_dir, ["case_gamma", "case_alpha", "case_beta"]), tmp / "runs")


def test_correct_scripted_run_scores_perfectly(correct_run):
    result = score_run(correct_run)
    s = result["scored"]
    assert s["n_questions"] == 6
    assert s["field_correct"] == {"num": 5, "den": 5, "value": 1.0}
    assert s["correct_abstain"] == {"num": 1, "den": 1, "value": 1.0}
    assert s["citation_coverage"] == {"num": 5, "den": 5, "value": 1.0}
    assert s["citation_correctness"] == {"num": 5, "den": 5, "value": 1.0}
    assert s["forbidden_citations"]["num"] == 0 and s["forbidden_citations"]["den"] == 5
    assert s["stale_refs"]["num"] == 0 and s["stale_refs"]["den"] == 2
    assert result["violations"] == {"future": 0, "other_patient": 0}
    assert result["failures"] == []
    assert result["scorer_policy_version"] == "0.2"
    assert "case_alpha.json" in result["gold_sha256"]  # gold digest recorded


def test_deliberately_wrong_run_scores_differently(wrong_run):
    result = score_run(wrong_run)
    s = result["scored"]
    # alpha q2 cited the (invisible) delayed lab -> submission rejected -> model_error;
    # alpha q3 cited e-lab-1 WITHOUT reading it -> read-before-cite rejection -> model_error
    assert result["operational"]["terminations"] == {"completed": 4, "model_error": 2}
    assert len(result["failures"]) == 2
    assert s["field_correct"]["num"] == 4 and s["field_correct"]["den"] == 5  # alpha q3 never completed
    assert s["correct_abstain"]["num"] == 0 and s["correct_abstain"]["den"] == 1
    # completed answers (gamma q1/q2, alpha q1, beta q1) all cite allowed refs
    assert s["citation_correctness"] == {"num": 4, "den": 4, "value": 1.0}
    # stale denominator now only gamma q2 (alpha q3 never got submitted)
    assert s["stale_refs"]["num"] == 0 and s["stale_refs"]["den"] == 1


def test_wrong_run_scores_differ_from_correct(correct_run, wrong_run):
    a = score_run(correct_run)["scored"]
    b = score_run(wrong_run)["scored"]
    assert a["field_correct"]["num"] != b["field_correct"]["num"]
    assert a["correct_abstain"]["num"] != b["correct_abstain"]["num"]


def test_empty_denominator_reported_as_na(tmp_path):
    # only gamma+beta: no should-abstain questions -> correct_abstain is N/A
    run_dir = run_harness(write_cfg(tmp_path, "noabstain", DATA / "scripts", ["case_gamma", "case_beta"]), tmp_path / "runs")
    result = evaluate(run_dir)
    s = result["scored"]
    assert s["n_should_abstain"] == 0
    assert s["correct_abstain"]["den"] == 0 and s["correct_abstain"]["value"] is None
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "N/A" in report


def test_failed_tasks_stay_in_denominator(tmp_path):
    # empty script dir -> every question terminates model_error, all counted
    empty = tmp_path / "empty_scripts"
    empty.mkdir()
    run_dir = run_harness(write_cfg(tmp_path, "broken", empty, ["case_gamma"]), tmp_path / "runs")
    result = evaluate(run_dir)
    s = result["scored"]
    assert s["field_correct"] == {"num": 0, "den": 2, "value": 0.0}  # failures kept in denominator
    assert result["operational"]["terminations"] == {"model_error": 2}
    assert len(result["failures"]) == 2
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "model_error" in report


def test_missing_usage_reported_as_unknown(correct_run):
    rows = [json.loads(l) for l in (correct_run / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(r["usage"] == "unknown" for r in rows)  # scripted backend: honest unknown, no fabricated tokens
    metrics = json.loads((correct_run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["operational"]["usage"] == "unknown"
