"""Scorer tests: exact_v0 / workflow_v0 / unscored; correct vs wrong differ."""
import json
from pathlib import Path

from ama.data import load_dataset
from ama.scorer import REGISTRY


def make_ds(tmp_path, episodes, targets, policy=None, scorer="workflow_v0"):
    ds = {"schema": "ama-dataset-v0", "name": "s", "splits": {"all": [e["episode_id"] for e in episodes]},
          "scorer": scorer}
    (tmp_path / "dataset.json").write_text(json.dumps(ds), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes) + "\n", encoding="utf-8")
    (tmp_path / "targets.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in targets) + "\n", encoding="utf-8")
    if policy:
        (tmp_path / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return load_dataset(tmp_path, with_targets=True)


EP = {"episode_id": "ep", "subject_id": "s", "metadata": {}, "turns": [
    {"turn_id": "t1", "time": "2026-01-01T00:00:00+00:00", "message": "m1",
     "evidence": [{"evidence_id": "us", "kind": "us", "text": "TI-RADS 4c", "artifact": None,
                   "source": "s", "metadata": {}}]},
    {"turn_id": "t2", "time": "2026-01-03T00:00:00+00:00", "message": "m2",
     "evidence": [{"evidence_id": "ct", "kind": "ct", "text": "恶性可能", "artifact": None,
                   "source": "s", "metadata": {}}]},
]}

TARGETS = [{"episode_id": "ep", "turns": {
    "t1": {"state": {"workflow_state": "initial_imaging"},
           "allowed_actions": ["order_biopsy", "order_contrast_imaging"],
           "forbidden_actions": ["start_treatment"],
           "required_evidence": ["us"]},
    "t2": {"state": {"workflow_state": "advanced_imaging",
                     "hypotheses": {"甲状腺恶性肿瘤": "supported"}},
           "allowed_actions": ["order_fna"],
           "forbidden_actions": ["start_treatment"],
           "required_evidence": ["ct"]},
}}]

POLICY = {"public": {"guidance": "流程：initial_imaging → advanced_imaging → pathology。"},
          "hidden": {"initial_state": "initial_imaging",
                     "evidence_tags": {"us": ["suspicious_lesion"],
                                       "ct": ["imaging_confirmed_suspicious"]},
                     "transitions": [
              {"from": "initial_imaging", "to": "advanced_imaging", "when": ["suspicious_lesion"],
               "allowed_actions": ["order_contrast_imaging", "order_biopsy"]},
              {"from": "advanced_imaging", "to": "pathology", "when": ["imaging_confirmed_suspicious"],
               "allowed_actions": ["order_fna", "order_biopsy"]},
          ]}}


def row(turn, decision):
    return {"episode_id": "ep", "turn_id": turn, "termination": "completed",
            "decision": decision, "read_ids": [], "model_calls": 1, "steps": 1,
            "usage": "unknown", "duration_ms": 1}


def dec(turn, state=None, action=None, cites=(), abstain=False):
    return {"turn_id": turn, "state": state or {}, "action": (
        {"name": action, "arguments": {}} if action else None),
        "citations": list(cites), "abstain": abstain, "note": ""}


def test_workflow_v0_correct_trajectory(tmp_path):
    ds = make_ds(tmp_path, [EP], TARGETS, POLICY)
    rows = [
        row("t1", dec("t1", {"workflow_state": "initial_imaging",
                             "hypotheses": [{"label": "甲状腺恶性肿瘤", "status": "possible"}]},
                      "order_contrast_imaging", ["us"])),
        row("t2", dec("t2", {"workflow_state": "advanced_imaging",
                             "hypotheses": [{"label": "甲状腺恶性肿瘤", "status": "supported"}]},
                      "order_fna", ["ct"])),
    ]
    out = REGISTRY["workflow_v0"](ds, {"ep": rows})
    agg = out["aggregate"]
    assert agg["state_field"] == {"num": 3, "den": 3, "value": 1.0}
    assert agg["action_correct"]["num"] == 2 and agg["refs_covered"]["num"] == 2
    assert out["guideline_violation_count"] == 0
    assert out["per_episode"]["ep"]["final_cumulative_success"] == {"num": 2, "den": 2, "value": 1.0}


def test_workflow_v0_wrong_trajectory_scores_worse(tmp_path):
    ds = make_ds(tmp_path, [EP], TARGETS, POLICY)
    rows = [
        row("t1", dec("t1", {"workflow_state": "initial_imaging"}, "order_contrast_imaging", ["us"])),
        # illegal jump + forbidden action + wrong state + missing required ref
        row("t2", dec("t2", {"workflow_state": "pathology"}, "start_treatment", [])),
    ]
    out = REGISTRY["workflow_v0"](ds, {"ep": rows})
    t2 = [t for t in out["per_episode"]["ep"]["turns"] if t["turn_id"] == "t2"][0]
    assert not t2["transition_valid"]
    assert "forbidden_action:start_treatment" in t2["guideline_violations"]
    assert t2["state_field_accuracy"]["num"] == 0
    assert out["per_episode"]["ep"]["final_cumulative_success"]["num"] < 2


def test_workflow_v0_missing_preconditions(tmp_path):
    policy = json.loads(json.dumps(POLICY))
    policy["hidden"]["transitions"][0]["when"] = ["tag_not_present"]
    ds = make_ds(tmp_path, [EP], TARGETS, policy)
    rows = [row("t1", dec("t1", {"workflow_state": "advanced_imaging"}, "order_biopsy", [])),
            row("t2", dec("t2", {"workflow_state": "advanced_imaging"}, "order_fna", ["ct"]))]
    out = REGISTRY["workflow_v0"](ds, {"ep": rows})
    t1 = out["per_episode"]["ep"]["turns"][0]
    assert not t1["transition_valid"]
    assert any("missing_preconditions" in v for v in t1["guideline_violations"])


def test_exact_v0_answers_abstain_refs(tmp_path):
    qa = {"episode_id": "qa", "subject_id": "p", "metadata": {},
          "turns": [{"turn_id": "t1", "time": None, "message": "肌酐?",
                     "evidence": [{"evidence_id": "lab", "kind": "lab", "text": "1.2",
                                   "artifact": None, "source": "s", "metadata": {}}]}]}
    targets = [{"episode_id": "qa", "turns": {"t1": {"answers": [1.2, "1.2"],
                                                    "required_evidence": ["lab"]}}}]
    ds = make_ds(tmp_path, [qa], targets, scorer="exact_v0")
    ok = REGISTRY["exact_v0"](ds, {"qa": [row("t1", dec("t1", {"answer": "1.2 mg/dL"}, cites=["lab"]))]})
    assert ok["aggregate"]["answer_correct"]["num"] == 1  # numeric-string tolerance
    assert ok["aggregate"]["refs_covered"]["num"] == 1
    wrong = REGISTRY["exact_v0"](ds, {"qa": [row("t1", dec("t1", {"answer": 9.9}, cites=["lab"]))]})
    assert wrong["aggregate"]["answer_correct"]["num"] == 0
    # unsubmitted turn stays in denominator
    failed = REGISTRY["exact_v0"](ds, {"qa": [dict(row("t1", dec("t1", {"answer": 1.2})),
                                                decision=None, termination="model_error")]})
    assert failed["aggregate"]["answer_correct"] == {"num": 0, "den": 1, "value": 0.0}


def test_exact_v0_expect_abstain(tmp_path):
    qa = {"episode_id": "qa2", "subject_id": "p", "metadata": {},
          "turns": [{"turn_id": "t1", "time": None, "message": "血钾?", "evidence": []}]}
    targets = [{"episode_id": "qa2", "turns": {"t1": {"expect_abstain": True}}}]
    ds = make_ds(tmp_path, [qa], targets, scorer="exact_v0")
    ok = REGISTRY["exact_v0"](ds, {"qa2": [row("t1", dec("t1", abstain=True))]})
    bad = REGISTRY["exact_v0"](ds, {"qa2": [row("t1", dec("t1", {"answer": 4.0}))]})
    assert ok["aggregate"]["answer_correct"]["num"] == 1
    assert bad["aggregate"]["answer_correct"]["num"] == 0


def test_unscored_reports_operations_only(tmp_path):
    ds = make_ds(tmp_path, [EP], [], scorer=None)
    out = REGISTRY["unscored"](ds, {"ep": [row("t1", dec("t1")), dict(row("t2", dec("t2")),
                                                                decision=None, termination="budget_exceeded")]})
    per = out["per_episode"]["ep"]
    assert per["completion"] == {"num": 1, "den": 2, "value": 0.5}
    assert "budget_exceeded" in per["terminations"]
