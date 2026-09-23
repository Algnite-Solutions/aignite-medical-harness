"""Answer scoring and source-task-specific endpoints."""
import json

from ama.data import load_dataset
from ama.scorer import score_exact, score_rocov2, score_workflow


def fixture(tmp_path, scorer, episodes, targets, rules=None):
    (tmp_path / "dataset.json").write_text(json.dumps({"schema": "ama-dataset", "name": "x",
                                                      "splits": {"all": [e["id"] for e in episodes]}}))
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n")
    (tmp_path / "targets.jsonl").write_text("\n".join(json.dumps(t) for t in targets) + "\n")
    (tmp_path / "eval.json").write_text(json.dumps({"scorer": scorer, "rules": rules or {}}))
    return load_dataset(tmp_path, with_targets=True)


def row(turn, answer, citations=()):
    return {"episode_id": "ep", "turn_id": turn, "decision": {
        "turn_id": turn, "answer": answer, "citations": list(citations)},
        "termination": "completed", "model_calls": 1, "usage": {}, "duration_ms": 1}


def test_exact_answer_and_abstention(tmp_path):
    ds = fixture(tmp_path, "exact", [{"id": "ep", "turns": [
        {"id": "t1", "evidence": [{"id": "e", "text": "x"}]}, {"id": "t2", "evidence": []}]}],
        [{"id": "ep", "turns": {"t1": {"answers": ["yes"], "required_evidence": ["e"]},
                                  "t2": {"expect_abstain": True}}}])
    scored = score_exact(ds, {"ep": [row("t1", "yes", ["e"]), row("t2", None)]})
    assert scored["aggregate"]["answer_correct"] == {"num": 2, "den": 2, "value": 1.0}


def test_exact_accepts_scalar_and_structured_targets(tmp_path):
    ds = fixture(tmp_path, "exact", [{"id": "ep", "turns": [
        {"id": "t1", "evidence": []}, {"id": "t2", "evidence": []}]}],
        [{"id": "ep", "turns": {"t1": {"answer": "yes"},
                                  "t2": {"answer": {"diagnosis": "appendicitis"}}}}])
    scored = score_exact(ds, {"ep": [row("t1", "yes"),
                                      row("t2", {"diagnosis": "appendicitis", "confidence": 0.8})]})
    assert scored["aggregate"]["answer_correct"]["num"] == 2


def test_workflow_uses_answer_next_step_and_hidden_rules(tmp_path):
    episode = {"id": "ep", "turns": [{"id": "t1", "evidence": [{"id": "e", "text": "suspicious"}]}]}
    target = {"id": "ep", "turns": {"t1": {"answer": {"workflow_state": "pathology"},
                                           "allowed_actions": ["order_biopsy"],
                                           "required_evidence": ["e"]}}}
    rules = {"initial_state": "initial", "evidence_tags": {"e": ["suspicious"]},
             "transitions": [{"from": "initial", "to": "pathology", "when": ["suspicious"],
                              "allowed_actions": ["order_biopsy"]}]}
    ds = fixture(tmp_path, "workflow", [episode], [target], rules)
    scored = score_workflow(ds, {"ep": [row("t1", {"workflow_state": "pathology",
                                                "next_step": "order_biopsy"}, ["e"])]})
    assert scored["aggregate"]["transition_valid"]["num"] == 1
    assert scored["aggregate"]["action_correct"]["num"] == 1


def test_rocov2_answer_fields(tmp_path):
    ds = fixture(tmp_path, "rocov2", [{"id": "ep", "turns": [
        {"id": "t1", "evidence": [{"id": "image", "text": "description"}]}]}],
        [{"id": "ep", "turns": {"t1": {"answer": {"caption": "left lung opacity", "cuis": ["C1"]}}}}])
    scored = score_rocov2(ds, {"ep": [row("t1", {"caption": "left lung opacity", "cuis": ["C1"]})]})
    assert scored["aggregate"]["cui_f1"]["value"] == 1.0
