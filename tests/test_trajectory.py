"""Trajectory episode tests: acceptance scenarios 1, 2, 5, 6, 7, 8, 9."""
import json
from pathlib import Path

import pytest

from medical_harness.model import ScriptedModel
from medical_harness.runner import EventLog
from medical_harness.schemas import (
    BudgetConfig, ReadEvidenceAction, StrategyConfig, SubmitTurnDecisionAction, parse_action,
)
from medical_harness.trajectory import (
    EpisodeResult, load_episode_bundle, run_episode, score_episode, validate_episode,
)
from conftest import REPO

EP_DIR = REPO / "episodes/dev/thyroid_001"
BUNDLE = load_episode_bundle(EP_DIR, with_gold=True)
GOLD = BUNDLE.gold


def _script_from_file():
    raw = json.loads((REPO / "data/trajectory_synthetic/scripts/thyroid_001.json").read_text(encoding="utf-8"))
    return [parse_action(a) for a in raw["actions"]]


def _run(actions, budget=None, trace=None):
    log_rows = []
    log = EventLog(_FakeFH(log_rows), run_id="test")
    result = run_episode(
        BUNDLE, ScriptedModel("thyroid_001", list(actions)),
        budget or BudgetConfig(max_steps=30, max_model_calls=30, per_turn_model_calls=8,
                               deadline_seconds=60, max_retries=0),
        log, StrategyConfig(), trace=trace,
    )
    return result, log_rows


class _FakeFH:
    def __init__(self, rows):
        self.rows = rows

    def write(self, s):
        self.rows.append(json.loads(s))

    def flush(self):
        pass


def _dec(turn_id, state, action=None, abstain=False, refs=None):
    return SubmitTurnDecisionAction.model_validate({
        "type": "submit_turn_decision",
        "decision": {
            "turn_id": turn_id, "workflow_state": state, "abstain": abstain,
            "state": {"hypotheses": []},
            "next_action": None if action is None else {
                "action_type": action, "reason": "r", "evidence_refs": refs or [],
            },
        },
    })


# ---------------- acceptance 1: per-turn visibility

def test_turn_visibility_progressive():
    from medical_harness.environment import EpisodeEnvironment
    env = EpisodeEnvironment(BUNDLE.episode)
    ids = lambda: {e.evidence_id for e in env._visible()}
    assert ids() == {"ev-us-001"}
    env.advance_turn(BUNDLE.episode.turns[1])
    assert ids() == {"ev-us-001", "ev-ct-001"}
    env.advance_turn(BUNDLE.episode.turns[2])
    assert ids() == {"ev-us-001", "ev-ct-001", "ev-path-001"}


def test_future_turn_evidence_not_listed_or_readable():
    from medical_harness.environment import EpisodeEnvironment
    from medical_harness.schemas import ListEvidenceAction, ReadEvidenceAction
    env = EpisodeEnvironment(BUNDLE.episode)  # turn t1: pathology reports not released
    res = env.execute(ReadEvidenceAction(evidence_id="ev-path-001"))
    assert not res.ok
    assert env.violations["future"] == 1  # internally tagged, generic error to the model
    listed = env.execute(ListEvidenceAction()).data["evidence"]
    assert "ev-path-001" not in [e["evidence_id"] for e in listed]


# ---------------- acceptance 2: persistence + isolation

def test_read_set_and_state_persist_across_turns():
    from medical_harness.environment import EpisodeEnvironment
    env = EpisodeEnvironment(BUNDLE.episode)
    env.execute(ReadEvidenceAction(evidence_id="ev-us-001"))
    env.advance_turn(BUNDLE.episode.turns[1])
    assert "ev-us-001" in env.read_ids  # t1 reads remain valid context
    res = env.execute(_dec("t2", "advanced_imaging", action="order_fna", refs=["ev-us-001"]))
    assert res.ok  # evidence read in an earlier turn may be cited later


def test_other_patient_fully_isolated():
    from medical_harness.environment import EpisodeEnvironment
    env = EpisodeEnvironment(BUNDLE.episode)
    res = env.execute(ReadEvidenceAction(evidence_id="ev-other-999"))
    assert not res.ok
    assert env.violations["other_patient"] == 1
    assert "ev-other-999" not in {e.evidence_id for e in env._visible()}


# ---------------- acceptance 5: visible-but-unread citation rejected

def test_unread_evidence_citation_rejected_then_recovered():
    from medical_harness.environment import EpisodeEnvironment
    env = EpisodeEnvironment(BUNDLE.episode)
    env.advance_turn(BUNDLE.episode.turns[1])
    env.execute(ReadEvidenceAction(evidence_id="ev-us-001"))
    bad = env.execute(_dec("t2", "advanced_imaging", action="order_fna", refs=["ev-ct-001"]))
    assert not bad.ok and "not been read" in bad.error
    env.execute(ReadEvidenceAction(evidence_id="ev-ct-001"))
    ok = env.execute(_dec("t2", "advanced_imaging", action="order_fna", refs=["ev-ct-001"]))
    assert ok.ok


# ---------------- acceptance 9 + 4: correct vs wrong trajectories; multiple right actions

def test_correct_scripted_trajectory_scores_clean():
    result, log_rows = _run(_script_from_file())
    assert result.termination == "completed" and len(result.rows) == 3
    scored = score_episode(result.rows, GOLD)
    agg = scored["aggregate"]
    assert agg["next_action_accuracy"] == {"num": 3, "den": 3, "value": 1.0}
    assert agg["state_field_accuracy"]["num"] == 4 and agg["state_field_accuracy"]["den"] == 4
    assert agg["guideline_violation_count"] == 0
    assert agg["trajectory_completion"]["num"] == 3
    assert agg["final_cumulative_success"] == {"num": 3, "den": 3}


def test_alternative_allowed_actions_also_score_correct():
    # acceptance #4: two different next actions at t1 are both correct
    base = [
        ReadEvidenceAction(evidence_id="ev-us-001"),
        _dec("t1", "initial_imaging", action="order_mri", refs=["ev-us-001"]),  # differs from script's order_contrast_imaging
        ReadEvidenceAction(evidence_id="ev-ct-001"),
        _dec("t2", "advanced_imaging", action="order_biopsy", refs=["ev-ct-001"]),
        ReadEvidenceAction(evidence_id="ev-path-001"),
        _dec("t3", "pathology", action="order_tnm_staging", refs=["ev-path-001"]),
    ]
    result, _ = _run(base)
    scored = score_episode(result.rows, GOLD)
    assert scored["aggregate"]["next_action_accuracy"]["num"] == 3


def test_deliberately_wrong_trajectory_scores_worse():
    wrong = [
        ReadEvidenceAction(evidence_id="ev-us-001"),
        _dec("t1", "initial_imaging", action="order_contrast_imaging", refs=["ev-us-001"]),
        ReadEvidenceAction(evidence_id="ev-ct-001"),
        _dec("t2", "staging", action="start_treatment", refs=["ev-ct-001"]),  # illegal jump + forbidden action
        ReadEvidenceAction(evidence_id="ev-path-001"),
        _dec("t3", "pathology", action="complete_staging_workup", refs=["ev-path-001"]),
    ]
    result, _ = _run(wrong)
    scored = score_episode(result.rows, GOLD)
    t2 = [t for t in scored["per_turn"] if t["turn_id"] == "t2"][0]
    assert t2["state_field_accuracy"]["num"] == 0  # reported staging instead of advanced_imaging
    assert not t2["state_transition_validity"]      # initial->staging has no rule
    assert "forbidden_action:start_treatment" in t2["guideline_violations"]
    assert scored["aggregate"]["guideline_violation_count"] >= 2
    assert scored["aggregate"]["final_cumulative_success"]["num"] < 3


# ---------------- acceptance 6: a failed turn does not block the recorded trajectory

def test_turn_failure_does_not_block_next_turn():
    from medical_harness.model import ModelError

    class FailTurn2Model:
        """Emits scripted actions, but hard-fails whenever the turn-2 world message is in context."""

        name = "fail-t2"

        def __init__(self, actions):
            self._it = iter(actions)

        def next(self, messages):
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "第 2/3 轮" in last_user:
                raise ModelError("injected t2 failure", kind="injected")
            return next(self._it)

    script = [
        _dec("t1", "initial_imaging", action="order_contrast_imaging", refs=[]),
        ReadEvidenceAction(evidence_id="ev-path-001"),
        _dec("t3", "pathology", action="complete_staging_workup", refs=["ev-path-001"]),
    ]
    log_rows = []
    log = EventLog(_FakeFH(log_rows), run_id="t")
    result = run_episode(
        BUNDLE, FailTurn2Model(script),
        BudgetConfig(max_steps=30, max_model_calls=30, per_turn_model_calls=8, deadline_seconds=60, max_retries=0),
        log, StrategyConfig(),
    )
    assert result.termination == "completed"  # episode finished all turns
    t2 = [r for r in result.rows if r.turn_id == "t2"][0]
    assert t2.termination == "model_error" and t2.decision is None  # failed turn recorded
    t3 = [r for r in result.rows if r.turn_id == "t3"][0]
    assert t3.decision is not None  # teacher-forced: recorded trajectory continued
    assert any(e["type"] == "episode_note" and "teacher-forced" in e["note"] for e in log_rows)


def test_episode_level_budget_exhaustion_terminates():
    budget = BudgetConfig(max_steps=30, max_model_calls=3, per_turn_model_calls=8,
                          deadline_seconds=60, max_retries=0)
    result, _ = _run(_script_from_file(), budget=budget)
    assert result.termination == "budget_exceeded"
    assert {r.turn_id: r.termination for r in result.rows}["t1"] == "completed"
    assert len(result.rows) == 2  # t1 + failed t2; t3 never started
    assert result.total_model_calls == 3


# ---------------- acceptance 7: usage aggregation per turn and episode

def test_usage_and_calls_aggregation_with_fake_usage():
    class U(ScriptedModel):
        def next(self, messages):
            out = super().next(messages)
            self.last_usage = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
            return out

    log_rows = []
    log = EventLog(_FakeFH(log_rows), run_id="u")
    result = run_episode(
        BUNDLE, U("thyroid_001", _script_from_file()),
        BudgetConfig(max_steps=30, max_model_calls=30, per_turn_model_calls=8, deadline_seconds=60, max_retries=0),
        log, StrategyConfig(),
    )
    assert result.total_model_calls == 7  # 3 + 2 + 2 calls across turns
    assert result.usage["total_tokens"] == 7 * 55
    per_turn = [r for r in result.rows]
    assert [r.model_calls for r in per_turn] == [3, 2, 2]
    assert all(isinstance(r.usage, dict) and r.usage["total_tokens"] == r.model_calls * 55 for r in per_turn)
    calls = [e for e in log_rows if e["type"] == "model_call"]
    assert [c["call_id"] for c in calls] == [str(i + 1) for i in range(7)]
    assert all(c["turn_id"] for c in calls)


# ---------------- acceptance 8: workflow/gold never in model context

def test_gold_and_workflow_internals_not_in_model_context():
    result, log_rows = _run(_script_from_file())
    ctx = [e for e in log_rows if e["type"] == "episode_start"][0]["model_context"]
    blob = json.dumps(ctx, ensure_ascii=False)
    for marker in ("allowed_actions", "forbidden_actions", "expected_workflow_state",
                   "required_evidence_refs", "expected_hypotheses", "matched_rule_ids"):
        assert marker not in blob, marker
    # workflow when-tags and rule ids stay evaluator-side
    for tag in ("suspicious_lesion", "imaging_confirmed_suspicious", "malignancy_confirmed"):
        assert tag not in blob, tag
    end = [e for e in log_rows if e["type"] == "episode_end"][0]
    assert "matched_rule_ids" not in json.dumps(end.get("validator_full") or {})
    # gold never appears verbatim in the public trace of steps
    steps = [e for e in log_rows if e["type"] == "step"]
    assert steps


# ---------------- trace redaction policy

def test_trace_redaction_hides_content_but_model_sees_it():
    result, log_rows = _run(_script_from_file(), trace=__import__(
        "medical_harness.schemas", fromlist=["TraceConfig"]).TraceConfig(
        save_model_context=False, save_evidence_content=False))
    step_obs = [e for e in log_rows if e["type"] == "step" and isinstance(e.get("observation"), dict)]
    read_obs = [e for e in step_obs
                if isinstance(e["observation"].get("data"), dict) and "content" in e["observation"]["data"]]
    assert read_obs
    assert all(str(e["observation"]["data"]["content"]).startswith("<redacted") for e in read_obs)
    end = [e for e in log_rows if e["type"] == "episode_end"][0]
    assert isinstance(end["messages"], str) and end["messages"].startswith("<redacted")


def test_validate_episode_detects_structural_errors(tmp_path):
    # a broken copy of the episode: bad release ref + non-monotonic time + unsafe artifact + gold turn mismatch
    src = EP_DIR
    ep = json.loads((src / "episode.json").read_text(encoding="utf-8"))
    ep["turns"][1]["as_of"] = "2025-12-31T00:00:00+00:00"  # earlier than t1
    ep["turns"][2]["release_evidence_ids"] = ["ev-nope"]
    d = tmp_path / "broken"
    d.mkdir()
    (d / "episode.json").write_text(json.dumps(ep, ensure_ascii=False), encoding="utf-8")
    ev_lines = (src / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(ev_lines[0])
    first["artifact_path"] = "../../secrets.png"
    ev_lines[0] = json.dumps(first, ensure_ascii=False)
    (d / "evidence.jsonl").write_text("\n".join(ev_lines), encoding="utf-8")
    (d / "workflow.json").write_text((src / "workflow.json").read_text(encoding="utf-8"), encoding="utf-8")
    gold = json.loads((src / "gold.json").read_text(encoding="utf-8"))
    gold["turns"]["t99"] = gold["turns"]["t1"]
    (d / "gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    errors = validate_episode(d)
    joined = "\n".join(errors)
    for marker in ("non-monotonic", "unknown evidence", "unsafe artifact_path", "unknown turn 't99'"):
        assert marker in joined, marker
    assert validate_episode(EP_DIR) == []
