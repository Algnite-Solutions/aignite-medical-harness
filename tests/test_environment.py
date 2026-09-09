"""Invariant tests: visibility (incl. delayed records), patient isolation,
state versioning/atomicity/conflict, no auto-resolution, disabled state tools."""
from datetime import datetime

from medical_harness.environment import TimelineEnvironment
from medical_harness.schemas import (
    Answer,
    AnswerClaim,
    GetStateAction,
    ListEvidenceAction,
    ProposeStateUpdateAction,
    ProposedClaim,
    ReadEvidenceAction,
    SubmitAnswerAction,
)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def visible_ids(env) -> list[str]:
    res = env.execute(ListEvidenceAction())
    assert res.ok
    return [r["evidence_id"] for r in res.data["evidence"]]


# ------------------------------------------------- 1. future / delayed evidence

def test_delayed_evidence_not_listed_read_or_citable(inputs):
    # alpha at 03-03: lab drawn 03-02 but recorded 03-04/03-05 -> invisible
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-03T12:00:00+00:00"))
    ids = visible_ids(env)
    assert ids == ["e-adm-1", "e-med-1"]  # future labs absent

    res = env.execute(ReadEvidenceAction(evidence_id="e-lab-1"))
    assert not res.ok and "not found" in res.error
    assert env.violations["future"] == 1  # internally tagged, never shown to model
    assert res.violation == "future" and res.violation not in (res.error or "")

    propose = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="creatinine", value=1.8, evidence_refs=["e-lab-1"])],
        expected_version=0,
    ))
    assert not propose.ok  # cannot cite invisible evidence
    submit = env.execute(SubmitAnswerAction(answer=Answer(
        question_id="q2",
        claims=[AnswerClaim(key="creatinine", value=1.8, evidence_refs=["e-lab-1"])],
    )))
    assert not submit.ok
    assert len(env.snapshots) == 1  # no partial writes


def test_recorded_time_equal_to_as_of_is_visible(inputs):
    env = TimelineEnvironment(inputs["case_gamma"], as_of=dt("2024-03-01T09:00:00+00:00"))
    assert visible_ids(env) == ["e-g1"]


def test_read_nonexistent_evidence_no_violation(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-01T12:00:00+00:00"))
    res = env.execute(ReadEvidenceAction(evidence_id="e-nope"))
    assert not res.ok
    assert env.violations == {"future": 0, "other_patient": 0}


# ------------------------------------------------- 2. patient isolation

def test_other_patient_evidence_invisible_everywhere(inputs):
    env = TimelineEnvironment(inputs["case_beta"], as_of=dt("2024-03-01T12:00:00+00:00"))
    assert visible_ids(env) == ["e-p02-lab"]  # p-03's same-named lab absent

    res = env.execute(ReadEvidenceAction(evidence_id="e-p03-lab"))
    assert not res.ok
    assert env.violations["other_patient"] == 1

    propose = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="creatinine", value=9.9, evidence_refs=["e-p03-lab"])],
        expected_version=0,
    ))
    assert not propose.ok
    submit = env.execute(SubmitAnswerAction(answer=Answer(
        question_id="q1",
        claims=[AnswerClaim(key="creatinine", value=9.9, evidence_refs=["e-p03-lab"])],
    )))
    assert not submit.ok


# ------------------------------------------------- 3. state versioning

def test_versioning_old_versions_kept_conflict_rejected_atomic(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-01T12:00:00+00:00"))

    ok = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="medication", value="aspirin 100mg daily", evidence_refs=["e-med-1"])],
        expected_version=0,
    ))
    assert ok.ok and ok.data["version"] == 1

    conflict = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="x", value=1, evidence_refs=["e-adm-1"])],
        expected_version=0,  # stale
    ))
    assert not conflict.ok and "version conflict" in conflict.error
    assert env.snapshots[-1].version == 1  # unchanged

    bad_batch = env.execute(ProposeStateUpdateAction(
        claims=[
            ProposedClaim(key="ok_key", value=1, evidence_refs=["e-adm-1"]),
            ProposedClaim(key="bad_key", value=2, evidence_refs=["e-lab-1"]),  # invisible at t0
        ],
        expected_version=1,
    ))
    assert not bad_batch.ok
    assert env.snapshots[-1].version == 1 and len(env.snapshots[-1].claims) == 1  # atomic: nothing written


def test_supersedes_marks_old_claim_and_keeps_history(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-01T12:00:00+00:00"))
    env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="medication", value="aspirin 100mg daily", evidence_refs=["e-med-1"])],
        expected_version=0,
    ))
    first_id = env.snapshots[-1].claims[0].claim_id
    env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="medication", value="clopidogrel 75mg daily",
                              evidence_refs=["e-med-1"], supersedes=first_id)],
        expected_version=1,
    ))
    state = env.execute(GetStateAction()).data
    old, new = state["claims"][0], state["claims"][1]
    assert old["status"] == "superseded" and old["valid_to"] is not None
    assert new["status"] == "active" and new["supersedes"] == first_id
    assert [s.version for s in env.snapshots] == [0, 1, 2]  # old versions retained


def test_supersedes_wrong_key_rejected(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-01T12:00:00+00:00"))
    env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="medication", value="aspirin", evidence_refs=["e-med-1"])],
        expected_version=0,
    ))
    cid = env.snapshots[-1].claims[0].claim_id
    res = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="other_key", value=1, evidence_refs=["e-adm-1"], supersedes=cid)],
        expected_version=1,
    ))
    assert not res.ok


# ------------------------------------------------- 4. conflicts are not auto-resolved

def test_contradictory_claims_both_stay_active(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-05T12:00:00+00:00"))
    env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="creatinine", value=1.8, evidence_refs=["e-lab-1"])],
        expected_version=0,
    ))
    env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="creatinine", value=2.1, evidence_refs=["e-lab-2"])],  # no supersedes
        expected_version=1,
    ))
    claims = env.snapshots[-1].claims
    assert len(claims) == 2
    assert all(c["status"] == "active" for c in
               [c.model_dump() for c in claims])  # runner never marks "latest wins"


# ------------------------------------------------- state tools under full_history

def test_state_tools_disabled_explicitly(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-01T12:00:00+00:00"),
                              allow_state_tools=False)
    res = env.execute(GetStateAction())
    assert not res.ok and "disabled" in res.error
    res = env.execute(ProposeStateUpdateAction(
        claims=[ProposedClaim(key="k", value=1, evidence_refs=["e-adm-1"])], expected_version=0,
    ))
    assert not res.ok and "disabled" in res.error
    assert env.execute(ListEvidenceAction()).ok  # evidence tools unaffected


# ------------------------------------------------- answer validation

def test_abstain_answer_with_claims_rejected(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=dt("2024-03-03T12:00:00+00:00"))
    res = env.execute(SubmitAnswerAction(answer=Answer(
        question_id="q2", abstain=True,
        claims=[AnswerClaim(key="creatinine", value=1.0)],
    )))
    assert not res.ok and "abstain" in res.error
