"""Workflow validator unit tests (acceptance #3, #4-workflow-side)."""
from medical_harness.trajectory import load_episode_bundle
from medical_harness.workflow import guidance_summary, validate_decision
from conftest import REPO

BUNDLE = load_episode_bundle(REPO / "episodes/dev/thyroid_001", with_gold=False)
WF = BUNDLE.workflow
TAGS_T1 = {"ultrasound", "suspicious_lesion"}
TAGS_T2 = TAGS_T1 | {"contrast_imaging", "imaging_confirmed_suspicious"}
TAGS_T3 = TAGS_T2 | {"pathology", "malignancy_confirmed"}


def test_staying_in_same_state_needs_no_rule():
    r = validate_decision(WF, "initial_imaging", "initial_imaging", "order_contrast_imaging", TAGS_T1)
    assert r.transition_valid and r.action_valid and not r.violations


def test_legal_transition_matches_rule_id():
    r = validate_decision(WF, "initial_imaging", "advanced_imaging", "order_biopsy", TAGS_T1)
    assert r.transition_valid and r.action_valid
    assert r.matched_rule_ids == ["initial_imaging->advanced_imaging"]
    # an imaging order is NOT available at the advanced_imaging node (only pathology steps are)
    r2 = validate_decision(WF, "initial_imaging", "advanced_imaging", "order_mri", TAGS_T1)
    assert r2.transition_valid and not r2.action_valid


def test_missing_preconditions_rejected_with_rule_id():
    r = validate_decision(WF, "initial_imaging", "advanced_imaging", "order_mri", set())  # suspicious_lesion absent
    assert not r.transition_valid
    assert any("missing_preconditions:suspicious_lesion" in v for v in r.violations)


def test_unknown_state_rejected():
    r = validate_decision(WF, "initial_imaging", "telepathy", "order_mri", TAGS_T1)
    assert not r.state_valid and not r.transition_valid
    assert any("unknown_state" in v for v in r.violations)


def test_no_transition_rule_rejected():
    r = validate_decision(WF, "initial_imaging", "staging", "order_mri", TAGS_T3)  # skip ahead
    assert not r.transition_valid
    assert any("no_transition:initial_imaging->staging" in v for v in r.violations)


def test_multiple_actions_at_node_all_valid():
    # acceptance #4: different allowed next actions are all structurally valid
    for action in ("order_contrast_imaging", "order_mri", "order_biopsy", "order_fna"):
        r = validate_decision(WF, "initial_imaging", "initial_imaging", action, TAGS_T1)
        assert r.action_valid, action
    r_bad = validate_decision(WF, "initial_imaging", "initial_imaging", "start_treatment", TAGS_T1)
    assert not r_bad.action_valid


def test_pathology_to_staging_requires_malignancy_tag():
    r = validate_decision(WF, "pathology", "staging", "complete_staging_workup", TAGS_T2)  # no pathology tag
    assert not r.transition_valid
    r2 = validate_decision(WF, "pathology", "staging", "complete_staging_workup", TAGS_T3)
    assert r2.transition_valid and r2.matched_rule_ids == ["pathology->staging"]


def test_guidance_summary_hides_evaluator_internals():
    text = guidance_summary(WF)
    assert "initial_imaging" in text and "order_biopsy" in text  # clinician-facing guidance present
    for hidden in ("suspicious_lesion", "imaging_confirmed_suspicious", "malignancy_confirmed",
                   "when", "rule_id", "->"):
        assert hidden not in text, hidden  # no when-tags / rule ids leaked to the model
