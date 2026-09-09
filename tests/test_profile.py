"""Profiler tests: the 'observe from data' pipeline produces correct structural stats."""
from medical_harness.profile import profile_case, profile_dir

from conftest import DATA


def test_alpha_lag_stats(inputs):
    p = profile_case(inputs["case_alpha"])
    # lags: e-adm-1 0.5h, e-med-1 1.0h, e-lab-1 49h, e-lab-2 74h
    assert p["n_evidence"] == 4
    assert p["lag_hours"]["min"] == 0.5
    assert p["lag_hours"]["max"] == 74.0
    assert p["lag_hours"]["median"] == 25.0
    assert p["patients"] == {"p-01": 4}
    assert p["modalities"] == {"note": 1, "order": 1, "lab": 2}
    assert p["duplicate_evidence"] == 0
    assert p["missing_recorded_time"] == 0  # schema requires recorded_time (noted honestly)
    assert p["n_questions"] == 3 and p["n_timepoints"] == 3


def test_profile_dir_covers_all_cases():
    p = profile_dir(DATA / "inputs")
    assert p["n_cases"] == 3
    assert p["n_evidence"] == 8
    assert {c["case_id"] for c in p["per_case"]} == {"case_alpha", "case_beta", "case_gamma"}
    assert any("claim 级" in n for n in p["notes"])  # honest boundary: conflicts need claim-level data
