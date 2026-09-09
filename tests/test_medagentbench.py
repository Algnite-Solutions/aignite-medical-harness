"""MedAgentBench adapter tests (offline tier only; FHIR tier is opt-in live)."""
import json
from pathlib import Path

import pytest

from medical_harness.adapters.medagentbench import (
    _extract_timestamps, load_tasks, profile_patients,
)


@pytest.fixture
def task_file(tmp_path):
    tasks = [
        {"id": "task1_1", "instruction": "age of S1?", "context": "It's 2023-11-13T10:15:00+00:00 now.", "sol": ["50"], "eval_MRN": "S1"},
        {"id": "task1_2", "instruction": "latest lab of S1?", "context": "It's 2024-01-02T09:00:00+00:00 now.", "sol": ["x"], "eval_MRN": "S1"},
        {"id": "task1_3", "instruction": "meds of S1?", "context": "", "sol": ["y"], "eval_MRN": "S1"},
        {"id": "task2_1", "instruction": "age of S2?", "context": "It's 2023-05-05T00:00:00+00:00 now.", "sol": ["30"], "eval_MRN": "S2"},
        {"id": "task3_1", "instruction": "MRN of nobody?", "context": "", "sol": ["Patient not found"], "eval_MRN": None},
    ]
    p = tmp_path / "test_data_v2.json"
    p.write_text(json.dumps(tasks), encoding="utf-8")
    return p


def test_load_tasks_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MEDAGENTBENCH_DATA_FILE"):
        load_tasks(tmp_path / "nope.json")


def test_profile_offline_tier(task_file, tmp_path):
    prof = profile_patients(data_file=task_file, patient_limit=5,
                            fhir_base="http://127.0.0.1:1/fhir")  # unreachable on purpose
    assert prof["source"]["n_tasks"] == 5 and prof["source"]["n_patients"] == 2
    assert prof["fhir"]["reachable"] is False
    assert "docker" in prof["fhir"]["note"]  # honest guidance instead of fabricated timelines
    patients = {p["mrn"]: p for p in prof["patients"]}
    assert patients["S1"]["n_tasks"] == 3
    assert len(patients["S1"]["context_timestamps"]) == 2
    # without FHIR nothing is declared includable — order must not be guessed
    assert all(p["trajectory_candidate"]["included"] is False for p in prof["patients"])
    assert prof["no_mrn_task_ids"] == ["task3_1"]
    # dev set suggestion falls back to task-count ranking, clearly labeled as needing FHIR verification
    assert "S1" in prof["dev_set_suggestion"]["candidates"]


def test_timestamp_extraction():
    assert _extract_timestamps("It's 2023-11-13T10:15:00+00:00 now.") == ["2023-11-13T10:15:00+00:00"]
    assert _extract_timestamps("") == []


def test_real_data_file_if_present():
    from medical_harness.adapters.medagentbench import DEFAULT_DATA_FILE
    if not DEFAULT_DATA_FILE.exists():
        pytest.skip(f"local MedAgentBench copy not present at {DEFAULT_DATA_FILE}")
    tasks = load_tasks(DEFAULT_DATA_FILE)
    assert len(tasks) == 300
    mrns = {t.get("eval_MRN") for t in tasks}
    assert len(mrns) >= 90
