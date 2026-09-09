"""Data profiler: structural observation over evidence streams.

This is the "observe challenges from data" instrument. v0 reports only
structure-derivable phenomena (lag, duplicates, counts). Claim-level phenomena
(conflicts, revisions) become observable once real data flows through the
adapter — deliberately not pre-enumerated here.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .schemas import CaseInput


def profile_case(case: CaseInput) -> dict[str, Any]:
    lags_h = [
        (e.recorded_time - e.event_time).total_seconds() / 3600.0
        for e in case.evidence
    ]
    seen: dict[tuple[str, str], int] = {}
    for e in case.evidence:
        key = (e.patient_id, e.content.strip())
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(n - 1 for n in seen.values() if n > 1)
    modalities: dict[str, int] = {}
    patients: dict[str, int] = {}
    for e in case.evidence:
        modalities[e.modality] = modalities.get(e.modality, 0) + 1
        patients[e.patient_id] = patients.get(e.patient_id, 0) + 1
    n_questions = sum(len(tp.questions) for tp in case.task.timepoints)
    return {
        "case_id": case.case_id,
        "n_timepoints": len(case.task.timepoints),
        "n_questions": n_questions,
        "n_evidence": len(case.evidence),
        "patients": patients,
        "modalities": modalities,
        "lag_hours": {
            "min": min(lags_h) if lags_h else None,
            "median": statistics.median(lags_h) if lags_h else None,
            "max": max(lags_h) if lags_h else None,
        },
        "duplicate_evidence": duplicates,
        "missing_recorded_time": 0,  # schema requires recorded_time; adapter must count honestly for real data
    }


def profile_dir(data_dir: Path) -> dict[str, Any]:
    from .runner import load_inputs

    cases = load_inputs(Path(data_dir))
    per_case = [profile_case(c) for c in cases]
    return {
        "n_cases": len(per_case),
        "n_evidence": sum(c["n_evidence"] for c in per_case),
        "per_case": per_case,
        "notes": [
            "recorded_time 在当前 schema 中必填，missing 计数为 0；接入真实数据时需放宽并如实计数。",
            "冲突率、修订率需要 claim 级信息（来自状态更新或人工标注），当前仅做结构统计——这是待从真实数据观测的部分。",
        ],
    }
