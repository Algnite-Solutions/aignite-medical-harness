"""MedAgentBench adapter (v0.2, tiered).

Tier 1 (offline, always available): audit the official task file by patient
MRN — task counts, ids, timestamps mentioned in task contexts, and an
inclusion audit for trajectory candidates.

Tier 2 (requires the official FHIR docker): aggregate real FHIR resources per
patient and order them by clinical time into turns. When the server is not
reachable we report that honestly instead of fabricating timelines.

We never convert tasks that cannot be reliably ordered into trajectories, and
we never report replay scores as original MedAgentBench benchmark scores.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DATA_FILE = Path(
    "~/Documents/Projects/lab_harness/data/medagentbench/MedAgentBench-main/data/medagentbench/test_data_v2.json"
).expanduser()
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?")


def load_tasks(data_file: Path | None) -> list[dict[str, Any]]:
    path = Path(data_file) if data_file else DEFAULT_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"MedAgentBench task file not found: {path} (set --data-file or MEDAGENTBENCH_DATA_FILE)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_timestamps(context: str) -> list[str]:
    return _TS_RE.findall(context or "")


def _fhir_reachable(fhir_base: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(fhir_base.rstrip("/") + "/metadata", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def fhir_patient_snapshot(fhir_base: str, mrn: str, timeout: float = 10.0) -> dict[str, Any]:
    """Fetch a patient's resources via Patient/$everything (official docker tier).

    Returns {"patient_id":..., "resources": [{"resourceType","id","date":...}], "resource_types": {...}}
    """
    import urllib.parse

    base = fhir_base.rstrip("/")
    with urllib.request.urlopen(f"{base}/Patient?identifier={urllib.parse.quote(mrn)}&_count=1", timeout=timeout) as resp:
        bundle = json.loads(resp.read().decode("utf-8"))
    entries = bundle.get("entry") or []
    if not entries:
        return {"patient_id": None, "resources": [], "resource_types": {}, "error": f"no patient with MRN {mrn}"}
    patient_id = entries[0]["resource"]["id"]
    with urllib.request.urlopen(f"{base}/Patient/{patient_id}/$everything", timeout=timeout) as resp:
        everything = json.loads(resp.read().decode("utf-8"))
    resources = []
    types: dict[str, int] = defaultdict(int)
    for entry in everything.get("entry") or []:
        res = entry.get("resource") or {}
        rtype = res.get("resourceType", "?")
        types[rtype] += 1
        date = None
        for key in ("effectiveDateTime", "effectivePeriod", "onsetDateTime", "issued", "authoredOn", "performedDateTime", "recordedDate"):
            if key in res:
                date = res[key] if isinstance(res[key], str) else (res[key].get("start") if isinstance(res[key], dict) else None)
                break
        if rtype != "Patient":
            resources.append({"resourceType": rtype, "id": res.get("id"), "date": date})
    dated = [r for r in resources if r["date"]]
    resources.sort(key=lambda r: r["date"] or "")
    return {
        "patient_id": patient_id,
        "resources": resources,
        "n_dated": len(dated),
        "resource_types": dict(types),
        "date_range": [min(r["date"] for r in dated), max(r["date"] for r in dated)] if dated else None,
    }


def profile_patients(data_file: Path | None = None, patient_limit: int = 5, fhir_base: str = "http://localhost:8080/fhir") -> dict[str, Any]:
    tasks = load_tasks(data_file)
    by_mrn: dict[str, list[dict]] = defaultdict(list)
    no_mrn: list[str] = []
    for t in tasks:
        mrn = t.get("eval_MRN")
        if mrn:
            by_mrn[mrn].append(t)
        else:
            no_mrn.append(t.get("id", "?"))

    reachable = _fhir_reachable(fhir_base)
    patients = []
    for mrn, mrn_tasks in sorted(by_mrn.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        timestamps = sorted({ts for t in mrn_tasks for ts in _extract_timestamps(t.get("context") or "")})
        entry: dict[str, Any] = {
            "mrn": mrn,
            "n_tasks": len(mrn_tasks),
            "task_ids": [t["id"] for t in mrn_tasks],
            "context_timestamps": timestamps,
            "trajectory_candidate": {
                "included": False,
                "reason": "requires FHIR resource ordering (start the official docker)",
            },
        }
        if reachable and len(patients) < patient_limit:
            try:
                snap = fhir_patient_snapshot(fhir_base, mrn)
                entry["fhir"] = {k: snap.get(k) for k in ("patient_id", "n_dated", "resource_types", "date_range")}
                clinical_types = {"Observation", "Condition", "MedicationRequest", "Procedure", "DiagnosticReport"}
                n_types = len(clinical_types & set(snap.get("resource_types", {})))
                enough_dates = snap.get("n_dated", 0) >= 3
                entry["trajectory_candidate"] = {
                    "included": bool(enough_dates and n_types >= 2),
                    "reason": (
                        "ok" if (enough_dates and n_types >= 2)
                        else f"n_dated={snap.get('n_dated', 0)} (need >=3), clinical_types={n_types} (need >=2)"
                    ),
                    "rule": ">=3 dated resources and >=2 of Observation/Condition/MedicationRequest/Procedure/DiagnosticReport",
                }
            except Exception as exc:
                entry["fhir"] = {"error": f"{type(exc).__name__}: {exc}"}
        patients.append(entry)

    dev_pool = [p["mrn"] for p in patients if p["trajectory_candidate"]["included"]]
    if not dev_pool:
        dev_pool = [p["mrn"] for p in patients[:patient_limit]]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "data_file": str(Path(data_file) if data_file else DEFAULT_DATA_FILE),
            "n_tasks": len(tasks),
            "n_patients": len(by_mrn),
            "note": "official MedAgentBench task file; replay/derived results must NOT be reported as original benchmark scores",
        },
        "fhir": {
            "base": fhir_base,
            "reachable": reachable,
            "note": (
                "official FHIR server reachable; per-patient resource aggregation active"
                if reachable else
                "FHIR server NOT reachable; start the official environment (docker pull jyxsu6/medagentbench:latest && docker run -p 8080:8080 medagentbench) to enable resource-level timelines. Offline tier audits tasks only and marks trajectory candidates excluded rather than guessing order."
            ),
        },
        "no_mrn_task_ids": no_mrn,
        "patients": patients,
        "dev_set_suggestion": {
            "rule": ">=3 tasks per patient first; FHIR tier then verifies >=3 dated resources and >=2 clinical resource types",
            "candidates": dev_pool[:patient_limit],
        },
    }
