"""MedAgentBench importer: 300 tasks → single-turn episodes (+ optional FHIR
multi-turn patient episodes). Key fix over the prototype: $everything pagination —
follow ALL Bundle next links, report truncation instead of treating page 1 as full.

Original sol arrays go to targets.jsonl (evaluator-only). Derived results must
never be reported as original MedAgentBench benchmark scores."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?")
DEFAULT_TASK_FILE = Path(
    "~/Documents/Projects/lab_harness/data/medagentbench/MedAgentBench-main/data/medagentbench/test_data_v2.json"
).expanduser()
FHIR_PAGE_LIMIT = 20  # hard cap; hitting it is reported as truncation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_summary(res: dict) -> str:
    parts: list[str] = []
    code = res.get("code") or {}
    if isinstance(code, dict):
        text = code.get("text")
        if not text:
            codings = code.get("coding") or []
            if codings:
                text = codings[0].get("display") or codings[0].get("code")
        if text:
            parts.append(str(text))
    for key in ("clinicalStatus", "verificationStatus", "valueString", "status"):
        v = res.get(key)
        if isinstance(v, dict):
            v = v.get("text") or (v.get("coding") or [{}])[0].get("display")
        if v:
            parts.append(f"{key}={v}")
    return "; ".join(parts[:4])


def _resource_date(res: dict) -> str | None:
    for key in ("effectiveDateTime", "onsetDateTime", "issued", "authoredOn", "performedDateTime", "recordedDate"):
        v = res.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and isinstance(v.get("start"), str):
            return v["start"]
    return None


def fetch_everything(fhir_base: str, patient_id: str, timeout: float = 15.0) -> dict[str, Any]:
    """Patient/$everything WITH pagination: follow every next link up to FHIR_PAGE_LIMIT."""
    base = fhir_base.rstrip("/")
    url = f"{base}/Patient/{urllib.parse.quote(patient_id)}/$everything"
    entries: list[dict] = []
    pages = 0
    truncated = False
    next_link: str | None = None
    while True:
        pages += 1
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            bundle = json.loads(resp.read().decode("utf-8"))
        entries.extend(bundle.get("entry") or [])
        next_link = next((l.get("url") for l in bundle.get("link") or [] if l.get("rel") == "next"), None)
        if next_link is None:
            break
        if pages >= FHIR_PAGE_LIMIT:
            truncated = True
            break
        url = next_link
    return {"entries": entries, "pages": pages, "truncated": truncated,
            "next_link_pending": bool(next_link) and truncated}


def import_medagentbench(source: Path, out: Path, fhir_base: str | None = None,
                         fhir_patients: int = 5, max_turns: int = 4,
                         per_turn_evidence: int = 6) -> dict[str, Any]:
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(source.read_text(encoding="utf-8"))

    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    excluded_no_mrn: list[str] = []
    no_timestamp = 0
    by_mrn: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        mrn = t.get("eval_MRN")
        if not mrn:
            excluded_no_mrn.append(t.get("id", "?"))
            continue
        by_mrn[mrn].append(t)
        ctx = t.get("context") or ""
        m = _TS_RE.search(ctx)
        turn_time = m.group(0) if m else None
        if turn_time is None:
            no_timestamp += 1
        message = t.get("instruction", "")
        if ctx:
            message = f"{message}\n（任务上下文：{ctx}）"
        episodes.append({
            "episode_id": t["id"],
            "subject_id": mrn,
            "metadata": {"source": "medagentbench", "raw": {"id": t["id"], "eval_MRN": mrn}},
            "turns": [{"turn_id": "t1", "time": turn_time, "message": message, "evidence": []}],
        })
        targets.append({"episode_id": t["id"], "turns": {
            "t1": {"answers": [str(s) for s in (t.get("sol") or [])]},
        }})

    fhir_report: dict[str, Any] = {"enabled": bool(fhir_base), "patients": []}
    if fhir_base:
        top_mrns = sorted(by_mrn, key=lambda m: (-len(by_mrn[m]), m))[:fhir_patients]
        for mrn in top_mrns:
            try:
                with urllib.request.urlopen(
                        f"{fhir_base.rstrip('/')}/Patient?identifier={urllib.parse.quote(mrn)}&_count=1",
                        timeout=15) as resp:
                    bundle = json.loads(resp.read().decode("utf-8"))
                entries = bundle.get("entry") or []
                if not entries:
                    fhir_report["patients"].append({"mrn": mrn, "error": "patient not found"})
                    continue
                pid = entries[0]["resource"]["id"]
                everything = fetch_everything(fhir_base, pid)
                dated = []
                dropped_undated = 0
                for entry in everything["entries"]:
                    res = entry.get("resource") or {}
                    if res.get("resourceType") == "Patient":
                        continue
                    date = _resource_date(res)
                    if not date:
                        dropped_undated += 1
                        continue
                    dated.append((date, res))
                dated.sort(key=lambda x: x[0])
                days: dict[str, list] = defaultdict(list)
                for date, res in dated:
                    days[date[:10]].append((date, res))
                day_keys = sorted(days)[-max_turns:]
                ep_evidence_count = 0
                turns = []
                for i, day in enumerate(day_keys, 1):
                    evs = []
                    for date, res in days[day][:per_turn_evidence]:
                        eid = f"fhir-{res['resourceType'].lower()}-{res.get('id')}"
                        evs.append({
                            "evidence_id": eid,
                            "kind": f"fhir_{res['resourceType'].lower()}",
                            "text": f"[FHIR {res['resourceType']} id={res.get('id')}] date={date}"
                                    + (f" | {_resource_summary(res)}" if _resource_summary(res) else ""),
                            "artifact": None,
                            "source": f"fhir,{res['resourceType']},{res.get('id')}",
                            "metadata": {"tags": [f"fhir_{res['resourceType'].lower()}"]},
                        })
                        ep_evidence_count += 1
                    turns.append({
                        "turn_id": f"t{i}",
                        "time": f"{day}T23:59:59+00:00",
                        "message": f"截至 {day} 的真实病历资源已进入环境（本批 {len(evs)} 条）。",
                        "evidence": evs,
                    })
                if turns:
                    episodes.append({
                        "episode_id": f"patient-{mrn}",
                        "subject_id": mrn,
                        "metadata": {"source": "medagentbench_fhir", "fhir_patient_id": pid,
                                     "pages": everything["pages"], "truncated": everything["truncated"]},
                        "turns": turns,
                    })
                fhir_report["patients"].append({
                    "mrn": mrn, "fhir_patient_id": pid, "pages": everything["pages"],
                    "truncated": everything["truncated"], "n_dated": len(dated),
                    "dropped_undated": dropped_undated, "n_turns": len(turns),
                    "n_evidence": ep_evidence_count,
                })
            except Exception as exc:
                fhir_report["patients"].append({"mrn": mrn, "error": f"{type(exc).__name__}: {exc}"})

    dataset = {
        "schema": "ama-dataset-v0",
        "name": out.name,
        "version": "0.1",
        "description": "MedAgentBench tasks (single-turn) "
                       + ("+ FHIR patient timelines (multi-turn, unscored)" if fhir_base else ""),
        "splits": {"all": [e["episode_id"] for e in episodes]},
        "scorer": "exact_v0",
        "license": "MedAgentBench (see upstream)",
    }
    (out / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes) + "\n", encoding="utf-8")
    (out / "targets.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in targets) + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(source), "source_sha256": _sha(source),
        "n_tasks": len(tasks), "n_imported": len(episodes) - len(fhir_report.get("patients", [])),
        "n_targets": len(targets),
        "excluded_no_mrn": excluded_no_mrn,
        "time_proxy": {"single_turn_without_timestamp": no_timestamp,
                       "rule": "turn time = timestamp parsed from task context; else null (documented, not guessed)"},
        "fhir": fhir_report,
        "note": "derived replay scores must not be reported as original MedAgentBench benchmark scores",
    }
    (out / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
