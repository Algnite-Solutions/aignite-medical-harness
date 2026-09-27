"""Import credentialed MIMIC-IV-Ext-CDM v1.1 into two AMA workflows.

No patient records are bundled with this package. The importer reads an existing,
authorized local copy and writes only to the caller's chosen output directory.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from ..dataset_card import write_dataset_cards
from ..tools import Tool, ToolResult

SOURCE_URL = "https://physionet.org/content/mimic-iv-ext-cdm/1.1/"
PATHOLOGIES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
TABLES = {
    "history_of_present_illness": ("hadm_id", "hpi"),
    "physical_examination": ("hadm_id", "pe"),
    "laboratory_tests": ("hadm_id", "itemid", "valuestr", "ref_range_lower", "ref_range_upper"),
    "microbiology": ("hadm_id", "test_itemid", "valuestr", "spec_itemid"),
    "radiology_reports": ("hadm_id", "note_id", "modality", "region", "exam_name", "text"),
    "discharge_diagnosis": ("hadm_id", "discharge_diagnosis"),
    "discharge_procedures": ("hadm_id", "discharge_procedure"),
    "icd_diagnosis": ("hadm_id", "icd_diagnosis"),
    "icd_procedures": ("hadm_id", "icd_code", "icd_title", "icd_version"),
}
LAB_COLUMNS = ("itemid", "label", "fluid", "category", "count", "corresponding_ids")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _table(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path.name}: expected columns {columns}, found {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed CSV row")
    return rows


def _one(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    result = {}
    for row in rows:
        hadm = row["hadm_id"]
        if hadm in result:
            raise ValueError(f"{name}: duplicate admission {hadm}")
        result[hadm] = row[next(key for key in row if key != "hadm_id")]
    return result


def _group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["hadm_id"]].append(row)
    return grouped


def _lab_lines(rows: list[dict[str, str]], labels: dict[str, str]) -> str:
    return "\n".join(f"{labels.get(r['itemid'], r['itemid'])} [{r['itemid']}]: {r['valuestr']}"
                     + (f" (reference {r['ref_range_lower']}–{r['ref_range_upper']})"
                        if r["ref_range_lower"] or r["ref_range_upper"] else "") for r in rows)


def _full_episode(hadm: str, case: dict, labels: dict[str, str]) -> dict:
    evidence = [{"id": "hpi", "type": "history", "text": case["hpi"]},
                {"id": "physical-examination", "type": "examination", "text": case["pe"]},
                {"id": "laboratory-tests", "type": "laboratory",
                 "text": _lab_lines(case["labs"], labels)}]
    if case["microbiology"]:
        evidence.append({"id": "microbiology", "type": "microbiology",
                         "text": "\n".join(f"Test {r['test_itemid']}, specimen {r['spec_itemid']}: {r['valuestr']}"
                                           for r in case["microbiology"])})
    for index, row in enumerate(case["imaging"], 1):
        evidence.append({"id": f"imaging-{index}", "type": "radiology",
                         "text": f"{row['modality']} {row['region']} — {row['exam_name']} "
                                 f"(note {row['note_id']}):\n{row['text']}"})
    return {"id": hadm, "turns": [{"id": "t1", "evidence": evidence}]}


def _write_dataset(folder: Path, *, mode: str, ids: list[str], cases: list[dict],
                   targets: list[dict], mapping: list[dict], labels: dict[str, str],
                   counts: dict[str, int], source: Path) -> None:
    folder.mkdir()
    interactive = mode == "interactive"
    _write_json(folder / "dataset.json", {
        "schema": "ama-dataset", "name": folder.name, "splits": {"all": ids},
    })
    if interactive:
        episode_rows = [{"id": case["hadm_id"], "turns": [{"id": "t1", "evidence": [
            {"id": "hpi", "type": "history", "text": case["hpi"]}]}]}
                        for case in cases]
    else:
        episode_rows = [_full_episode(c["hadm_id"], c, labels) for c in cases]
    _write_jsonl(folder / "episodes.jsonl", episode_rows)
    _write_jsonl(folder / "cases.jsonl", cases)
    _write_json(folder / "lab_mapping.json", mapping)
    if interactive:
        _write_json(folder / "tool_data_files.json", ["cases.jsonl", "lab_mapping.json"])
    _write_jsonl(folder / "targets.jsonl", targets)
    _write_json(folder / "eval.json", {"scorer": "mimic_cdm"})
    instruction = (
        "You are reviewing a de-identified abdominal-pathology case for research evaluation. "
        "Choose the most likely diagnosis from appendicitis, cholecystitis, diverticulitis, "
        "and pancreatitis. This is not a live clinical decision.\n"
        + ("Start with the HPI. Request physical examination, laboratory tests, microbiology, "
           "and imaging through the available tools as needed. Finish with a JSON Decision "
           "whose answer is {\"diagnosis\": \"one of four labels\", "
           "\"treatment_plan\": \"concise free text\"}. Cite hpi if used; tool "
           "responses are logged but cannot be cited by ID in this AMA version.\n"
           if interactive else
           "All extracted information is provided together. Finish with a JSON Decision "
           "whose answer is {\"diagnosis\": \"one of four labels\"}.\n")
    )
    (folder / "instructions.txt").write_text(instruction, encoding="utf-8")
    report = {"source": SOURCE_URL, "source_dir": str(source.resolve()), "mode": mode,
              "admissions": len(ids), "source_row_counts": counts,
              "max_case_characters": max((len(json.dumps(c, ensure_ascii=False)) for c in cases), default=0),
              "license": "PhysioNet Credentialed Health Data License 1.5.0"}
    _write_json(folder / "import_report.json", report)
    shutil.copyfile(source / "LICENSE.txt", folder / "LICENSE.txt")
    write_dataset_cards(
        folder, name=folder.name, source=SOURCE_URL,
        license_name="PhysioNet Credentialed Health Data License 1.5.0",
        purpose_en=("HPI-first requested-examination research benchmark." if interactive else
                    "Full-information second-reader research benchmark."),
        purpose_zh=("HPI 起始、按需请求检查的研究基准。" if interactive else "完整病例第二读者研究基准。"),
        construction_en="One episode per admission. Discharge outcomes are evaluator-only targets; all source rows are retained in cases.jsonl, lab_mapping.json and targets.jsonl.",
        construction_zh="每次住院一个 Episode。出院结局仅存于评估目标；所有源数据行保存在 cases.jsonl、lab_mapping.json 和 targets.jsonl。",
        episodes=len(ids), turns=len(ids),
        evidence=(len(ids) if interactive else sum(len(ep["turns"][0]["evidence"]) for ep in episode_rows)),
        scorer="mimic_cdm", targets=len(targets),
        limitations_en="Diagnosis is scored as one of four selected pathologies. Treatment plans are recorded but not automatically scored. The source CSVs do not expose event timestamps; some extracted examination text may reflect later care. Full-information cases may exceed model context limits. Keep derived data and run logs on restricted storage.",
        limitations_zh="诊断按四种已筛选病种评分；治疗计划仅记录，不自动评分。源 CSV 不提供事件时间戳，部分查体文本可能反映后续诊疗。完整病例可能超过模型上下文限制。派生数据和运行日志须保存在受限存储中。",
    )


def import_mimic_cdm(source: Path, out: Path, *, limit: int | None = None,
                     ids: list[str] | None = None) -> dict:
    """Build both self-contained AMA datasets; refuse to overwrite existing outputs."""
    source, out = Path(source), Path(out)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    tables = {name: _table(source / f"{name}.csv", columns) for name, columns in TABLES.items()}
    mapping = _table(source / "lab_test_mapping.csv", LAB_COLUMNS)
    pathology = json.loads((source / "pathology_ids.json").read_text(encoding="utf-8"))
    if set(pathology) != set(PATHOLOGIES):
        raise ValueError("unexpected pathology labels")
    labels_by_id: dict[str, str] = {}
    for label, admission_ids in pathology.items():
        for hadm in admission_ids:
            key = str(hadm)
            if key in labels_by_id:
                raise ValueError(f"admission in multiple pathology groups: {key}")
            labels_by_id[key] = label
    hpi = _one(tables["history_of_present_illness"], "history_of_present_illness")
    pe = _one(tables["physical_examination"], "physical_examination")
    discharge = _one(tables["discharge_diagnosis"], "discharge_diagnosis")
    expected = set(hpi)
    if not expected or set(pe) != expected or set(discharge) != expected or set(labels_by_id) != expected:
        raise ValueError("HPI, examination, discharge diagnosis, and pathology IDs do not match")
    grouped = {name: _group(rows) for name, rows in tables.items() if name not in
               {"history_of_present_illness", "physical_examination", "discharge_diagnosis"}}
    if any(set(rows) - expected for rows in grouped.values()):
        raise ValueError("source table references unknown admission")
    lab_labels = {r["itemid"]: r["label"] for r in mapping if r["itemid"]}
    if set(r["itemid"] for r in tables["laboratory_tests"]) - set(lab_labels):
        raise ValueError("laboratory result has no mapping")
    selected = sorted(expected, key=int) if ids is None else list(dict.fromkeys(ids))
    if set(selected) - expected:
        raise ValueError(f"unknown admission IDs: {sorted(set(selected) - expected)}")
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no admissions selected")
    cases = [{"hadm_id": hadm, "hpi": hpi[hadm], "pe": pe[hadm],
              "labs": grouped["laboratory_tests"].get(hadm, []),
              "microbiology": grouped["microbiology"].get(hadm, []),
              "imaging": grouped["radiology_reports"].get(hadm, [])} for hadm in selected]
    target_rows = [{"id": hadm, "turns": {"t1": {
        "answer": {"diagnosis": labels_by_id[hadm]},
        "discharge_diagnosis": discharge[hadm],
        "discharge_procedures": grouped["discharge_procedures"].get(hadm, []),
        "icd_diagnosis": grouped["icd_diagnosis"].get(hadm, []),
        "icd_procedures": grouped["icd_procedures"].get(hadm, []),
    }}} for hadm in selected]
    counts = {name: len(rows) for name, rows in tables.items()}
    counts["lab_test_mapping"] = len(mapping)
    final_names = ["mimic_cdm_interactive", "mimic_cdm_full_info"]
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in final_names):
        raise FileExistsError("MIMIC-CDM output already exists; choose an empty output parent")
    staging = out / f".mimic_cdm_import_{uuid4().hex}"
    staging.mkdir()
    try:
        for mode, name in (("interactive", final_names[0]), ("full_info", final_names[1])):
            _write_dataset(staging / name, mode=mode, ids=selected, cases=cases,
                           targets=target_rows, mapping=mapping, labels=lab_labels,
                           counts=counts, source=source)
        for name in final_names:
            os.rename(staging / name, out / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"admissions": len(selected), "outputs": [str(out / name) for name in final_names],
            "source_row_counts": counts}


def load_cases(folder: Path) -> dict:
    cases = {}
    with (folder / "cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            if case["hadm_id"] in cases:
                raise ValueError(f"duplicate case data: {case['hadm_id']}")
            cases[case["hadm_id"]] = case
    mapping = json.loads((folder / "lab_mapping.json").read_text(encoding="utf-8"))
    return {"cases": cases, "mapping": mapping}


def make_case_tools(data: dict, episode_id: str) -> list[Tool]:
    """Bind trusted built-in retrieval tools to one admission, never to targets."""
    case = data["cases"][episode_id]
    mapping = data["mapping"]
    scan_cursor: dict[tuple[str, str], int] = defaultdict(int)

    def physical_examination() -> ToolResult:
        return ToolResult(f"Physical examination:\n{case['pe']}")

    def search_lab_tests(query: str) -> list[dict]:
        key = query.strip().casefold()
        if len(key) < 2:
            raise ValueError("provide at least two characters")
        found = [r for r in mapping if key in r["label"].casefold() or key == r["itemid"]]
        return [{"itemid": r["itemid"], "label": r["label"], "fluid": r["fluid"]}
                for r in found[:20]]

    def laboratory_tests(names: list[str]) -> ToolResult:
        if not isinstance(names, list) or not names or len(names) > 20 or not all(
                isinstance(name, str) and name.strip() for name in names):
            raise ValueError("names must contain 1–20 non-empty lab names or itemids")
        results = []
        for name in names:
            key = name.strip().casefold()
            matched = [r for r in mapping if r["label"].casefold() == key or r["itemid"] == key]
            if not matched:
                results.append({"requested": name, "status": "unknown test; use search_lab_tests"})
                continue
            equivalent = set()
            for row in matched:
                equivalent.add(row["itemid"])
                equivalent.update(str(value) for value in json.loads(row["corresponding_ids"]))
            values = [r for r in case["labs"] if r["itemid"] in equivalent]
            if not values:
                results.append({"requested": name, "status": "not available for this admission"})
                continue
            for row in values:
                evidence_id = f"lab-{row['itemid']}"
                results.append({"requested": name, "evidence_id": evidence_id,
                                "label": next((m["label"] for m in mapping
                                               if m["itemid"] == row["itemid"]), row["itemid"]),
                                "value": row["valuestr"],
                                "ref_range_lower": row["ref_range_lower"],
                                "ref_range_upper": row["ref_range_upper"]})
        return ToolResult(json.dumps(results, ensure_ascii=False))

    def microbiology() -> ToolResult:
        if not case["microbiology"]:
            return ToolResult("No microbiology results available for this admission.")
        return ToolResult(json.dumps({"results": case["microbiology"]}, ensure_ascii=False))

    def imaging(modality: str, region: str) -> ToolResult:
        if not modality.strip() or not region.strip():
            raise ValueError("modality and region are required")
        aliases = {"xray": "radiograph", "x-ray": "radiograph", "ctu": "ct",
                   "mrcp": "mri", "mre": "mri", "mra": "mri"}
        want_modality = aliases.get(modality.strip().casefold(), modality.strip().casefold())
        want_region = region.strip().casefold()
        matches = [(i, row) for i, row in enumerate(case["imaging"], 1)
                   if aliases.get(row["modality"].casefold(), row["modality"].casefold())
                   == want_modality and row["region"].casefold() == want_region]
        key = (want_modality, want_region)
        index = scan_cursor[key]
        if index >= len(matches):
            return ToolResult("No further matching imaging report available.")
        scan_cursor[key] += 1
        source_index, row = matches[index]
        evidence_id = f"imaging-{source_index}"
        return ToolResult(json.dumps({"evidence_id": evidence_id, **row}, ensure_ascii=False))

    object_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        Tool("physical_examination", "Request the patient's physical examination.", object_schema,
             physical_examination),
        Tool("search_lab_tests", "Find global laboratory test names or item IDs; does not reveal patient results.",
             {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"], "additionalProperties": False}, search_lab_tests),
        Tool("laboratory_tests", "Request results for up to 20 named laboratory tests or item IDs.",
             {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}},
              "required": ["names"], "additionalProperties": False}, laboratory_tests),
        Tool("microbiology", "Request the available microbiology results.", object_schema, microbiology),
        Tool("imaging", "Request one imaging findings report by modality and anatomical region.",
             {"type": "object", "properties": {"modality": {"type": "string"},
                                                "region": {"type": "string"}},
              "required": ["modality", "region"], "additionalProperties": False}, imaging),
    ]


def score_mimic_cdm(dataset, decisions: dict[str, list[dict]]) -> dict:
    """Exact four-way diagnosis score; retain treatment answers without judging them."""
    from ..scorer import expected_rows

    per_episode = {}
    correct = completed = 0
    per_pathology = {label: {"correct": 0, "total": 0} for label in PATHOLOGIES}
    for episode in dataset.episodes:
        target = dataset.targets.get(episode.id)
        wanted = (target.turns.get("t1", {}).get("answer", {}).get("diagnosis")
                  if target else None)
        if wanted not in per_pathology:
            raise ValueError(f"missing or invalid pathology target: {episode.id}")
        row = expected_rows(episode, decisions)[0]
        answer = (row.get("decision") or {}).get("answer")
        given = answer.get("diagnosis") if isinstance(answer, dict) else None
        normalized = given.strip().casefold() if isinstance(given, str) else None
        submitted = row.get("decision") is not None
        is_correct = normalized == wanted
        completed += int(submitted)
        correct += int(is_correct)
        per_pathology[wanted]["correct"] += int(is_correct)
        per_pathology[wanted]["total"] += 1
        per_episode[episode.id] = {"submitted": submitted, "correct": is_correct,
                                   "reference_pathology": wanted,
                                   "predicted_pathology": normalized,
                                   "termination": row["termination"]}
    total = len(dataset.episodes)
    return {"scorer": "mimic_cdm", "per_episode": per_episode,
            "aggregate": {"diagnosis_accuracy": {"num": correct, "den": total,
                                                 "value": correct / total if total else None},
                          "completion": {"num": completed, "den": total,
                                         "value": completed / total if total else None},
                          "per_pathology": per_pathology,
                          "treatment_scored": False}}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Import MIMIC-IV-Ext-CDM into two AMA datasets")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="parent of both processed datasets")
    parser.add_argument("--limit", type=int, help="small deterministic subset for local verification")
    parser.add_argument("--id", action="append", dest="ids", help="select one admission ID")
    args = parser.parse_args(argv)
    report = import_mimic_cdm(args.source, args.out, limit=args.limit, ids=args.ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
