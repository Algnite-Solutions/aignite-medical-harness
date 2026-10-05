"""Prepare local MedAgentBench tasks and FHIR data; no model or grader is run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..data import validate_dataset
from ..dataset_card import write_dataset_cards

RESOURCES = ("Patient", "Observation")
SNAPSHOT_SCHEMA = "medagentbench-fhir-snapshot-v1"
# Existing seed-42, three-per-task sample in MedAgentBench configs/samples/.
SMOKE_IDS = tuple(
    f"task{task}_{sample}"
    for task, samples in (
        (1, (1, 4, 21)), (2, (8, 9, 24)), (3, (5, 8, 24)),
        (4, (4, 22, 24)), (5, (3, 18, 29)), (6, (2, 14, 19)),
        (7, (1, 3, 7)), (8, (8, 17, 20)), (9, (1, 7, 18)),
        (10, (18, 21, 23)),
    ) for sample in samples
)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{number}: expected an object")
                rows.append(value)
    return rows


def _base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("FHIR base must be an HTTP(S) URL without credentials, query or fragment")
    return value.rstrip("/") + "/"


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/fhir+json, application/json"})
    # Local Docker services must not be routed through a configured HTTP proxy.
    if urllib.parse.urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()
    with opener.open(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"FHIR response is not an object: {url}")
    return value


def _unique_resources(rows: list[dict], resource_type: str) -> list[dict]:
    indexed = {}
    for row in rows:
        resource_id = row.get("id")
        if row.get("resourceType") != resource_type or not isinstance(resource_id, str) or not resource_id:
            raise ValueError(f"invalid {resource_type} resource or missing ID")
        if resource_id in indexed and indexed[resource_id] != row:
            raise ValueError(f"conflicting {resource_type}/{resource_id} records")
        indexed[resource_id] = row
    # Keep FHIR result order, including the order of measurements with equal times.
    return list(indexed.values())


def _download_resource(base: str, resource_type: str, timeout: float) -> tuple[list[dict], dict]:
    url = base + resource_type + "?_count=5000&_total=accurate&_format=json"
    origin = urllib.parse.urlsplit(base)
    visited, rows, total = set(), [], None
    while url:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
            raise ValueError(f"FHIR pagination left the configured server: {url}")
        if url in visited:
            raise ValueError(f"FHIR pagination loop for {resource_type}")
        visited.add(url)
        bundle = _get_json(url, timeout)
        if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "searchset":
            raise ValueError(f"expected a FHIR search Bundle for {resource_type}")
        current_total = bundle.get("total")
        if current_total is not None:
            if type(current_total) is not int or current_total < 0:
                raise ValueError(f"invalid FHIR total for {resource_type}")
            if total is not None and total != current_total:
                raise ValueError(f"FHIR {resource_type} total changed during export")
            total = current_total
        entries = bundle.get("entry", [])
        if not isinstance(entries, list):
            raise ValueError(f"invalid FHIR entries for {resource_type}")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
                raise ValueError(f"missing FHIR resource in {resource_type} page")
            rows.append(entry["resource"])
        if len(visited) % 10 == 0:
            print(f"{resource_type}: {len(rows)} records read ({len(visited)} pages)", flush=True)
        links = bundle.get("link", [])
        if not isinstance(links, list) or any(not isinstance(link, dict) for link in links):
            raise ValueError(f"invalid FHIR pagination links for {resource_type}")
        next_links = [link.get("url") for link in links if link.get("relation") == "next"]
        if len(next_links) > 1 or any(not isinstance(link, str) or not link for link in next_links):
            raise ValueError(f"invalid FHIR next page for {resource_type}")
        url = urllib.parse.urljoin(url, next_links[0]) if next_links else None
    unique = _unique_resources(rows, resource_type)
    if total is not None and len(unique) != total:
        raise ValueError(f"incomplete {resource_type} export: expected {total}, got {len(unique)}")
    return unique, {"pages": len(visited), "received": len(rows), "count": len(unique), "total": total}


def export_snapshot(folder: Path, fhir_base: str, *, timeout: float = 60) -> dict:
    """Export all patients and observations; callers publish only a complete dataset."""
    base = _base_url(fhir_base)
    folder.mkdir()
    resources = {}
    for resource_type in RESOURCES:
        print(f"exporting {resource_type}...", flush=True)
        rows, stats = _download_resource(base, resource_type, timeout)
        path = folder / f"{resource_type}.jsonl"
        _jsonl(path, rows)
        resources[resource_type] = {**stats, "sha256": _sha(path)}
        print(f"{resource_type}: {len(rows)} records, {stats['pages']} pages", flush=True)
    metadata = {"schema": SNAPSHOT_SCHEMA, "complete": True, "fhir_base": base,
                "exported_at": datetime.now(timezone.utc).isoformat(), "resources": resources}
    _json(folder / "metadata.json", metadata)
    return metadata


def _load_snapshot(folder: Path) -> tuple[dict, dict[str, list[dict]]]:
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SNAPSHOT_SCHEMA or metadata.get("complete") is not True:
        raise ValueError("snapshot is incomplete or has an unsupported schema")
    _base_url(metadata["fhir_base"])
    data = {}
    for resource_type in RESOURCES:
        path = folder / f"{resource_type}.jsonl"
        stats = metadata.get("resources", {}).get(resource_type, {})
        if stats.get("sha256") != _sha(path):
            raise ValueError(f"snapshot checksum mismatch: {path.name}")
        rows = _read_jsonl(path)
        unique = _unique_resources(rows, resource_type)
        if len(rows) != len(unique) or stats.get("count") != len(unique):
            raise ValueError(f"snapshot count mismatch: {resource_type}")
        if stats.get("total") is not None and stats["total"] != len(unique):
            raise ValueError(f"snapshot total mismatch: {resource_type}")
        data[resource_type] = unique
    return metadata, data


def _patients(resources: list[dict]) -> list[dict]:
    patients = []
    for resource in resources:
        row = {"patient_id": resource["id"], "identifiers": resource.get("identifier", []),
               "names": resource.get("name", [])}
        for source, destination in (("birthDate", "birth_date"), ("gender", "gender")):
            if source in resource:
                row[destination] = resource[source]
        patients.append(row)
    return patients


def _patient_reference(reference: str, patient_ids: set[str]) -> str:
    if not isinstance(reference, str):
        raise ValueError("Observation has no patient reference")
    path = urllib.parse.urlsplit(reference).path
    match = re.search(r"(?:^|/)Patient/([^/]+)(?:/_history/[^/]+)?$", path)
    if not match or match.group(1) not in patient_ids:
        raise ValueError(f"unresolved Observation patient reference: {reference}")
    return match.group(1)


def _observation(resource: dict) -> dict:
    row = {"source_id": resource["id"], "evidence_id": f"observation-{resource['id']}"}
    code = resource.get("code")
    if not isinstance(code, dict):
        raise ValueError(f"Observation/{resource['id']} has no code")
    row["code"] = code
    label = code.get("text") or next((x.get("display") for x in code.get("coding", []) if x.get("display")), None)
    if label:
        row["label"] = label
    values = [(key, value) for key, value in resource.items() if key.startswith("value")]
    if len(values) > 1:
        raise ValueError(f"Observation/{resource['id']} has multiple value fields")
    if values:
        key, value = values[0]
        row["value_type"] = key
        if key == "valueQuantity":
            if "value" in value:
                row["value"] = value["value"]
            for field in ("unit", "system", "code", "comparator"):
                if field in value:
                    row[{"system": "unit_system", "code": "unit_code"}.get(field, field)] = value[field]
        else:
            row["value"] = value
    for field in ("status", "category", "component", "dataAbsentReason"):
        if field in resource:
            row[field] = resource[field]
    # Preserve every effective[x] form; timestamps are not rounded or converted.
    for field, value in resource.items():
        if field.startswith("effective"):
            row[field] = value
    if "effectiveDateTime" in resource:
        row["observed_at"] = resource["effectiveDateTime"]
    if "referenceRange" in resource:
        row["reference_ranges"] = resource["referenceRange"]
    return row


def convert_cases(data: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Normalize source facts without calculating any benchmark answer."""
    patients = _patients(data["Patient"])
    cases = {p["patient_id"]: {"patient_id": p["patient_id"], "observations": []} for p in patients}
    patient_ids = set(cases)
    for resource in data["Observation"]:
        patient_id = _patient_reference(resource.get("subject", {}).get("reference"), patient_ids)
        cases[patient_id]["observations"].append(_observation(resource))
    return patients, list(cases.values())


def _tasks(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("test_data_v2.json must contain a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not re.fullmatch(r"task(?:10|[1-9])_[1-9][0-9]*", str(row.get("id", ""))):
            raise ValueError("invalid MedAgentBench case ID")
        if row["id"] in seen:
            raise ValueError(f"duplicate case ID: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError(f"{row['id']}: missing instruction")
        if not isinstance(row.get("context", ""), str):
            raise ValueError(f"{row['id']}: context must be text")
    return rows


def import_medagentbench(source: Path, out: Path, *, fhir_base: str = "http://localhost:8080/fhir/",
                        snapshot: Path | None = None, ids: list[str] | None = None,
                        limit: int | None = None, timeout: float = 60) -> dict:
    source, out = Path(source).resolve(), Path(out).absolute()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    task_file = source / "data/medagentbench/test_data_v2.json"
    func_file = source / "data/medagentbench/funcs_v1.json"
    grader_dir = source / "src/server/tasks/medagentbench"
    files = {"tasks": task_file, "functions": func_file,
             "refsol": grader_dir / "refsol.py", "utils": grader_dir / "utils.py"}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))
    tasks = _tasks(task_file)
    functions = json.loads(func_file.read_text(encoding="utf-8"))
    if not isinstance(functions, list):
        raise ValueError("funcs_v1.json must be a list")
    if ids is not None:
        wanted = set(ids)
        unknown = wanted - {row["id"] for row in tasks}
        if unknown:
            raise ValueError(f"unknown task IDs: {sorted(unknown)}")
        tasks = [row for row in tasks if row["id"] in wanted]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no tasks selected")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Failed exports/conversions never publish a partial dataset.
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        if snapshot is None:
            export_snapshot(stage / "snapshot", fhir_base, timeout=timeout)
        else:
            original = Path(snapshot).resolve()
            _load_snapshot(original)
            (stage / "snapshot").mkdir()
            for name in ("metadata.json", "Patient.jsonl", "Observation.jsonl"):
                shutil.copyfile(original / name, stage / "snapshot" / name)
        metadata, raw = _load_snapshot(stage / "snapshot")
        patients, cases = convert_cases(raw)
        _jsonl(stage / "patients.jsonl", patients)
        _jsonl(stage / "cases.jsonl", cases)
        episodes = []
        for task in tasks:
            evidence = ([{"id": "context", "type": "context", "text": task["context"]}]
                        if task.get("context", "").strip() else [])
            episodes.append({"id": task["id"], "turns": [{"id": "t1", "observation": task["instruction"],
                                                           "evidence": evidence}]})
        selected = [row["id"] for row in tasks]
        smoke = [task_id for task_id in SMOKE_IDS if task_id in set(selected)]
        _json(stage / "dataset.json", {"schema": "ama-dataset", "name": out.name,
                                       "splits": {"test": selected, "smoke": smoke}})
        _jsonl(stage / "episodes.jsonl", episodes)
        _jsonl(stage / "targets.jsonl", [{"id": task["id"], "turns": {"t1": {"case_data": task}}}
                                        for task in tasks])
        _json(stage / "eval.json", {"scorer": "medagentbench"})
        (stage / "instructions.txt").write_text(
            "Complete the requested patient lookup, calculation, or simulated clinical operation.\n"
            "Use only the task context and facts released by tools. Task times are fixed by the context.\n"
            "The task answer is an object with one field, answers, containing the JSON list that would\n"
            "be supplied to FINISH in MedAgentBench. For operations without a requested answer use [].\n"
            "Patient-not-found and missing-measurement answers follow the individual task instructions.\n"
            "Operations are simulated; they do not change the local patient records.\n",
            encoding="utf-8")
        (stage / "evaluator").mkdir()
        for name in ("refsol", "utils"):
            shutil.copyfile(files[name], stage / "evaluator" / files[name].name)
        shutil.copyfile(func_file, stage / "evaluator" / func_file.name)
        if (source / "LICENSE").is_file():
            shutil.copyfile(source / "LICENSE", stage / "LICENSE")
        report = {"source_dir": str(source), "source_sha256": {key: _sha(path) for key, path in files.items()},
                  "fhir_base": metadata["fhir_base"], "snapshot_schema": metadata["schema"],
                  "snapshot_exported_at": metadata["exported_at"], "snapshot_resources": metadata["resources"],
                  "selected_ids": selected, "task_count": len(tasks), "patient_count": len(patients),
                  "observation_count": len(raw["Observation"]), "smoke_ids": smoke,
                  "per_task": dict(Counter(task_id.split("_")[0] for task_id in selected)),
                  "anomalies": [], "tools_implemented": False, "scorer_implemented": False}
        _json(stage / "import_report.json", report)
        write_dataset_cards(
            stage, name=out.name, source="MedAgentBench v2; " + str(source),
            license_name="See copied source LICENSE and upstream data terms.",
            purpose_en="Local task and patient-data preparation for an AMA adaptation of MedAgentBench.",
            purpose_zh="为 MedAgentBench 的 AMA 本地适配准备任务、患者数据和评分快照。",
            construction_en="One episode per task. All patients and observations are exported independently of task selection. References are evaluator-only.",
            construction_zh="一个任务一个 Episode。患者和观察记录全量导出，不随任务筛选缩小；评分字段仅保存在 targets。",
            episodes=len(tasks), turns=len(tasks), evidence=sum(len(ep["turns"][0]["evidence"]) for ep in episodes),
            scorer="medagentbench (not yet implemented)", targets=len(tasks),
            limitations_en="Importer only: clinical tools and scorer are not implemented. This is not an official benchmark result. No answers or clinical recommendations are precomputed. FHIR search semantics must be validated in the future grading adapter.",
            limitations_zh="本阶段仅实现 importer，医疗工具和评分器尚未接入。不是官方 benchmark 成绩。不预计算答案或临床决策；后续 grader 适配须验证 FHIR 查询语义。",
            artifacts_en="patients.jsonl and cases.jsonl contain local source facts. snapshot/ retains original FHIR resources; evaluator/ contains unexecuted grader copies.",
            artifacts_zh="patients.jsonl、cases.jsonl 保存本地事实；snapshot/ 保留原始 FHIR 资源；evaluator/ 保存未执行的 grader 副本。",
        )
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("invalid converted dataset: " + "; ".join(errors))
        if out.exists():
            raise FileExistsError(f"output already exists: {out}")
        stage.rename(out)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local MedAgentBench repository")
    parser.add_argument("--out", type=Path, required=True, help="new AMA dataset directory")
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    connection.add_argument("--snapshot", type=Path, help="existing complete snapshot directory; no network")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60, help="seconds per FHIR request")
    args = parser.parse_args(argv)
    try:
        report = import_medagentbench(args.source, args.out, fhir_base=args.fhir_base,
                                     snapshot=args.snapshot, ids=args.ids, limit=args.limit, timeout=args.timeout)
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        print(f"import failed: {exc}")
        return 1
    print(json.dumps({key: report[key] for key in ("task_count", "patient_count", "observation_count", "per_task")},
                     ensure_ascii=False, indent=2))
    print(f"imported: {args.out}; tools and scorer are not yet implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
