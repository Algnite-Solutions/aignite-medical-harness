"""Import MedAgentBench data and expose offline clinical tools for AMA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

# Absolute imports also work when AMA loads this file through --tools.
from ama.data import validate_dataset
from ama.dataset_card import write_dataset_cards
from ama.tools import Tool, ToolResult

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


# Tool inputs describe the agent's decisions; no benchmark answers or clinical
# defaults belong here. Pydantic is already an AMA dependency.
_Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset or Z")
    return parsed


class _Code(_Input):
    system: _Text = Field(description="Coding system URI: NDC http://hl7.org/fhir/sid/ndc, "
                         "LOINC http://loinc.org, or SNOMED http://snomed.info/sct, as appropriate.")
    code: _Text = Field(description="Actual code supplied by the task context; no automatic code mapping.")
    display: _Text | None = None


class _Quantity(_Input):
    value: float
    unit: _Text


class _Order(_Input):
    patient_id: _Text
    code: _Code
    authored_at: _Text = Field(description="Time the order is signed, using the fixed task context and an ISO timestamp with timezone.")
    status: _Text = Field(description="MedAgentBench's order interface specifies active. Supply explicitly.")
    intent: _Text = Field(description="MedAgentBench's order interface specifies order. Supply explicitly.")

    @field_validator("authored_at")
    @classmethod
    def valid_authored_at(cls, value):
        _timestamp(value)
        return value


class _MedicationOrder(_Order):
    order_type: Literal["medication"]
    dose: _Quantity
    route: _Text = Field(description="Medication route as text, e.g. IV or oral, selected by the agent.")
    rate: _Quantity | None = Field(default=None, description="Original FHIR rateQuantity value and unit. "
                                  "The benchmark magnesium instructions express infusion time in h (hours). "
                                  "Supply when needed; it is not calculated from dose.")


class _ServiceOrder(_Order):
    order_type: Literal["test", "referral"]
    priority: _Text = Field(description="MedAgentBench's service interface specifies stat. Supply explicitly.")
    occurrence_at: _Text | None = Field(default=None, description="Time the test/service is to occur, "
                                       "as an ISO timestamp with timezone. Distinct from authored_at.")
    note: _Text | None = Field(default=None, description="Free text comment, e.g. the requested SBAR referral text.")

    @field_validator("occurrence_at")
    @classmethod
    def valid_occurrence_at(cls, value):
        if value is not None:
            _timestamp(value)
        return value


class _OrderArguments(_Input):
    order: Annotated[_MedicationOrder | _ServiceOrder, Field(discriminator="order_type")]


class _Observation(_Input):
    patient_id: _Text
    code: _Text
    value: str | float
    unit: _Text
    observed_at: _Text
    category: _Text
    status: _Text

    @field_validator("observed_at")
    @classmethod
    def valid_observed_at(cls, value):
        _timestamp(value)
        return value

    @field_validator("value")
    @classmethod
    def valid_value(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("observation value must not be blank")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("observation value must be finite")
        return value


class _ObservationArguments(_Input):
    observation: _Observation


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _page(rows: list[dict], offset: int, limit: int, key: str) -> dict:
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer between 1 and 200")
    stop = offset + limit
    return {key: rows[offset:stop], "total": len(rows), "offset": offset,
            "next_offset": stop if stop < len(rows) else None}


def _result(value: dict, evidence_ids=()) -> ToolResult:
    return ToolResult(json.dumps(value, ensure_ascii=False, allow_nan=False),
                      evidence_ids=list(evidence_ids))


def _tool_schema(model: type[BaseModel]) -> dict:
    """Expose nested types directly; some model tool templates ignore $ref."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            name = value["$ref"].removeprefix("#/$defs/")
            value = {**definitions[name], **{k: v for k, v in value.items() if k != "$ref"}}
        result = {k: expand(v) for k, v in value.items() if k != "discriminator"}
        if "oneOf" in result and all(branch.get("type") == "object" for branch in result["oneOf"]):
            result["type"] = "object"
        return result

    return expand(schema)


class _PatientStore:
    """Read only public facts. Index large cases lazily, without retaining all labs."""

    def __init__(self, patients_path: Path, cases_path: Path):
        self.patients = {row["patient_id"]: row for row in _read_jsonl(patients_path)}
        self.cases_path = cases_path
        self.offsets = None

    def patient(self, patient_id: str) -> dict:
        _text(patient_id, "patient_id")
        if patient_id not in self.patients:
            raise ValueError("unknown patient_id; use patient_search with the MRN identifier first")
        return {**self.patients[patient_id], "evidence_id": f"patient-{patient_id}"}

    @lru_cache(maxsize=8)
    def observations(self, patient_id: str) -> list[dict]:
        self.patient(patient_id)
        if self.offsets is None:
            offsets = {}
            with self.cases_path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if line.strip():
                        row = json.loads(line)
                        key = row["patient_id"]
                        if key in offsets:
                            raise ValueError(f"duplicate patient in cases.jsonl: {key}")
                        offsets[key] = offset
            self.offsets = offsets
        if patient_id not in self.offsets:
            raise ValueError(f"missing patient in cases.jsonl: {patient_id}")
        with self.cases_path.open("rb") as stream:
            stream.seek(self.offsets[patient_id])
            return json.loads(stream.readline())["observations"]


@lru_cache(maxsize=2)
def _patient_store(patients_path: Path, cases_path: Path, version: tuple) -> _PatientStore:
    # File metadata invalidates the cache if a local dataset is replaced.
    return _PatientStore(patients_path, cases_path)


def make_tools(dataset_dir: Path, episode_id: str) -> list[Tool]:
    """AMA's existing tool factory; each episode gets its own submission sequence."""
    if dataset_dir is None or episode_id is None:
        raise ValueError("MedAgentBench tools require dataset_dir and episode_id")
    _text(episode_id, "episode_id")
    folder = Path(dataset_dir).resolve()
    patients_path, cases_path = folder / "patients.jsonl", folder / "cases.jsonl"
    version = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (patients_path, cases_path))
    store = _patient_store(patients_path, cases_path, version)
    sequence = 0

    def patient_search(identifier=None, given=None, family=None, birth_date=None, offset=0, limit=50):
        filters = {k: _text(v, k) for k, v in {"identifier": identifier, "given": given,
                   "family": family, "birth_date": birth_date}.items() if v is not None}
        if not filters:
            raise ValueError("provide identifier, given, family, or birth_date")
        matches = []
        for patient in store.patients.values():
            if identifier is not None and not any(
                    item.get("value") == identifier for item in patient["identifiers"]):
                continue
            if birth_date is not None and patient.get("birth_date") != birth_date:
                continue
            # Given and family must match the same FHIR name entry, not aliases
            # combined across different entries. Name tokens are case insensitive.
            if given is not None or family is not None:
                if not any((given is None or given.casefold() in [v.casefold() for v in n.get("given", [])])
                           and (family is None or family.casefold() == n.get("family", "").casefold())
                           for n in patient["names"]):
                    continue
            matches.append(store.patient(patient["patient_id"]))
        page = _page(matches, offset, limit, "patients")
        return _result(page, [p["evidence_id"] for p in page["patients"]])

    def patient_information(patient_id):
        patient = store.patient(patient_id)
        return _result({"patient": patient}, [patient["evidence_id"]])

    def laboratory_results(patient_id, code=None, after=None, before=None, offset=0, limit=100):
        if code is not None:
            _text(code, "code")
        start = _timestamp(_text(after, "after")) if after is not None else None
        end = _timestamp(_text(before, "before")) if before is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("after must not be later than before")
        rows = []
        for row in store.observations(patient_id):
            if code is not None and code not in [c.get("code") for c in row["code"].get("coding", [])]:
                continue
            if start is not None or end is not None:
                if "observed_at" not in row:
                    continue
                observed = _timestamp(row["observed_at"])
                if (start is not None and observed < start) or (end is not None and observed > end):
                    continue
            rows.append(row)
        page = {"patient_id": patient_id, **_page(rows, offset, limit, "results")}
        return _result(page, [row["evidence_id"] for row in page["results"]])

    def receipt(operation: str, field: str, payload: dict) -> ToolResult:
        nonlocal sequence
        sequence += 1
        submission_id = f"submission-{episode_id}-{sequence}"
        return _result({"operation": operation, "status": "accepted", "simulation": True,
                        "episode_id": episode_id, "submission_id": submission_id,
                        field: payload}, [submission_id])

    def submit_order(order):
        _OrderArguments.model_validate({"order": order})
        return receipt("submit_order", "order", order)

    def record_observation(observation):
        _ObservationArguments.model_validate({"observation": observation})
        return receipt("record_observation", "observation", observation)

    def parameters(properties, required=()):
        return {"type": "object", "properties": properties, "required": list(required),
                "additionalProperties": False}

    string = {"type": "string", "minLength": 1}
    paging = {"offset": {"type": "integer", "minimum": 0, "default": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": 200}}
    return [
        Tool("patient_search", "Search local patients by exact identifier (MRN), birth_date (YYYY-MM-DD), "
             "or case-insensitive exact given name token and family name. All supplied filters must match. "
             "Returns source facts, not calculated ages; paginate using next_offset.",
             parameters({**{key: string for key in ("identifier", "given", "family", "birth_date")}, **paging}),
             patient_search),
        Tool("patient_information", "Read demographics using Patient.id returned by patient_search. "
             "An MRN is an identifier and need not equal Patient.id.",
             parameters({"patient_id": string}, ["patient_id"]), patient_information),
        Tool("laboratory_results", "Read local Observation facts with their original units, times and evidence IDs. "
             "code filters exact coding.code (e.g. MG, GLU, K, A1C); after/before are inclusive ISO timestamps "
             "with timezone. Timed filters exclude records without observed_at. Source order is preserved, "
             "not latest-first; use next_offset until null. No averages or treatment decisions are computed.",
             parameters({**{key: string for key in ("patient_id", "code", "after", "before")}, **paging},
                        ["patient_id"]), laboratory_results),
        Tool("submit_order", "Submit a structured simulated medication, test or referral order. "
             "Use Patient.id, explicit code system/code, times and clinical fields; none are filled in. "
             "Accepted means input format accepted, not clinically correct. The receipt is logged by AMA; "
             "it does not change patient records or create test results. Every call is a separate submission.",
             _tool_schema(_OrderArguments), submit_order),
        Tool("record_observation", "Submit a structured simulated observation, such as BP with value '118/77', "
             "unit 'mmHg', category 'vital-signs', status 'final' and an explicit observed_at timestamp. "
             "Accepted means input format accepted, not clinically correct. It is logged without changing "
             "the source facts. Supply all values yourself from the task.",
             _tool_schema(_ObservationArguments), record_observation),
    ]


def dataset_instructions() -> str:
    return (
        "Complete the requested patient lookup, calculation, or simulated clinical operation.\n"
        "Use only the task context and facts released by tools. Task times are fixed by the context.\n"
        "Resolve MRNs through patient_search(identifier=...) and use the returned patient_id in other tools.\n"
        "Read all relevant pages (next_offset=null means complete). Tools do not calculate task answers.\n"
        "Use an AMA answer object: mrn for the requested medical record number (read from identifiers),\n"
        "age for an age calculation,\n"
        "value for a laboratory value or average, and observed_at when its measurement time is requested.\n"
        "Put the task's patient-not-found text in mrn and its missing-measurement sentinel in value.\n"
        "For an operation, use submit_order or record_observation when indicated; describing a plan is not\n"
        "a submission. The final answer can contain completed and any requested value/observed_at.\n"
        "Submission receipts confirm format acceptance only, not clinical correctness or execution.\n"
        "Operations are simulated and logged; they do not change local patient records or generate results.\n"
    )


_EVAL_TIME = "2023-11-13T10:15:00+00:00"
_SBAR = ("Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
         "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
         "and provide management recommendations.")
_SUBMISSIONS = {"submit_order": ("order", _OrderArguments),
                "record_observation": ("observation", _ObservationArguments)}


def _logged_submissions(history: list[dict], episode_id: str) -> tuple[int, list[dict]]:
    """Pair actual assistant calls with tool receipts, never with prose or final answers."""
    if not isinstance(history, list):
        raise ValueError("episode history must be a list")
    pending, accepted, attempts = {}, [], 0
    for message in history:
        if not isinstance(message, dict):
            raise ValueError("history message must be an object")
        if message.get("role") == "assistant":
            if pending:
                raise ValueError("tool calls have no corresponding results")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("invalid logged tool call")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise ValueError("invalid or duplicate tool call ID")
                function = call["function"]
                name = function.get("name")
                if not isinstance(name, str):
                    raise ValueError("invalid logged tool name")
                attempts += name in _SUBMISSIONS
                pending[call_id] = call
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not pending or call_id != next(iter(pending)):
                raise ValueError("orphaned or out-of-order tool result")
            call = pending.pop(call_id)
            function = call["function"]
            name = function["name"]
            if name not in _SUBMISSIONS:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("submission result must contain text")
            body = json.loads(content.split("\nCiteable evidence IDs:", 1)[0])
            if isinstance(body, dict) and set(body) == {"error"}:
                continue  # Failed attempts are retained, but are not accepted orders.
            field, input_model = _SUBMISSIONS[name]
            args = json.loads(function.get("arguments", "{}"))
            input_model.model_validate(args)
            if (call.get("type") != "function" or not isinstance(body, dict)
                    or set(body) != {"operation", "status", "simulation", "episode_id", "submission_id", field}
                    or body["operation"] != name or body["status"] != "accepted"
                    or body["simulation"] is not True or body["episode_id"] != episode_id
                    or body["submission_id"] != f"submission-{episode_id}-{len(accepted) + 1}"
                    or body[field] != args[field]):
                raise ValueError("submission receipt does not match the logged call")
            accepted.append({"operation": name, **{field: args[field]}})
    if pending:
        raise ValueError("tool calls have no corresponding results")
    return attempts, accepted


def _number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reference(task: int, case: dict, store: _PatientStore) -> dict:
    """Compute evaluator-only truth using the same facts and rules as refsol.py."""
    if task == 1:
        sol = case.get("sol")
        if not isinstance(sol, list) or len(sol) != 1 or not isinstance(sol[0], str):
            raise ValueError("task1 requires one reference MRN or Patient not found string")
        return {"mrn": sol[0]}
    mrn = _text(case.get("eval_MRN"), "eval_MRN")
    patients = [p for p in store.patients.values()
                if any(item.get("value") == mrn for item in p["identifiers"])]
    if len(patients) != 1:
        raise ValueError(f"reference MRN must resolve to one patient: {mrn}")
    patient = patients[0]
    reference = {"patient_id": patient["patient_id"]}
    if task == 2:
        born = datetime.strptime(patient["birth_date"], "%Y-%m-%d")
        today = datetime(2023, 11, 13)  # Original grader's fixed date, not the wall clock.
        return {**reference, "age": today.year - born.year - ((today.month, today.day) < (born.month, born.day))}
    if task in (3, 8):
        return reference
    code = {4: "MG", 5: "MG", 6: "GLU", 7: "GLU", 9: "K", 10: "A1C"}[task]
    cutoff = _timestamp(_EVAL_TIME) - timedelta(hours=24) if task in (4, 5, 6) else None
    measurements = []
    for row in store.observations(patient["patient_id"]):
        if code not in [coding.get("code") for coding in row["code"].get("coding", [])]:
            continue
        if row.get("value_type") != "valueQuantity" or not _number(row.get("value")):
            raise ValueError(f"reference {code} observation has no finite quantity: {row.get('source_id')}")
        time = _timestamp(row["observed_at"])
        # refsol.py uses only a lower bound for its 24-hour window, no upper bound.
        if cutoff is None or time >= cutoff:
            measurements.append((time, row))
    reference["has_measurement"] = bool(measurements)
    if task == 6:
        reference["value"] = sum(row["value"] for _, row in measurements) / len(measurements) if measurements else -1
    elif measurements:
        # max returns the first item on a tie, preserving the original server order.
        time, row = max(measurements, key=lambda pair: pair[0])
        reference.update(value=row["value"], observed_at=row["observed_at"])
    else:
        reference["value"] = -1
    return reference


def _answer_correct(task: int, answer: dict, reference: dict) -> bool:
    if task in (3, 8) or (task == 5 and not reference["has_measurement"]):
        return True  # The original grader does not inspect FINISH here.
    if task == 1:
        return answer.get("mrn") == reference["mrn"]
    if task == 2:
        return _number(answer.get("age")) and answer["age"] == reference["age"]
    if task in (5, 9, 10) and "value" not in answer:
        return task != 10 or "observed_at" not in answer  # Original [] answer is permitted.
    given = answer.get("value")
    if not _number(given):
        return False
    if task == 6:
        return abs(given - reference["value"]) < 0.1
    if given != reference["value"]:
        return False
    return task != 10 or not reference["has_measurement"] or answer.get("observed_at") == reference["observed_at"]


def _order_matches(action: dict, reference: dict, kind: str, system: str, code: str) -> bool:
    order = action.get("order", {})
    return (action.get("operation") == "submit_order" and order.get("order_type") == kind
            and order.get("patient_id") == reference["patient_id"]
            and order.get("code", {}).get("system") == system and order.get("code", {}).get("code") == code
            and order.get("status") == "active" and order.get("intent") == "order")


def _actions_correct(task: int, attempts: int, actions: list[dict], reference: dict) -> bool:
    if task in (1, 2, 4, 6, 7):
        return attempts == 0
    if task == 3:
        if len(actions) != 1 or actions[0].get("operation") != "record_observation":
            return False
        row = actions[0]["observation"]
        return all(row.get(key) == value for key, value in {
            "patient_id": reference["patient_id"], "code": "BP", "value": "118/77", "unit": "mmHg",
            "observed_at": _EVAL_TIME, "category": "vital-signs", "status": "final"}.items())
    if task == 5:
        if not reference["has_measurement"] or reference["value"] > 1.9:
            return attempts == 0
        if len(actions) != 1 or not _order_matches(actions[0], reference, "medication", "http://hl7.org/fhir/sid/ndc", "0338-1715-40"):
            return False
        order = actions[0]["order"]
        dose = 4 if reference["value"] < 1 else 2 if reference["value"] < 1.5 else 1
        return ("2023-11-13T10:15" in order["authored_at"] and order["route"] == "IV"
                and order["dose"] == {"value": dose, "unit": "g"}
                and order.get("rate") == {"value": dose, "unit": "h"})
    if task == 8:
        if len(actions) != 1 or not _order_matches(actions[0], reference, "referral", "http://snomed.info/sct", "306181000000106"):
            return False
        order = actions[0]["order"]
        return order["authored_at"] == _EVAL_TIME and order["priority"] == "stat" and _SBAR in (order.get("note") or "")
    if task == 9:
        if not reference["has_measurement"] or reference["value"] >= 3.5:
            return attempts == 0
        if (len(actions) != 2
                or not _order_matches(actions[0], reference, "medication", "http://hl7.org/fhir/sid/ndc", "40032-917-01")
                or not _order_matches(actions[1], reference, "test", "http://loinc.org", "2823-3")):
            return False
        medication, test = actions[0]["order"], actions[1]["order"]
        dose = (3.5 - reference["value"]) / 0.1 * 10
        return ("2023-11-13T10:15" in medication["authored_at"] and medication["route"].strip().lower() == "oral"
                and abs(medication["dose"]["value"] - dose) <= 0.1 and medication["dose"]["unit"] == "mEq"
                and test["authored_at"] == _EVAL_TIME and test["priority"] == "stat"
                and "2023-11-14T08:" in (test.get("occurrence_at") or ""))
    if task == 10:
        needs_order = not reference["has_measurement"] or _timestamp(reference["observed_at"]) < _timestamp("2022-11-13T10:15:00+00:00")
        if not needs_order:
            return attempts == 0
        if len(actions) != 1 or not _order_matches(actions[0], reference, "test", "http://loinc.org", "4548-4"):
            return False
        order = actions[0]["order"]
        return order["authored_at"] == _EVAL_TIME and order["priority"] == "stat"
    raise ValueError(f"unsupported task: {task}")


def score_medagentbench(dataset, decisions: dict[str, list[dict]], *, histories=None) -> dict:
    """Native AMA scoring. Reference code is not imported; no server or model is called."""
    from ama.scorer import expected_rows

    if not isinstance(histories, dict):
        raise ValueError("MedAgentBench scoring requires messages.json histories; regenerate the run if missing")
    folder = Path(dataset.folder).resolve()
    patients_path, cases_path = folder / "patients.jsonl", folder / "cases.jsonl"
    version = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (patients_path, cases_path))
    store = _patient_store(patients_path, cases_path, version)
    per_episode, per_task = {}, {}
    counts = Counter()
    for episode in dataset.episodes:
        match = re.fullmatch(r"task(10|[1-9])_[1-9][0-9]*", episode.id)
        if not match or [turn.id for turn in episode.turns] != ["t1"]:
            raise ValueError(f"invalid MedAgentBench episode: {episode.id}")
        task = int(match.group(1))
        target = dataset.targets.get(episode.id)
        case = target.turns.get("t1", {}).get("case_data") if target else None
        if not isinstance(case, dict) or case.get("id") != episode.id:
            raise ValueError(f"missing or mismatched MedAgentBench target: {episode.id}")
        try:
            reference = _reference(task, case, store)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"cannot grade {episode.id}: {exc}") from exc
        row = expected_rows(episode, decisions)[0]
        answer = (row.get("decision") or {}).get("answer")
        submitted = row.get("decision") is not None and row["termination"] == "completed"
        answer_is_object = isinstance(answer, dict)
        attempts, actions, history_error = 0, [], None
        try:
            if episode.id not in histories:
                raise ValueError("missing episode history")
            attempts, actions = _logged_submissions(histories[episode.id], episode.id)
        except (ValueError, KeyError, TypeError) as exc:
            history_error = str(exc)
        answer_correct = submitted and answer_is_object and _answer_correct(task, answer, reference)
        actions_correct = history_error is None and _actions_correct(task, attempts, actions, reference)
        correct = answer_correct and actions_correct
        per_episode[episode.id] = {
            "task": f"task{task}", "submitted": submitted, "answer_is_object": answer_is_object,
            "answer_correct": answer_correct, "actions_correct": actions_correct, "correct": correct,
            "submission_attempts": attempts if history_error is None else None,
            "accepted_submissions": len(actions) if history_error is None else None,
            "history_valid": history_error is None, "history_error": history_error,
            "termination": row["termination"], "reference": reference}
        for key, value in {"correct": correct, "submitted": submitted, "answer_correct": answer_correct,
                           "actions_correct": actions_correct, "answer_objects": answer_is_object,
                           "history_valid": history_error is None}.items():
            counts[key] += int(value)
        group = per_task.setdefault(f"task{task}", {"num": 0, "den": 0})
        group["num"] += int(correct)
        group["den"] += 1
    total = len(dataset.episodes)
    rate = lambda n: {"num": n, "den": total, "value": n / total if total else None}
    for group in per_task.values():
        group["value"] = group["num"] / group["den"]
    return {"scorer": "medagentbench", "protocol": "ama-medagentbench-v1",
            "per_episode": per_episode, "aggregate": {
                "task_success": rate(counts["correct"]), "answer_accuracy": rate(counts["answer_correct"]),
                "action_accuracy": rate(counts["actions_correct"]), "completion": rate(counts["submitted"]),
                "answer_object_format": rate(counts["answer_objects"]), "valid_history": rate(counts["history_valid"]),
                "per_task": per_task}}


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
        (stage / "instructions.txt").write_text(dataset_instructions(), encoding="utf-8")
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
                  "anomalies": [], "tools_implemented": True, "scorer_implemented": True}
        _json(stage / "import_report.json", report)
        write_dataset_cards(
            stage, name=out.name, source="MedAgentBench v2; " + str(source),
            license_name="See copied source LICENSE and upstream data terms.",
            purpose_en="Local task and patient-data preparation for an AMA adaptation of MedAgentBench.",
            purpose_zh="为 MedAgentBench 的 AMA 本地适配准备任务、患者数据和评分快照。",
            construction_en="One episode per task. All patients and observations are exported independently of task selection. References are evaluator-only.",
            construction_zh="一个任务一个 Episode。患者和观察记录全量导出，不随任务筛选缩小；评分字段仅保存在 targets。",
            episodes=len(tasks), turns=len(tasks), evidence=sum(len(ep["turns"][0]["evidence"]) for ep in episodes),
            scorer="medagentbench", targets=len(tasks),
            limitations_en="Native AMA scoring uses structured answers and logged simulated actions with the original task rules. The simplified tool interface differs from the official FHIR interface; report scores as an AMA adaptation. Missing or invalid histories cannot prove successful operations. No answers or clinical recommendations are precomputed for the agent.",
            limitations_zh="原生 AMA 评分结合结构化答案、模拟提交日志和原任务规则。简化工具接口与官方 FHIR 接口不同，成绩应标注为 AMA 适配。缺失或无效日志不能证明操作成功。不向模型预提供答案或临床决策。",
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
    print(f"imported: {args.out}; offline tools and medagentbench scorer available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
