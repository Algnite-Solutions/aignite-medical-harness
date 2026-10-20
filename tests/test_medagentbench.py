import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ama.data import load_dataset, validate_dataset
from ama.importers import medagentbench as med
from ama.runner import execute
from ama.tools import load_tools
from fakes import FakeModel, call, calls, decision


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    data = root / "data/medagentbench"
    grader = root / "src/server/tasks/medagentbench"
    data.mkdir(parents=True)
    grader.mkdir(parents=True)
    tasks = [{"id": f"task{task}_{sample}", "instruction": "Inspect patient P1.",
              "context": "It's 2023-11-13T10:15:00+00:00 now." if task != 1 else "",
              "eval_MRN": "PRIVATE_MRN", **({"sol": ["PRIVATE_ANSWER"]} if task == 1 else {})}
             for task in range(1, 11) for sample in range(1, 31)]
    med._json(data / "test_data_v2.json", tasks)
    med._json(data / "funcs_v1.json", [{"name": "GET Patient"}])
    # Copying must never import or execute reference code.
    (grader / "refsol.py").write_text("raise RuntimeError('must not execute grader')\n")
    (grader / "utils.py").write_text("raise RuntimeError('must not execute utils')\n")
    (root / "LICENSE").write_text("fixture license")
    return root


@pytest.fixture
def resources():
    return {
        "Patient": [
            {"resourceType": "Patient", "id": "P1", "identifier": [{"system": "mrn", "value": "M1"}],
             "name": [{"family": "Example", "given": ["Alice"]}], "birthDate": "1980-02-29"},
            {"resourceType": "Patient", "id": "P2", "identifier": [{"value": "M2"}]},
        ],
        "Observation": [
            {"resourceType": "Observation", "id": "O1", "subject": {"reference": "Patient/P1"},
             "code": {"coding": [{"system": "lab", "code": "GLU", "display": "Glucose"}]},
             "effectiveDateTime": "2023-11-13T10:15:00.123+08:00", "status": "final",
             "valueQuantity": {"value": 4.25, "unit": "mmol/L", "system": "ucum", "code": "mmol/L",
                               "comparator": "<"},
             "referenceRange": [{"low": {"value": 3.9}, "high": {"value": 6.1}}]},
            {"resourceType": "Observation", "id": "O2",
             "subject": {"reference": "http://example/fhir/Patient/P1/_history/1"},
             "code": {"text": "Glucose", "coding": [{"code": "GLU"}]},
             "effectiveDateTime": "2023-11-12T09:00:00+00:00", "valueQuantity": {"value": 93, "unit": "mg/dL"}},
            {"resourceType": "Observation", "id": "O3", "subject": {"reference": "Patient/P1"},
             "code": {"text": "BP"}, "valueString": "118/77 mmHg"},
        ],
    }


def write_snapshot(folder, resources):
    folder.mkdir()
    stats = {}
    for kind, rows in resources.items():
        path = folder / f"{kind}.jsonl"
        med._jsonl(path, rows)
        stats[kind] = {"count": len(rows), "total": len(rows), "pages": 1,
                       "received": len(rows), "sha256": med._sha(path)}
    med._json(folder / "metadata.json", {"schema": med.SNAPSHOT_SCHEMA, "complete": True,
              "fhir_base": "http://example/fhir/", "exported_at": "2026-10-05T00:00:00+00:00",
              "resources": stats})
    return folder


@pytest.fixture
def snapshot(tmp_path, resources):
    return write_snapshot(tmp_path / "snapshot", resources)


def test_offline_import_structure_and_isolation(source, snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(med, "_get_json", lambda *a: pytest.fail("offline import used the network"))
    out = tmp_path / "out"
    report = med.import_medagentbench(source, out, snapshot=snapshot)
    assert validate_dataset(out) == []
    dataset = load_dataset(out, with_targets=True)
    assert len(dataset.episodes) == len(dataset.targets) == 300
    assert dataset.info.splits["test"] == [episode.id for episode in dataset.episodes]
    assert dataset.info.splits["smoke"] == list(med.SMOKE_IDS)
    assert len(dataset.info.splits["smoke"]) == 30
    assert report["per_task"] == {f"task{i}": 30 for i in range(1, 11)}
    assert report["patient_count"] == 2 and report["observation_count"] == 3
    assert dataset.episodes[0].turns[0].evidence == []
    assert dataset.episodes[30].turns[0].evidence[0].id == "context"
    for name in ("episodes.jsonl", "instructions.txt", "patients.jsonl", "cases.jsonl"):
        text = (out / name).read_text()
        assert "PRIVATE_MRN" not in text and "PRIVATE_ANSWER" not in text
        assert "eval_MRN" not in text and '"sol"' not in text
    assert dataset.targets["task1_1"].turns["t1"]["case_data"]["sol"] == ["PRIVATE_ANSWER"]
    for name in ("refsol.py", "utils.py"):
        assert (out / "evaluator" / name).read_bytes() == (source / "src/server/tasks/medagentbench" / name).read_bytes()
    assert (out / "DATASET_CARD.zh-CN.md").is_file()
    assert report["scorer_implemented"] and report["tools_implemented"]
    assert "AMA answer object" in (out / "instructions.txt").read_text()
    assert "FINISH" not in (out / "instructions.txt").read_text()


def test_normalization_preserves_source_facts(resources):
    patients, cases = med.convert_cases(resources)
    assert patients[0]["identifiers"] == resources["Patient"][0]["identifier"]
    assert patients[0]["names"] == resources["Patient"][0]["name"]
    assert patients[0]["birth_date"] == "1980-02-29"
    assert "birth_date" not in patients[1]
    observations = cases[0]["observations"]
    assert len(observations) == 3  # Repeated GLU tests remain separate.
    assert observations[0]["value"] == 4.25 and observations[0]["unit"] == "mmol/L"
    assert observations[0]["observed_at"] == "2023-11-13T10:15:00.123+08:00"
    assert observations[0]["unit_system"] == "ucum" and observations[0]["comparator"] == "<"
    assert observations[0]["reference_ranges"] == resources["Observation"][0]["referenceRange"]
    assert observations[1]["value"] == 93 and observations[1]["unit"] == "mg/dL"
    assert observations[2]["value"] == "118/77 mmHg"
    assert "observed_at" not in observations[2]
    assert len({row["evidence_id"] for row in observations}) == 3
    assert cases[1] == {"patient_id": "P2", "observations": []}


def test_equal_time_measurements_keep_server_order(resources):
    # Original graders pick the first encountered value when timestamps tie.
    first, second = resources["Observation"][:2]
    second["effectiveDateTime"] = first["effectiveDateTime"]
    resources["Observation"] = [second, first]
    resources["Observation"] = med._unique_resources(resources["Observation"], "Observation")
    _, cases = med.convert_cases(resources)
    assert [row["value"] for row in cases[0]["observations"]] == [93, 4.25]


def test_absent_measurement_value_is_not_invented(resources):
    resource = resources["Observation"][0]
    resource["valueQuantity"] = {"unit": "mmol/L"}
    resource["dataAbsentReason"] = {"text": "not available"}
    _, cases = med.convert_cases(resources)
    record = cases[0]["observations"][0]
    assert "value" not in record
    assert record["unit"] == "mmol/L" and record["dataAbsentReason"] == resource["dataAbsentReason"]


def test_subset_never_shrinks_patient_data(source, snapshot, tmp_path):
    out = tmp_path / "subset"
    report = med.import_medagentbench(source, out, snapshot=snapshot, ids=["task2_8", "task1_4"], limit=1)
    assert report["selected_ids"] == ["task1_4"]  # Original task order.
    assert report["smoke_ids"] == ["task1_4"]
    assert len(med._read_jsonl(out / "patients.jsonl")) == 2
    assert len(med._read_jsonl(out / "snapshot/Observation.jsonl")) == 3


def test_snapshot_reimport_is_deterministic(source, snapshot, tmp_path):
    first, second = tmp_path / "first/dataset", tmp_path / "second/dataset"
    med.import_medagentbench(source, first, snapshot=snapshot, limit=2)
    med.import_medagentbench(source, second, snapshot=first / "snapshot", limit=2)
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()


def test_no_overwrite(source, snapshot, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        med.import_medagentbench(source, out, snapshot=snapshot)
    assert (out / "keep").read_text() == "original"


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": -1}, {"ids": []},
                                    {"ids": ["unknown"]}, {"timeout": 0}])
def test_invalid_arguments_leave_no_output(source, snapshot, tmp_path, options):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        med.import_medagentbench(source, out, snapshot=snapshot, **options)
    assert not out.exists()


@pytest.mark.parametrize("broken", ["checksum", "count", "complete", "reference"])
def test_broken_snapshot_not_published(source, snapshot, tmp_path, broken):
    if broken == "checksum":
        with (snapshot / "Patient.jsonl").open("a") as stream:
            stream.write("\n")
    elif broken == "reference":
        rows = med._read_jsonl(snapshot / "Observation.jsonl")
        rows[0]["subject"]["reference"] = "Patient/unknown"
        med._jsonl(snapshot / "Observation.jsonl", rows)
    metadata = json.loads((snapshot / "metadata.json").read_text())
    if broken == "count":
        metadata["resources"]["Patient"]["count"] = 999
    if broken == "complete":
        metadata["complete"] = False
    if broken == "reference":
        metadata["resources"]["Observation"]["sha256"] = med._sha(snapshot / "Observation.jsonl")
    med._json(snapshot / "metadata.json", metadata)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        med.import_medagentbench(source, out, snapshot=snapshot)
    assert not out.exists()
    assert not list(tmp_path.glob(".out-*"))


def bundle(rows, total, next_url=None):
    return {"resourceType": "Bundle", "type": "searchset", "total": total,
            "entry": [{"resource": row} for row in rows],
            "link": [{"relation": "next", "url": next_url}] if next_url else []}


def test_live_export_paginates_and_is_reusable(source, resources, tmp_path, monkeypatch):
    calls = []
    def get(url, timeout):
        calls.append(url)
        if urlsplit(url).path.endswith("Patient"):
            if "page=2" in url:
                return bundle(resources["Patient"][1:], 2)
            return bundle(resources["Patient"][:1], 2, "/fhir/Patient?page=2")
        return bundle(resources["Observation"], 3)
    monkeypatch.setattr(med, "_get_json", get)
    out = tmp_path / "live"
    report = med.import_medagentbench(source, out, fhir_base="http://example/fhir", limit=1)
    assert len(calls) == 3
    assert report["snapshot_resources"]["Patient"]["pages"] == 2
    assert med._load_snapshot(out / "snapshot")[1] == resources
    monkeypatch.setattr(med, "_get_json", lambda *a: pytest.fail("unexpected network"))
    assert med.import_medagentbench(source, tmp_path / "offline", snapshot=out / "snapshot", limit=1)["patient_count"] == 2


@pytest.mark.parametrize("failure", ["http", "loop", "incomplete", "conflict", "origin", "wrong_type"])
def test_export_failure_cleans_output(source, resources, tmp_path, monkeypatch, failure):
    def get(url, timeout):
        if failure == "http":
            raise OSError("connection unavailable")
        if failure == "loop":
            return bundle(resources["Patient"], 2, url)
        if failure == "incomplete":
            return bundle(resources["Patient"][:1], 2)
        if failure == "conflict":
            return bundle([resources["Patient"][0], {**resources["Patient"][0], "birthDate": "1900-01-01"}], 1)
        if failure == "origin":
            return bundle(resources["Patient"], 2, "http://other/fhir/Patient")
        return {"resourceType": "OperationOutcome"}
    monkeypatch.setattr(med, "_get_json", get)
    out = tmp_path / "out"
    with pytest.raises((ValueError, OSError)):
        med.import_medagentbench(source, out, fhir_base="http://example/fhir/")
    assert not out.exists() and not list(tmp_path.glob(".out-*"))


def test_empty_observations_are_valid(source, resources, tmp_path, monkeypatch):
    resources["Observation"] = []
    snapshot = write_snapshot(tmp_path / "snapshot", resources)
    report = med.import_medagentbench(source, tmp_path / "out", snapshot=snapshot, limit=1)
    assert report["observation_count"] == 0
    assert all(not row["observations"] for row in med._read_jsonl(tmp_path / "out/cases.jsonl"))


def test_source_validation(source, snapshot, tmp_path):
    path = source / "data/medagentbench/test_data_v2.json"
    tasks = json.loads(path.read_text())
    med._json(path, tasks + tasks[:1])
    with pytest.raises(ValueError, match="duplicate"):
        med.import_medagentbench(source, tmp_path / "out", snapshot=snapshot)
    med._json(path, tasks)
    (source / "src/server/tasks/medagentbench/refsol.py").unlink()
    with pytest.raises(FileNotFoundError, match="refsol.py"):
        med.import_medagentbench(source, tmp_path / "out", snapshot=snapshot)


def test_cli_success_and_failure(source, snapshot, tmp_path, capsys):
    out = tmp_path / "cli"
    args = ["--source", str(source), "--snapshot", str(snapshot), "--out", str(out), "--limit", "1"]
    assert med.main(args) == 0
    assert "scorer available" in capsys.readouterr().out
    assert med.main(args) == 1
    assert "already exists" in capsys.readouterr().out


@pytest.fixture
def tool_dataset(source, snapshot, tmp_path):
    out = tmp_path / "dataset"
    med.import_medagentbench(source, out, snapshot=snapshot, ids=["task3_1", "task9_1"])
    return out


TOOLS_FILE = Path("src/ama/importers/medagentbench.py")


def episode_tools(folder, episode="task3_1"):
    return {t.name: t for t in load_tools(TOOLS_FILE, dataset_dir=folder, episode_id=episode)}


def invoke(tool, **arguments):
    return tool.invoke(json.dumps(arguments))


def medication(**fields):
    return {"order_type": "medication", "patient_id": "P1",
            "code": {"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"},
            "authored_at": "2023-11-13T10:15:00+00:00", "status": "active", "intent": "order",
            "dose": {"value": 20, "unit": "mEq"}, "route": "oral", **fields}


def service(kind="test", **fields):
    return {"order_type": kind, "patient_id": "P1",
            "code": {"system": "http://loinc.org", "code": "2823-3"},
            "authored_at": "2023-11-13T10:15:00+00:00", "status": "active", "intent": "order",
            "priority": "stat", **fields}


def observation(**fields):
    return {"patient_id": "P1", "code": "BP", "value": "118/77", "unit": "mmHg",
            "observed_at": "2023-11-13T10:15:00+00:00", "category": "vital-signs",
            "status": "final", **fields}


def test_tools_load_through_existing_file_interface_without_private_data(tool_dataset, monkeypatch):
    original = Path.open
    def public_only(path, *args, **kwargs):
        assert path.name not in {"targets.jsonl", "eval.json", "refsol.py", "utils.py", "funcs_v1.json"}
        assert "snapshot" not in path.parts and "evaluator" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_only)
    monkeypatch.setattr(med.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    monkeypatch.setattr(med.urllib.request.OpenerDirector, "open", lambda *a, **k: pytest.fail("unexpected network"))
    tools = episode_tools(tool_dataset)
    assert set(tools) == {"patient_search", "patient_information", "laboratory_results",
                          "submit_order", "record_observation"}
    for tool in tools.values():
        assert tool.definition()["function"]["parameters"]["additionalProperties"] is False
        assert '"$ref"' not in json.dumps(tool.parameters)
    assert tools["record_observation"].parameters["properties"]["observation"]["type"] == "object"
    assert tools["submit_order"].parameters["properties"]["order"]["type"] == "object"
    assert tools["record_observation"].parameters["properties"]["observation"]["properties"]["patient_id"]["type"] == "string"
    assert json.loads(invoke(tools["patient_search"], identifier="M1").text)["patients"][0]["patient_id"] == "P1"
    assert json.loads(invoke(tools["laboratory_results"], patient_id="P1", code="GLU").text)["total"] == 2
    assert json.loads(invoke(tools["submit_order"], order=medication()).text)["status"] == "accepted"
    assert json.loads(invoke(tools["record_observation"], observation=observation()).text)["status"] == "accepted"
    with pytest.raises(ValueError, match="require"):
        med.make_tools(tool_dataset, None)


def test_patient_search_uses_source_identifiers_and_name_filters(tool_dataset):
    rows = med._read_jsonl(tool_dataset / "patients.jsonl")
    rows[0]["names"].append({"given": ["Jane"], "family": "Alias"})
    rows.append({**rows[0], "patient_id": "P3"})  # Ambiguity must not pick an answer.
    med._jsonl(tool_dataset / "patients.jsonl", rows)
    tools = episode_tools(tool_dataset)
    result = invoke(tools["patient_search"], identifier="M1", limit=1)
    page = json.loads(result.text)
    assert page["total"] == 2 and page["next_offset"] == 1
    assert result.evidence_ids == ["patient-P1"]
    assert "age" not in page["patients"][0]
    result = invoke(tools["patient_search"], given="alice", family="EXAMPLE", birth_date="1980-02-29", offset=1)
    assert [p["patient_id"] for p in json.loads(result.text)["patients"]] == ["P3"]
    assert result.evidence_ids == ["patient-P3"]
    assert json.loads(invoke(tools["patient_search"], given="Alice", family="Alias").text)["patients"] == []
    assert invoke(tools["patient_search"], identifier="unknown").evidence_ids == []
    result = invoke(tools["patient_information"], patient_id="P2")
    assert "birth_date" not in json.loads(result.text)["patient"]
    assert result.evidence_ids == ["patient-P2"]
    with pytest.raises(ValueError, match="unknown patient_id"):
        invoke(tools["patient_information"], patient_id="M1")
    for args in ({}, {"identifier": 1}, {"identifier": " "}, {"identifier": "M1", "limit": True}):
        with pytest.raises(ValueError):
            invoke(tools["patient_search"], **args)


def test_laboratory_pages_preserve_units_times_missing_values_and_evidence(tool_dataset):
    cases = med._read_jsonl(tool_dataset / "cases.jsonl")
    missing = {**cases[0]["observations"][0], "source_id": "O4", "evidence_id": "observation-O4"}
    missing.pop("value")
    cases[0]["observations"].append(missing)
    med._jsonl(tool_dataset / "cases.jsonl", cases)
    tool = episode_tools(tool_dataset)["laboratory_results"]
    first = invoke(tool, patient_id="P1", code="GLU", limit=1)
    page = json.loads(first.text)
    assert page["total"] == 3 and page["next_offset"] == 1
    assert first.evidence_ids == ["observation-O1"]
    row = page["results"][0]
    assert row["value"] == 4.25 and row["unit"] == "mmol/L" and row["comparator"] == "<"
    assert row["observed_at"] == "2023-11-13T10:15:00.123+08:00"
    last = invoke(tool, patient_id="P1", code="GLU", offset=1)
    page = json.loads(last.text)
    assert page["next_offset"] is None
    assert page["results"][0]["value"] == 93 and page["results"][0]["unit"] == "mg/dL"
    assert "value" not in page["results"][1]
    assert last.evidence_ids == ["observation-O2", "observation-O4"]
    assert invoke(tool, patient_id="P2").evidence_ids == []
    assert invoke(tool, patient_id="P1", code="MISSING").evidence_ids == []
    # Source order is unchanged even if a time filter is supplied. Boundaries are inclusive.
    filtered = invoke(tool, patient_id="P1", code="GLU", after="2023-11-13T02:15:00.123Z",
                      before="2023-11-13T02:15:00.123+00:00")
    assert filtered.evidence_ids == ["observation-O1", "observation-O4"]
    filtered = invoke(tool, patient_id="P1", after="2023-11-12T00:00:00Z")
    assert "observation-O3" not in filtered.evidence_ids  # Untimed BP stays absent from timed queries.
    for args in ({"after": "2023-11-13"}, {"after": "2023-11-14T00:00Z", "before": "2023-11-13T00:00Z"},
                 {"limit": 201}, {"offset": -1}, {"code": []}):
        with pytest.raises(ValueError):
            invoke(tool, patient_id="P1", **args)


def test_submissions_preserve_wrong_clinical_decisions_without_changing_facts(tool_dataset):
    before = {name: med._sha(tool_dataset / name) for name in ("patients.jsonl", "cases.jsonl")}
    tools = episode_tools(tool_dataset)
    # Incorrect patient, code, dose and time are decisions for the future scorer,
    # not reasons for the tool to repair the model's output.
    wrong = medication(patient_id="wrong-patient", dose={"value": -999, "unit": "wrong-unit"},
                       code={"system": "wrong-system", "code": "wrong-code"}, route="wrong-route",
                       authored_at="2000-01-01T00:00:00Z")
    first = json.loads(invoke(tools["submit_order"], order=wrong).text)
    assert first["order"] == wrong and first["simulation"] is True
    assert first["submission_id"] == "submission-task3_1-1"
    assert "rate" not in first["order"]  # No hidden clinical defaults.
    second = json.loads(invoke(tools["submit_order"], order=wrong).text)
    assert second["submission_id"] == "submission-task3_1-2"  # Duplicate submits remain distinct.
    for kind in ("test", "referral"):
        data = service(kind, occurrence_at="2023-11-14T08:00:00Z", note="SBAR text")
        assert json.loads(invoke(tools["submit_order"], order=data).text)["order"] == data
    result = invoke(tools["record_observation"], observation=observation())
    assert json.loads(result.text)["observation"] == observation()
    assert result.evidence_ids == ["submission-task3_1-5"]
    other = episode_tools(tool_dataset, "task9_1")
    assert json.loads(invoke(other["submit_order"], order=medication()).text)["submission_id"] == "submission-task9_1-1"
    restart = episode_tools(tool_dataset)
    assert json.loads(invoke(restart["submit_order"], order=medication()).text)["submission_id"] == "submission-task3_1-1"
    assert before == {name: med._sha(tool_dataset / name) for name in before}
    assert json.loads(invoke(tools["laboratory_results"], patient_id="P1").text)["total"] == 3


def test_malformed_submissions_are_rejected_without_receipts(tool_dataset):
    tools = episode_tools(tool_dataset)
    missing = medication()
    del missing["route"]
    invalid = [missing, medication(dose={"value": True, "unit": "g"}), medication(extra="unexpected"),
               medication(authored_at="2023-11-13T10:15:00"), medication(dose={"value": float("nan"), "unit": "g"}),
               medication(order_type="unsupported"), medication(patient_id=" "), service(occurrence_at="bad"),
               service(authored_at="2023-11-13T10:15:00"), service(authored_at="bad")]
    for order in invalid:
        with pytest.raises(ValueError):
            invoke(tools["submit_order"], order=order)
    for row in (observation(observed_at="bad"), observation(value=True), observation(value=""), observation(unit=None)):
        with pytest.raises(ValueError):
            invoke(tools["record_observation"], observation=row)
    assert json.loads(invoke(tools["submit_order"], order=medication()).text)["submission_id"] == "submission-task3_1-1"


def test_runner_persists_calls_errors_receipts_in_order_and_native_answer(tool_dataset, tmp_path):
    order = medication()
    followup = service(occurrence_at="2023-11-14T08:00:00+00:00")
    bad = {**order, "authored_at": "no-time"}
    model = FakeModel(calls(call("submit_order", json.dumps({"order": bad}), id="bad")),
                      calls(call("submit_order", json.dumps({"order": order}), id="med"),
                            call("submit_order", json.dumps({"order": followup}), id="lab")),
                      decision(answer={"completed": True, "value": 3.3},
                               citations=["submission-task9_1-1", "submission-task9_1-2"]))
    run = execute(tool_dataset, model, episode_ids=["task9_1"], tools_path=TOOLS_FILE,
                  runs_root=tmp_path / "runs")
    history = json.loads((run / "messages.json").read_text())["task9_1"]
    results = [row for row in history if row["role"] == "tool"]
    assert [row["tool_call_id"] for row in results] == ["bad", "med", "lab"]
    assert "error" in json.loads(results[0]["content"])
    receipts = [json.loads(row["content"].split("\nCiteable evidence IDs:")[0]) for row in results[1:]]
    assert [row["order"] for row in receipts] == [order, followup]
    assert [row["submission_id"] for row in receipts] == ["submission-task9_1-1", "submission-task9_1-2"]
    output = json.loads((run / "decisions.jsonl").read_text())
    assert output["termination"] == "completed"
    assert output["decision"]["answer"] == {"completed": True, "value": 3.3}
    assert "PRIVATE_MRN" not in json.dumps(model.requests) and "PRIVATE_ANSWER" not in json.dumps(model.requests)
    assert json.loads((run / "manifest.json").read_text())["tools"] == str(TOOLS_FILE.resolve())
