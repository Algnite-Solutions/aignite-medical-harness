import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ama.data import load_dataset, validate_dataset
from ama.importers import medagentbench as med


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
    assert not report["scorer_implemented"] and not report["tools_implemented"]


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
    assert "not yet implemented" in capsys.readouterr().out
    assert med.main(args) == 1
    assert "already exists" in capsys.readouterr().out
