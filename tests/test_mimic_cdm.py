import csv
import json
from pathlib import Path

import pytest

from ama.data import load_dataset, validate_dataset
from ama.importers.mimic_cdm import LAB_COLUMNS, TABLES, import_mimic_cdm, score_mimic_cdm
from ama.runner import execute
from ama.tools import load_tools
from fakes import FakeModel, call, calls, decision


def _csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _source(root):
    root.mkdir()
    rows = {
        "history_of_present_illness": [
            {"hadm_id": "1", "hpi": "right lower quadrant pain"},
            {"hadm_id": "2", "hpi": "right upper quadrant pain"}],
        "physical_examination": [
            {"hadm_id": "1", "pe": "tenderness A"},
            {"hadm_id": "2", "pe": "tenderness B"}],
        "laboratory_tests": [
            {"hadm_id": "1", "itemid": "100", "valuestr": "12", "ref_range_lower": "4",
             "ref_range_upper": "10"},
            {"hadm_id": "2", "itemid": "100", "valuestr": "19", "ref_range_lower": "4",
             "ref_range_upper": "10"}],
        "microbiology": [{"hadm_id": "1", "test_itemid": "9", "valuestr": "none",
                          "spec_itemid": "7"}],
        "radiology_reports": [
            {"hadm_id": "1", "note_id": "n1", "modality": "CT", "region": "Abdomen",
             "exam_name": "CT abdomen", "text": "finding A"},
            {"hadm_id": "2", "note_id": "n2", "modality": "CT", "region": "Abdomen",
             "exam_name": "CT abdomen", "text": "finding B"}],
        "discharge_diagnosis": [
            {"hadm_id": "1", "discharge_diagnosis": "appendicitis"},
            {"hadm_id": "2", "discharge_diagnosis": "cholecystitis"}],
        "discharge_procedures": [{"hadm_id": "1", "discharge_procedure": "appendectomy"}],
        "icd_diagnosis": [{"hadm_id": "1", "icd_diagnosis": "appendicitis"},
                          {"hadm_id": "2", "icd_diagnosis": "cholecystitis"}],
        "icd_procedures": [{"hadm_id": "1", "icd_code": "47", "icd_title": "appendectomy",
                            "icd_version": "9"}],
    }
    for name, columns in TABLES.items():
        _csv(root / f"{name}.csv", columns, rows[name])
    _csv(root / "lab_test_mapping.csv", LAB_COLUMNS, [
        {"itemid": "100", "label": "White blood cells", "fluid": "Blood",
         "category": "Hematology", "count": "2", "corresponding_ids": "[100]"}])
    (root / "pathology_ids.json").write_text(json.dumps({
        "appendicitis": [1], "cholecystitis": [2], "diverticulitis": [], "pancreatitis": []}))
    (root / "LICENSE.txt").write_text("credentialed test fixture")
    return root


def test_two_datasets_are_valid_and_targets_hidden(tmp_path):
    source = _source(tmp_path / "source")
    out = tmp_path / "processed"
    report = import_mimic_cdm(source, out)
    assert report["admissions"] == 2
    for name in ("mimic_cdm_interactive", "mimic_cdm_full_info"):
        folder = out / name
        assert validate_dataset(folder) == []
        ds = load_dataset(folder, with_targets=True)
        assert len(ds.episodes) == 2 and len(ds.targets) == 2
        assert "appendectomy" not in (folder / "episodes.jsonl").read_text()
        assert "appendectomy" not in (folder / "cases.jsonl").read_text()
    interactive = load_dataset(out / "mimic_cdm_interactive")
    assert [e.id for e in interactive.episodes[0].turns[0].evidence] == ["hpi"]
    full = load_dataset(out / "mimic_cdm_full_info")
    assert len(full.episodes[0].turns[0].evidence) == 5
    with pytest.raises(FileExistsError):
        import_mimic_cdm(source, out)


def test_episode_tools_are_bound_and_tool_data_hashed(tmp_path):
    out = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), out)
    folder = out / "mimic_cdm_interactive"
    tools_file = Path("src/ama/importers/mimic_cdm_tools.py")
    tools1 = {t.name: t for t in load_tools(tools_file, dataset_dir=folder, episode_id="1")}
    tools2 = {t.name: t for t in load_tools(tools_file, dataset_dir=folder, episode_id="2")}
    assert set(tools1) == {"physical_examination", "laboratory_results", "microbiology",
                           "list_imaging", "imaging"}
    assert "tenderness A" in tools1["physical_examination"].invoke("{}").text
    assert tools1["physical_examination"].invoke("{}").evidence_ids == ["physical-examination"]
    assert "tenderness B" in tools2["physical_examination"].invoke("{}").text
    labs = json.loads(tools1["laboratory_results"].invoke("{}").text)["results"]
    assert tools1["laboratory_results"].invoke("{}").evidence_ids == ["laboratory-tests", "lab-100"]
    assert [(row["label"], row["value"], row["fluid"]) for row in labs] == [
        ("White blood cells", "12", "Blood")]
    catalog = json.loads(tools1["list_imaging"].invoke("{}").text)["reports"]
    assert catalog == [{"report_id": "imaging-1", "modality": "CT", "region": "Abdomen",
                        "exam_name": "CT abdomen"}]
    assert "finding" not in json.dumps(catalog)
    assert tools1["list_imaging"].invoke("{}").evidence_ids == []
    report = json.loads(tools1["imaging"].invoke('{"report_id":"imaging-1"}').text)
    assert report["evidence_id"] == "imaging-1" and "finding A" in report["text"]
    assert tools1["imaging"].invoke('{"report_id":"imaging-1"}').evidence_ids == ["imaging-1"]
    assert "finding B" not in tools1["imaging"].invoke('{"report_id":"imaging-1"}').text
    with pytest.raises(ValueError, match="unknown imaging report ID"):
        tools1["imaging"].invoke('{"report_id":"imaging-2"}')

    model = FakeModel(calls(call("physical_examination", "{}")),
                      decision(answer={"diagnosis": "appendicitis", "treatment_plan": "surgery"},
                               citations=["hpi"]))
    run = execute(folder, model, episode_ids=["1"], tools_path=tools_file,
                  runs_root=tmp_path / "runs")
    manifest = json.loads((run / "manifest.json").read_text())
    assert "dataset_sha256" not in manifest
    assert manifest["tools"] == str(tools_file.resolve())
    assert json.loads((run / "decisions.jsonl").read_text())["decision"]["answer"]["diagnosis"] \
        == "appendicitis"
    dataset = load_dataset(folder, with_targets=True)
    dataset.episodes = dataset.select(episode_id="1")
    rows = {"1": [json.loads((run / "decisions.jsonl").read_text())]}
    assert score_mimic_cdm(dataset, rows)["aggregate"]["diagnosis_accuracy"]["value"] == 1
    rows["1"][0]["decision"]["answer"] = "appendicitis"
    aggregate = score_mimic_cdm(dataset, rows)["aggregate"]
    assert aggregate["diagnosis_accuracy"]["value"] == 1
    assert aggregate["answer_object_format"]["value"] == 0


def test_checksum_manifest_is_not_required(tmp_path):
    source = _source(tmp_path / "source")
    _csv(source / "physical_examination.csv", TABLES["physical_examination"], [
        {"hadm_id": "1", "pe": "updated exam A"},
        {"hadm_id": "2", "pe": "updated exam B"}])
    import_mimic_cdm(source, tmp_path / "processed")
    assert "updated exam A" in (tmp_path / "processed" / "mimic_cdm_full_info"
                                / "episodes.jsonl").read_text()


def test_malformed_source_still_fails_before_output(tmp_path):
    source = _source(tmp_path / "source")
    (source / "physical_examination.csv").write_text("wrong,columns\n1,exam\n")
    with pytest.raises(ValueError, match="expected columns"):
        import_mimic_cdm(source, tmp_path / "processed")
    assert not (tmp_path / "processed").exists()
