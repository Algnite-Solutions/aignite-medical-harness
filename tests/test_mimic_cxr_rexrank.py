import json
import gzip
import csv

import pytest
from PIL import Image

from ama.data import load_dataset, validate_dataset
from ama.importers import mimic_cxr_rexrank as rex


def records(count=102):
    return {str(s): {"subject_id": 10000001, "split": "test",
                    "image_path": [f"/p10/p10000001/s{s}/img-{s}.jpg"],
                    "key_image_path": f"/p10/p10000001/s{s}/img-{s}.jpg",
                    "frontal_lateral": ["AP"], "section_findings": "SYNTHETIC FINDINGS",
                    "section_impression": float("nan"), "report": "SYNTHETIC OFFICIAL REPORT"}
            for s in range(1, count + 1)}


def test_selection_stable_under_source_order_and_path_validation():
    data = records()
    assert rex.select_records(data) == rex.select_records(dict(reversed(list(data.items()))))
    assert len(rex.select_records(data)[0]) == 100
    data["1"]["image_path"] = ["/../secret.jpg"]
    with pytest.raises(ValueError, match="unsafe"):
        rex.select_records(data)


def test_pinned_source_rejects_modified_file(tmp_path):
    p = tmp_path / "source.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        rex.load_official(p)


def test_materialize_answers_isolation_missing_images_and_frozen_ids(tmp_path, monkeypatch):
    data = records()
    monkeypatch.setattr(rex, "load_official", lambda _: data)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(rex, "SOURCE_SHA256", rex.digest(source))
    frozen = tmp_path / "frozen"
    rex.freeze_cohort(source, frozen)
    cohort = frozen / "cohort.json"
    root = tmp_path / "images"
    root.mkdir()
    with gzip.open(root / "mimic-cxr-2.0.0-split.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "study_id", "dicom_id", "split"])
        writer.writeheader()
        for s in data:
            writer.writerow({"subject_id": "10000001", "study_id": s,
                             "dicom_id": f"img-{s}", "split": "test"})
    out = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError):
        rex.materialize(source, cohort, root, out)
    assert not out.exists()
    for sid in json.loads(cohort.read_text())["study_ids"]:
        p = root / rex.image_relative(data[sid]["image_path"][0], "10000001", sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(p)
    rex.materialize(source, cohort, root, out)
    assert validate_dataset(out) == []
    ds = load_dataset(out, with_targets=True)
    target = next(iter(ds.targets.values())).turns["t1"]
    assert target["report"] == "SYNTHETIC OFFICIAL REPORT"
    assert target["answer"]["impression"] == ""
    assert "SYNTHETIC" not in (out / "episodes.jsonl").read_text()
    assert load_dataset(out).targets == {}
    with pytest.raises(FileExistsError):
        rex.materialize(source, cohort, root, out)
    changed = json.loads(cohort.read_text()); changed["study_ids"].reverse()
    cohort.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen"):
        rex.materialize(source, cohort, root, tmp_path / "bad-cohort")
