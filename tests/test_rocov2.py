import csv
import json
from pathlib import Path

import pytest

from ama.cli import main
from ama.data import load_dataset, validate_dataset
from ama.importers.rocov2 import import_rocov2
from ama.scorer import score_rocov2


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    images = source / "test"
    images.mkdir(parents=True)
    for image_id in ("ROCO_test_001", "ROCO_test_002"):
        (images / f"{image_id}.jpg").write_bytes(f"jpeg-{image_id}".encode())
    write_csv(source / "test_captions.csv", ["ID", "Caption"], [
        {"ID": "ROCO_test_001", "Caption": "CT chest axial view."},
        {"ID": "ROCO_test_002", "Caption": "T2 weighted MRI."},
    ])
    write_csv(source / "test_concepts_manual.csv", ["ID", "CUIs"], [
        {"ID": "ROCO_test_001", "CUIs": "C0040405;C0000001"},
        {"ID": "ROCO_test_002", "CUIs": "C0024485"},
    ])
    write_csv(source / "license_information.csv", ["ID", "PMCID", "Attribution", "Link"], [
        {"ID": "ROCO_test_001", "PMCID": "PMC1", "Attribution": "CC BY A", "Link": "https://example/1"},
        {"ID": "ROCO_test_002", "PMCID": "PMC2", "Attribution": "CC BY-NC B", "Link": "https://example/2"},
    ])
    write_csv(source / "cui_mapping.csv", ["CUI", "Canonical name"], [
        {"CUI": "C0040405", "Canonical name": "CT"},
        {"CUI": "C0000001", "Canonical name": "Finding"},
        {"CUI": "C0024485", "Canonical name": "MRI"},
    ])
    return source


def test_importer_writes_single_turn_dataset_cards_and_artifacts(tmp_path):
    source = source_fixture(tmp_path)
    out = tmp_path / "out"
    report = import_rocov2(source, out, ids=["ROCO_test_002", "ROCO_test_001"], limit=1)
    assert report["selected_ids"] == ["ROCO_test_002"]
    assert validate_dataset(out) == []
    ds = load_dataset(out, with_targets=True)
    assert ds.eval_config.scorer == "rocov2" and len(ds.episodes) == 1
    evidence = ds.episodes[0].turns[0].evidence[0]
    assert not evidence.text and evidence.file == "artifacts/ROCO_test_002.jpg"
    assert (out / evidence.file).read_bytes() == (source / "test" / "ROCO_test_002.jpg").read_bytes()
    public = json.dumps(ds.episodes[0].model_dump(mode="json"))
    assert "T2 weighted MRI" not in public and "C0024485" not in public
    assert (out / "DATASET_CARD.md").exists() and (out / "DATASET_CARD.zh-CN.md").exists()


def test_importer_cli_and_missing_explicit_id(tmp_path, capsys):
    source = source_fixture(tmp_path)
    out = tmp_path / "cli"
    assert main(["import", "rocov2", "--source", str(source), "--out", str(out),
                 "--id", "ROCO_test_001"]) == 0
    assert "imported ->" in capsys.readouterr().out
    with pytest.raises(ValueError, match="missing"):
        import_rocov2(source, tmp_path / "bad", ids=["not-present"])


def test_rocov2_scorer_normalizes_caption_and_cui_sets(tmp_path):
    out = tmp_path / "out"
    import_rocov2(source_fixture(tmp_path), out, ids=["ROCO_test_001"])
    dataset = load_dataset(out, with_targets=True)
    decisions = {"ROCO_test_001": [{
        "turn_id": "t1", "termination": "completed", "model_calls": 2,
        "usage": "unknown", "duration_ms": 1,
        "decision": {"turn_id": "t1", "answer": {"caption": "ct, CHEST!", "cuis": ["c0000001", "C0040405", "C0040405"]},
                     "citations": ["image-ROCO_test_001"]},
    }]}
    result = score_rocov2(dataset, decisions)
    turn = result["per_episode"]["ROCO_test_001"]["turns"][0]
    assert turn["caption_token_precision"]["value"] == 1
    assert turn["caption_token_recall"]["value"] == 0.5
    assert turn["caption_token_f1"]["value"] == pytest.approx(2 / 3)
    assert turn["cui_f1"]["value"] == 1 and turn["refs_covered"] is True


def test_rocov2_scorer_treats_malformed_answer_as_empty(tmp_path):
    out = tmp_path / "out"
    import_rocov2(source_fixture(tmp_path), out, ids=["ROCO_test_001"])
    dataset = load_dataset(out, with_targets=True)
    decisions = {"ROCO_test_001": [{
        "turn_id": "t1", "termination": "completed", "model_calls": 1,
        "usage": "unknown", "duration_ms": 1,
        "decision": {"turn_id": "t1", "answer": "bad", "citations": []},
    }]}
    result = score_rocov2(dataset, decisions)
    assert result["aggregate"]["caption_token_f1"]["value"] == 0
    assert result["aggregate"]["cui_f1"]["value"] == 0


def test_rocov2_import_run_eval(tmp_path, monkeypatch):
    out = tmp_path / "out"
    import_rocov2(source_fixture(tmp_path), out, ids=["ROCO_test_001"])
    assert validate_dataset(out) == []
    from ama.model import Model
    from fakes import FakeModel, decision
    model = FakeModel(decision(answer={"caption": "CT chest axial view.",
                                       "cuis": ["C0040405", "C0000001"]},
                               citations=["image-ROCO_test_001"]))
    monkeypatch.setattr(Model, "from_config", lambda *a, **kw: model)
    runs = tmp_path / "runs"
    assert main(["run", str(out), "--model", "test-model", "--runs-root", str(runs)]) == 0
    run_dir = next(runs.iterdir())
    assert main(["eval", str(run_dir)]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["scored"]["aggregate"]["caption_token_f1"]["value"] == 1
