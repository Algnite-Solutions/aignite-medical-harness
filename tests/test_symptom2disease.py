import csv
import json
from pathlib import Path

import pytest

from ama.cli import main
from ama.data import load_dataset, validate_dataset
from ama.importers.symptom2disease import import_symptom2disease


def source_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Unnamed: 0", "label", "text"])
        writer.writeheader()
        writer.writerows([
            {"Unnamed: 0": "0", "label": "Psoriasis", "text": "  My skin is itchy.  "},
            {"Unnamed: 0": "1", "label": "Migraine", "text": "I have a headache."},
            {"Unnamed: 0": "2", "label": "", "text": "I feel tired."},
            {"Unnamed: 0": "3", "label": "Migraine", "text": " "},
            {"Unnamed: 0": "4", "label": "Migraine", "text": "I have a headache."},
        ])
    return path


def test_import_generates_valid_single_turn_dataset_without_label_leakage(tmp_path):
    source = source_csv(tmp_path / "Symptom2Disease.csv")
    out = tmp_path / "dataset"
    report = import_symptom2disease(source, out)

    assert validate_dataset(out) == []
    dataset = load_dataset(out, with_targets=True)
    assert dataset.info.splits == {"all": ["s2d-000001", "s2d-000002", "s2d-000005"]}
    assert dataset.eval_config.scorer == "unscored"
    assert report["n_source_rows"] == 5 and report["n_imported"] == 3
    assert report["n_eligible"] == 3 and report["n_not_selected_due_limit"] == 0
    assert report["n_not_selected_due_row_filter"] == 0
    assert report["excluded"] == {"empty_label": 1, "empty_text": 1}
    assert report["duplicate_text_in_selected"] == 1
    assert all(len(ep.turns) == 1 and len(ep.turns[0].evidence) == 1 for ep in dataset.episodes)
    assert dataset.episodes[0].turns[0].evidence[0].text == "My skin is itchy."
    assert "Psoriasis" not in (out / "episodes.jsonl").read_text()
    assert "Migraine" not in (out / "instructions.txt").read_text()
    assert dataset.targets["s2d-000001"].turns["t1"]["answer"] == {"disease": "Psoriasis"}
    assert json.loads((out / "provenance.jsonl").read_text().splitlines()[0])["source_row"] == 2


def test_import_limit_is_deterministic_and_existing_output_is_preserved(tmp_path):
    source = source_csv(tmp_path / "Symptom2Disease.csv")
    first, second = tmp_path / "first", tmp_path / "second"
    report = import_symptom2disease(source, first, limit=1)
    import_symptom2disease(source, second, limit=1)
    assert report["n_not_selected_due_limit"] == 2
    for name in ("episodes.jsonl", "targets.jsonl", "instructions.txt"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert load_dataset(first).info.splits == load_dataset(second).info.splits
    before = (first / "targets.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        import_symptom2disease(source, first, limit=2)
    assert (first / "targets.jsonl").read_bytes() == before


def test_import_selected_source_rows_preserves_original_ids(tmp_path):
    source = source_csv(tmp_path / "Symptom2Disease.csv")
    out = tmp_path / "selected"
    report = import_symptom2disease(source, out, rows=[6, 2])
    assert validate_dataset(out) == []
    assert load_dataset(out).info.splits["all"] == ["s2d-000001", "s2d-000005"]
    assert report["selection"]["rows"] == [2, 6]
    assert report["selected_source_rows"] == [
        {"id": "s2d-000001", "source_row": 2},
        {"id": "s2d-000005", "source_row": 6},
    ]
    assert report["n_not_selected_due_row_filter"] == 1
    with pytest.raises(ValueError, match="not found"):
        import_symptom2disease(source, tmp_path / "missing", rows=[100])
    with pytest.raises(ValueError, match="choose --row or --limit"):
        import_symptom2disease(source, tmp_path / "both", rows=[2], limit=1)


def test_cli_import_and_invalid_input(tmp_path, capsys):
    source = source_csv(tmp_path / "Symptom2Disease.csv")
    out = tmp_path / "output"
    assert main(["import", "symptom2disease", "--source", str(source), "--out", str(out)]) == 0
    assert "imported ->" in capsys.readouterr().out
    assert validate_dataset(out) == []
    with pytest.raises(SystemExit, match="2"):
        main(["import", "symptom2disease", "--source", str(source),
              "--out", str(tmp_path / "bad"), "--id", "x"])
    with pytest.raises(SystemExit, match="2"):
        main(["import", "symptom2disease", "--source", str(source),
              "--out", str(tmp_path / "bad"), "--split", "test"])
    with pytest.raises(SystemExit, match="0"):
        main(["import", "symptom2disease", "--help"])
    help_text = capsys.readouterr().out
    assert "--limit" in help_text and "--row" in help_text
    assert "--split" not in help_text and "--id" not in help_text
    empty = tmp_path / "empty.csv"
    empty.write_text("wrong,column\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label and text"):
        import_symptom2disease(empty, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_committed_demo_is_valid_and_has_no_visible_labels():
    root = Path(__file__).resolve().parents[1] / "datasets" / "symptom2disease_demo"
    assert validate_dataset(root) == []
    public = load_dataset(root)
    full = load_dataset(root, with_targets=True)
    assert len(public.episodes) == len(full.targets) == 6
    assert public.targets == {}
    assert all(len(ep.turns) == 1 for ep in public.episodes)
    assert {target.turns["t1"]["answer"]["disease"] for target in full.targets.values()} == {
        "Psoriasis", "Varicose Veins", "Typhoid", "Chicken pox", "Impetigo", "Dengue",
    }
    report = json.loads((root / "import_report.json").read_text(encoding="utf-8"))
    assert report["selection"]["rows"] == [2, 52, 102, 152, 202, 252]
    assert not Path(report["source_file"]).is_absolute()
