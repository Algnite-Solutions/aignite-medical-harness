from __future__ import annotations

import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from ama.data import load_dataset, validate_dataset
from ama.importers.pathvqa import REPOSITORY, REVISION, import_pathvqa
from ama.pathvqa_score import score_pathvqa
from scripts.pathvqa_baseline import prepare, summarize


def _source(tmp_path, open_answer="bright red"):
    source = tmp_path / "raw"
    (source / "data").mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="JPEG")
    picture = buffer.getvalue()
    files = []
    for split in ("train", "validation", "test"):
        path = source / "data" / f"{split}-00000-of-00001.parquet"
        table = pa.Table.from_pylist([
            {"image": {"bytes": picture, "path": None}, "question": "Is this red?", "answer": "yes"},
            {"image": {"bytes": picture, "path": None}, "question": "What color?", "answer": open_answer},
        ])
        pq.write_table(table, path)
        files.append({"path": f"data/{path.name}",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (source / "snapshot_manifest.json").write_text(json.dumps({
        "repository": REPOSITORY, "resolved_revision": REVISION, "files": files,
    }), encoding="utf-8")
    return source


def test_import_scoring_and_reference_isolation(tmp_path):
    source = _source(tmp_path)
    out = tmp_path / "dataset"
    report = import_pathvqa(source, out, benchmark_count=2)
    assert report["n_imported"] == report["n_targets"] == 6
    assert report["n_unique_images"] == 1
    assert validate_dataset(out) == []
    assert len(list((out / "artifacts" / "images").iterdir())) == 1
    visible = load_dataset(out)
    assert visible.targets == {}
    assert "bright red" not in (out / "instructions.txt").read_text()
    dataset = load_dataset(out, with_targets=True)
    assert dataset.eval_config.scorer == "pathvqa_v1"
    assert len(dataset.info.splits["automedbench_test_2"]) == 2
    dataset.episodes = dataset.select(split="test")
    decisions = {
        "pathvqa-test-000000": [{"turn_id": "t1", "decision": {"answer": "yes"},
                                  "termination": "completed", "model_calls": 1,
                                  "usage": None, "duration_ms": 1}],
        "pathvqa-test-000001": [{"turn_id": "t1", "decision": {"answer": "red"},
                                  "termination": "completed", "model_calls": 1,
                                  "usage": None, "duration_ms": 1}],
    }
    aggregate = score_pathvqa(dataset, decisions)["aggregate"]
    assert aggregate["binary_accuracy"] == 1
    assert aggregate["open_exact_match"] == 0
    assert aggregate["open_token_f1"] == 2 / 3


def test_missing_and_invalid_answers_count_as_wrong(tmp_path):
    out = tmp_path / "dataset"
    import_pathvqa(_source(tmp_path), out, benchmark_count=2)
    dataset = load_dataset(out, with_targets=True)
    dataset.episodes = dataset.select(split="test")
    decisions = {"pathvqa-test-000000": [{"turn_id": "t1", "decision": {"answer": {"yes": True}},
                                           "termination": "completed", "model_calls": 1,
                                           "usage": None, "duration_ms": 1}]}
    aggregate = score_pathvqa(dataset, decisions)["aggregate"]
    assert aggregate["n_expected"] == 2
    assert aggregate["binary_accuracy"] == 0
    assert aggregate["open_token_f1"] == 0
    assert aggregate["completion_rate"] == 0.5


def test_checksum_mismatch_rejects_source(tmp_path):
    source = _source(tmp_path)
    (source / "data" / "test-00000-of-00001.parquet").write_bytes(b"broken")
    try:
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch was accepted")


def test_smoke_import_does_not_invent_a_500_case_split(tmp_path):
    out = tmp_path / "dataset"
    report = import_pathvqa(_source(tmp_path), out, limit_per_split=1)
    assert report["n_imported"] == 3
    assert report["automedbench"]["selected_count"] == 0
    assert "automedbench_test_500" not in load_dataset(out).info.splits


def test_existing_adapter_target_and_benchmark_mapping_are_preserved(tmp_path):
    out = tmp_path / "dataset"
    import_pathvqa(_source(tmp_path, open_answer="x-ray (blue)"), out, benchmark_count=2)
    dataset = load_dataset(out, with_targets=True)
    assert dataset.targets["pathvqa-test-000001"].turns["t1"]["normalized_answer"] == "x-ray (blue"
    origins = {row["id"]: row for row in map(json.loads, (out / "provenance.jsonl").read_text().splitlines())}
    assert origins["pathvqa-test-000000"]["automedbench_qid"] == "pathvqa_00000"
    assert origins["pathvqa-test-000001"]["automedbench_qid"] == "pathvqa_00001"


@pytest.mark.parametrize("answer,expected,valid", [
    ("YES!", 1, 1), ("no", 0, 1), ("yes because it is red", 0, 0),
    ("", 0, 0), (None, 0, 0), ({"answer": "yes"}, 0, 0),
])
def test_binary_label_contract(tmp_path, answer, expected, valid):
    out = tmp_path / "dataset"
    import_pathvqa(_source(tmp_path), out, benchmark_count=2)
    dataset = load_dataset(out, with_targets=True)
    dataset.episodes = dataset.select(episode_id="pathvqa-test-000000")
    decisions = {"pathvqa-test-000000": [{"turn_id": "t1", "decision": {"answer": answer},
                 "termination": "completed", "model_calls": 1, "usage": None, "duration_ms": 1}]}
    aggregate = score_pathvqa(dataset, decisions)["aggregate"]
    assert aggregate["binary_accuracy"] == expected
    assert aggregate["binary_valid_choice_rate"] == valid


def test_cohort_identity_and_incomplete_run_rejection(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    ids = [f"synthetic-{index:03d}" for index in range(100)]
    (dataset / "dataset.json").write_text(json.dumps({"splits": {"automedbench_test_500": ids}}))
    (dataset / "targets.jsonl").write_text("".join(json.dumps({"id": eid, "turns": {"t1": {
        "question_type": "binary" if index % 2 else "open", "answer": "synthetic"}}}) + "\n"
        for index, eid in enumerate(ids)))
    report = {"repository": REPOSITORY, "revision": REVISION, "source_dir": "/original"}
    (dataset / "import_report.json").write_text(json.dumps(report))
    lock = tmp_path / "cohort.json"
    original = prepare(dataset, lock)
    report["source_dir"] = "/another/local/path"
    (dataset / "import_report.json").write_text(json.dumps(report))
    assert prepare(dataset, lock) == original
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"episode_ids": original["episode_ids"]}))
    (run / "decisions.jsonl").write_text("")
    with pytest.raises(ValueError, match="cover the frozen cohort exactly"):
        summarize(dataset, lock, [run], tmp_path / "summary.json")
    report["revision"] = "different"
    (dataset / "import_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="existing cohort differs"):
        prepare(dataset, lock)
