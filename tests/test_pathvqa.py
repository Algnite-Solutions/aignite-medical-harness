from __future__ import annotations

import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from ama.data import load_dataset, validate_dataset
from ama.importers.pathvqa import REPOSITORY, REVISION, import_pathvqa as _import_pathvqa
from ama.pathvqa_score import _f1, normalize_answer, score_pathvqa
from ama.runner import execute


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
    (source / "source.lock.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION,
        "files": [{**item, "size": (source / item["path"]).stat().st_size} for item in files],
    }), encoding="utf-8")
    return source


def import_pathvqa(source, out, **kwargs):
    return _import_pathvqa(source, out, source_lock=source / "source.lock.json", **kwargs)


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
        assert "size mismatch" in str(exc) or "checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch was accepted")


def test_source_lock_rejects_mutable_manifest_and_duplicate_shards(tmp_path):
    source = _source(tmp_path)
    path = source / "snapshot_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="committed lock"):
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)
    manifest["files"][0]["sha256"] = json.loads((source / "source.lock.json").read_text())["files"][0]["sha256"]
    manifest["files"].append(dict(manifest["files"][0]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate source manifest"):
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)


def test_missing_shard_and_existing_destination_are_rejected(tmp_path):
    source = _source(tmp_path)
    manifest_path = source / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly the locked"):
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)
    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileExistsError):
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)


def test_unsafe_source_path_and_corrupt_image_are_rejected(tmp_path):
    source = _source(tmp_path)
    manifest_path = source / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "data/../../outside.parquet"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="committed lock"):
        import_pathvqa(source, tmp_path / "dataset", benchmark_count=2)

    source = _source(tmp_path / "second")
    shard = source / "data" / "test-00000-of-00001.parquet"
    table = pq.read_table(shard).to_pylist()
    table[0]["image"]["bytes"] = b"not an image"
    pq.write_table(pa.Table.from_pylist(table), shard)
    manifest_path = source / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock_path = source / "source.lock.json"
    lock = json.loads(lock_path.read_text())
    for records in (manifest["files"], lock["files"]):
        item = next(record for record in records if record["path"].startswith("data/test-"))
        item["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        item["size"] = shard.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(Exception, match="image|cannot identify"):
        import_pathvqa(source, tmp_path / "other-dataset", benchmark_count=2)


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


def test_real_runner_request_excludes_gold_answer(tmp_path):
    source = _source(tmp_path, open_answer="secret_synthetic_gold_731")
    out = tmp_path / "dataset"
    import_pathvqa(source, out, benchmark_count=2)

    class FakeModel:
        name = "fake-vision"
        config = {"model": "fake-vision"}
        timeout = 1
        last_usage = None

        def complete(self, history, tools=None):
            assert "secret_synthetic_gold_731" not in json.dumps(history)
            return {"role": "assistant", "content": json.dumps({"turn_id": "t1", "answer": "red",
                "citations": ["image"], "reasoning_summary": "The supplied image supports the answer."})}

    run_dir = execute(out, FakeModel(), episode_ids=["pathvqa-test-000001"],
                      runs_root=tmp_path / "runs", max_calls=1)
    history = json.loads((run_dir / "messages.json").read_text())
    assert "secret_synthetic_gold_731" not in json.dumps(history)


@pytest.mark.parametrize("answer,expected,valid", [
    ("YES!", 1, 1), ("no", 0, 1), ("yes because it is red", 0, 0),
    ("", 0, 0), ("...", 0, 0), (None, 0, 0), ({"answer": "yes"}, 0, 0),
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


def test_lexical_normalization_and_multiset_f1():
    assert normalize_answer("ＹＥＳ！") == "yes"
    assert normalize_answer("...？") == ""
    assert _f1("red red", "red blue") == 0.5
    assert _f1("red red", "red") == 2 / 3



