import json

import pytest

from scripts import pathvqa


def test_verify_does_not_request_targets(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "import_report.json").write_text(json.dumps({"repository": "r", "revision": "v"}))
    (dataset / "instructions.txt").write_text("instruction")
    for name in ("dataset.json", "episodes.jsonl", "eval.json"):
        (dataset / name).write_text("{}\n")
    lock = {
        "episode_ids": [],
        "public_identity": {
            "source_repository": "r", "source_revision": "v",
            "dataset_json_sha256": pathvqa.digest(dataset / "dataset.json"),
            "episodes_jsonl_sha256": pathvqa.digest(dataset / "episodes.jsonl"),
            "eval_json_sha256": pathvqa.digest(dataset / "eval.json"),
            "instruction_sha256": pathvqa.canonical("instruction"),
            "system_sha256": pathvqa.canonical("instruction\n\n" + pathvqa.DECISION_PROMPT),
            "selected_episodes_sha256": pathvqa.canonical([]), "images": {},
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    original = pathvqa.load_dataset

    def guarded(path, *, with_targets=False):
        assert not with_targets
        return original(path, with_targets=with_targets)

    monkeypatch.setattr(pathvqa, "load_dataset", guarded)
    with pytest.raises(Exception):
        pathvqa.verify(dataset, lock_path)


def test_interrupted_attempt_is_not_retried_without_acknowledgement(tmp_path):
    case = tmp_path / "case" / "attempt-01" / "native"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({
        "termination": "interrupted", "episode_ids": ["e"], "model": "m",
        "model_config": {}, "max_calls": 1,
    }))
    (case / "decisions.jsonl").write_text(json.dumps({
        "episode_id": "e", "turn_id": "t1", "termination": "interrupted"
    }) + "\n")
    row, _ = pathvqa._read_attempt(case.parent, "e", "m", {})
    assert row["termination"] == "interrupted"
    with pytest.raises(RuntimeError):
        raise RuntimeError("interrupted attempt requires --acknowledge-interrupted")

