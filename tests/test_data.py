"""Dataset schema and evaluator-only data boundary."""
import json

import pytest

from ama.data import load_dataset, validate_dataset


def write_dataset(root, episodes, targets=None, splits=None):
    (root / "dataset.json").write_text(json.dumps({"schema": "ama-dataset", "name": "test",
                                                  "splits": splits or {"all": [e["id"] for e in episodes]}}))
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n")
    if targets is not None:
        (root / "targets.jsonl").write_text("\n".join(json.dumps(t) for t in targets) + "\n")
    return root


def example():
    return {"id": "ep", "turns": [
        {"id": "t1", "observation": "History", "evidence": [{"id": "hpi", "text": "pain"}]},
        {"id": "t2", "observation": "Labs", "evidence": [{"id": "lab", "type": "lab", "text": "WBC 14"}]},
    ]}


def test_minimal_multiturn_needs_no_fake_time(tmp_path):
    root = write_dataset(tmp_path, [example()])
    assert validate_dataset(root) == []
    episode = load_dataset(root).episodes[0]
    assert episode.id == "ep" and episode.turns[1].available_at is None
    assert episode.turns[1].evidence[0].type == "lab"


def test_unknown_schema_rejected(tmp_path):
    root = write_dataset(tmp_path, [example()])
    (root / "dataset.json").write_text('{"schema":"unknown","name":"old","splits":{}}')
    assert "unsupported dataset schema" in validate_dataset(root)[0]


def test_duplicate_ids_and_unknown_refs(tmp_path):
    ep = example()
    ep["turns"][1]["id"] = "t1"
    ep["turns"][1]["evidence"][0]["id"] = "hpi"
    root = write_dataset(tmp_path, [ep], targets=[{"id": "ep", "turns": {"missing": {}}}],
                         splits={"all": ["ep", "absent"]})
    errors = validate_dataset(root)
    assert any("duplicate turn" in e for e in errors)
    assert any("duplicate evidence" in e for e in errors)
    assert any("unknown episode" in e for e in errors)
    assert any("unknown turn" in e for e in errors)


def test_file_safety_and_missing_content(tmp_path):
    ep = example()
    ep["turns"][0]["evidence"] = [{"id": "x", "file": "../escape.png"}]
    assert any("unsafe file" in e for e in validate_dataset(write_dataset(tmp_path, [ep])))
    ep["turns"][0]["evidence"] = [{"id": "x"}]
    assert "neither text nor file" in validate_dataset(write_dataset(tmp_path, [ep]))[0]


def test_true_timestamps_must_be_monotonic_when_present(tmp_path):
    ep = example()
    ep["turns"][0]["available_at"] = "2026-01-03T00:00:00Z"
    ep["turns"][1]["available_at"] = "2026-01-02T00:00:00Z"
    assert any("non-monotonic" in e for e in validate_dataset(write_dataset(tmp_path, [ep])))


def test_run_loader_never_reads_evaluator_files(tmp_path, monkeypatch):
    root = write_dataset(tmp_path, [example()], targets=[{"id": "ep", "turns": {"t1": {"answer": "x"}}}])
    (root / "eval.json").write_text('{"scorer":"rocov2"}')
    (root / "provenance.jsonl").write_text('private source\n')
    original = type(root).read_text

    def guarded(path, *args, **kwargs):
        if path.name in {"targets.jsonl", "eval.json", "provenance.jsonl"}:
            raise AssertionError(f"run opened {path.name}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(root), "read_text", guarded)
    ds = load_dataset(root, with_targets=False)
    assert ds.targets == {} and ds.eval_config is None
