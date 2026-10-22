"""The release lock covers questions, images and prompts before any API request."""
import io
import json

import pytest
from PIL import Image

from scripts.pathvqa_v2 import _attempts, _temporary, identity, mini_dataset, prepare, run, verify
from scripts.render_pathvqa_results import render


def _fixture(tmp_path):
    folder = tmp_path / "dataset"
    images = folder / "artifacts" / "images"
    images.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    (images / "red.png").write_bytes(buffer.getvalue())
    ids = [f"synthetic-{i:03d}" for i in range(500)]
    (folder / "dataset.json").write_text(json.dumps({"schema": "ama-dataset", "name": "synthetic",
        "splits": {"test": ids, "automedbench_test_500": ids}}))
    (folder / "episodes.jsonl").write_text("".join(json.dumps({"id": eid, "turns": [{"id": "t1",
        "observation": f"Synthetic question {i}?", "evidence": [{"id": "image", "type": "image",
        "file": "artifacts/images/red.png"}]}]}) + "\n" for i, eid in enumerate(ids)))
    (folder / "targets.jsonl").write_text("".join(json.dumps({"id": eid, "turns": {"t1": {
        "answer": "yes" if i < 49 else "red", "question_type": "binary" if i < 49 else "open"}}})
        + "\n" for i, eid in enumerate(ids)))
    (folder / "eval.json").write_text('{"scorer":"pathvqa_v1"}')
    (folder / "instructions.txt").write_text("Synthetic task instruction.\n")
    (folder / "import_report.json").write_text(json.dumps({"repository": "synthetic", "revision": "test"}))
    old = tmp_path / "old.lock.json"
    old.write_text(json.dumps({"size": 100, "question_types": {"binary": 49, "open": 51},
                               "episode_ids": ids[:100]}))
    return folder, old


def test_v2_lock_catches_question_prompt_and_image_drift(tmp_path):
    folder, old = _fixture(tmp_path)
    lock_path = tmp_path / "new.lock.json"
    lock = prepare(folder, old, lock_path)
    assert verify(folder, lock_path) == lock
    mini = mini_dataset(folder, lock, tmp_path / "mini")
    assert len((mini / "episodes.jsonl").read_text().splitlines()) == 100
    assert len((mini / "targets.jsonl").read_text().splitlines()) == 100
    (mini / "instructions.txt").write_text("Tampered mini prompt")
    with pytest.raises(ValueError, match="changed"):
        mini_dataset(folder, lock, mini)
    (mini / "instructions.txt").write_text("Synthetic task instruction.\n")
    (folder / "instructions.txt").write_text("Changed prompt")
    with pytest.raises(ValueError, match="changed"):
        verify(folder, lock_path)
    (folder / "instructions.txt").write_text("Synthetic task instruction.\n")
    image = folder / "artifacts" / "images" / "red.png"
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify(folder, lock_path)


def test_only_transport_and_transient_http_errors_are_retried():
    assert _temporary({"termination": "api_error", "error": "model endpoint returned HTTP 429"})
    assert _temporary({"termination": "api_error", "error": "model connection failed: TimeoutError"})
    assert not _temporary({"termination": "api_error", "error": "model endpoint returned HTTP 400"})
    assert not _temporary({"termination": "invalid_decision", "error": "not JSON"})


def test_public_table_uses_saved_aggregate():
    summary = {"runs": [{"alias": "synthetic", "termination_counts": {"completed": 1},
        "n_retries": 2, "aggregate": {"n_expected": 2, "valid_answer_rate": 0.5,
            "binary_accuracy": 1, "binary_count": 1, "open_exact_match": 0,
            "open_count": 1, "open_token_f1": 0.25,
            "output_quality": {"strict_output_compliance": {"num": 1, "den": 2}}}}]}
    assert "| `synthetic` | 1/2 | 1/2 | 1/1 | 0/1 | 0.2500 | 1/2 | 2 |" in render(summary)


def test_orphan_attempt_file_stops_resume(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="orphan"):
        _attempts(case)


@pytest.mark.parametrize("first,expected_calls", [
    ({"termination": "api_error", "error": "model endpoint returned HTTP 429"}, 2),
    ({"termination": "invalid_decision", "error": "bad JSON"}, 1),
])
def test_resumption_retries_only_without_a_received_answer(tmp_path, monkeypatch, first, expected_calls):
    from ama.model import Model
    import ama.runner

    folder, _ = _fixture(tmp_path)
    eid = "synthetic-000"
    lock_path = tmp_path / "one.lock.json"
    lock_path.write_text(json.dumps({"episode_ids": [eid], "identity": identity(folder, [eid])}))

    class FakeModel:
        name = "synthetic-model"
        config = {"temperature": 0, "max_tokens": 8192, "model": "synthetic"}

    monkeypatch.setattr(Model, "from_config", lambda *args, **kwargs: FakeModel())
    seen = []

    def fake_execute(dataset, model, *, episode_ids, runs_root, max_calls):
        native = runs_root / "native"
        native.mkdir()
        record = first if not seen else {"termination": "completed", "error": None}
        seen.append(record)
        (native / "manifest.json").write_text(json.dumps({"termination": "completed",
            "episode_ids": episode_ids, "model": model.name, "model_config": model.config, "max_calls": max_calls}))
        (native / "decisions.jsonl").write_text(json.dumps({"episode_id": eid, "turn_id": "t1",
            "decision": None, **record}) + "\n")
        return native

    monkeypatch.setattr(ama.runner, "execute", fake_execute)
    root = tmp_path / "runs"
    run(folder, lock_path, root, tmp_path / "config.json", "synthetic-model", interval=0)
    run(folder, lock_path, root, tmp_path / "config.json", "synthetic-model", interval=0)
    assert len(seen) == expected_calls
