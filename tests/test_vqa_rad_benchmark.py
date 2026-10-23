"""Synthetic tests for the real-model experiment driver; no model API calls."""
import json
from pathlib import Path

import pytest

from ama.data import load_dataset
from ama.importers.vqa_rad import import_vqa_rad
from ama.model import ModelError
from fakes import FakeModel
from scripts import vqa_rad_benchmark as protocol
from test_vqa_rad import source_fixture


def cohort_fixture(tmp_path, count=3):
    dataset = tmp_path / "dataset"
    import_vqa_rad(source_fixture(tmp_path / "source"), dataset)
    root = tmp_path / "cohort"
    cohort = protocol.prepare(dataset, root, count=count)
    return dataset, root, cohort


def fake(name, dataset, cohort, replies=None):
    ds = load_dataset(dataset, with_targets=True)
    if replies is None:
        replies = [json.dumps({"turn_id": "t1", "answer": ds.targets[episode_id].turns["t1"]["answer"],
                               "citations": ["image", "question"], "reasoning_summary": "Synthetic example."})
                   for episode_id in cohort["episode_ids"]]
    model = FakeModel(*replies)
    model.name, model.config, model.timeout = name, {"model": name, "temperature": 0}, 180
    model.last_error_kind = None
    return model


def test_cohort_is_test_only_reproducible_and_refuses_changed_source(tmp_path):
    dataset, root, cohort = cohort_fixture(tmp_path)
    assert cohort["count"] == 3 and cohort["source_test_count"] == 4
    assert len(set(cohort["episode_ids"])) == 3
    assert all(name.startswith("vqarad-test-") for name in cohort["episode_ids"])
    assert protocol.prepare(dataset, root, count=3) == cohort
    with pytest.raises(ValueError, match="settings differ"):
        protocol.prepare(dataset, root, count=2)
    (dataset / "instructions.txt").write_text("changed visible input")
    with pytest.raises(ValueError, match="dataset file changed"):
        protocol.verify(root)


def test_pilot_reused_by_full_run_and_three_models_compared(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    names = ["glm", "qwen", "deepseek"]
    models = {name: fake(name, dataset, cohort) for name in names}
    # Preserve the intentional PR #12 difference in output token caps.
    for name in names[1:]:
        models[name].config["max_tokens"] = 8192
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda name, *a, **k: models[name])
    runs = tmp_path / "runs"
    for name in names:
        assert protocol.run_model(root, name, config=tmp_path / "models.json", runs_root=runs,
                                  delay=0, max_new_cases=1) is None
        assert len(models[name].requests) == 1
        assert protocol.read(runs / name / "batch.json")["status"] == "partial"
    with pytest.raises(ValueError, match="incomplete"):
        protocol.compare(root, names, runs)
    for name in names:
        assert protocol.run_model(root, name, config=tmp_path / "models.json", runs_root=runs, delay=0)
        assert len(models[name].requests) == 3
        protocol.run_model(root, name, config=tmp_path / "models.json", runs_root=runs, delay=0)
        assert len(models[name].requests) == 3
    comparison = protocol.compare(root, names, runs)
    for result in comparison["results"].values():
        assert result["aggregate"]["accuracy_heuristic"] == {"num": 3.0, "den": 3, "value": 1.0}
        assert result["aggregate"]["output_quality"]["strict_output_compliance"]["value"] == 1
    assert all(row["ties"] == 3 for row in comparison["paired"].values())
    assert models["glm"].requests[0] == models["qwen"].requests[0]
    messages = runs / "glm/merged/messages.json"
    saved = protocol.read(messages)
    saved[cohort["episode_ids"][0]][0]["content"] = "different input"
    messages.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="actual model input"):
        protocol.compare(root, names, runs)


def test_received_invalid_answers_are_frozen(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    last = json.dumps({"turn_id": "t1", "answer": {"text": "yes"},
                       "citations": ["image", "question"], "reasoning_summary": "Synthetic."})
    model = fake("invalid", dataset, cohort,
                 ["not JSON", {"role": "assistant", "content": None}, last])
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda *a, **k: model)
    runs = tmp_path / "runs"
    protocol.run_model(root, "invalid", config=tmp_path / "config.json", runs_root=runs, delay=0)
    protocol.run_model(root, "invalid", config=tmp_path / "config.json", runs_root=runs, delay=0)
    assert len(model.requests) == 3
    batch = protocol.read(runs / "invalid/batch.json")
    assert batch["status"] == "completed_with_invalid_answers"
    assert all(len(attempts) == 1 for attempts in batch["attempts"].values())
    result = protocol.compare(root, ["invalid"], runs)["results"]["invalid"]
    assert result["terminations"] == {"invalid_decision": 1, "api_error": 1, "completed": 1}
    assert result["aggregate"]["accuracy_heuristic"]["den"] == 3


def test_transport_resume_does_not_regenerate_previous_answer(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    model = fake("transport", dataset, cohort)
    original = model.complete
    calls = []
    def transport_once(history, tools=None):
        calls.append(1)
        model.last_error_kind = None
        if len(calls) == 2:
            model.last_error_kind = "transport"
            raise ModelError("model connection failed: TimeoutError")
        return original(history, tools)
    model.complete = transport_once
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda *a, **k: model)
    runs = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="waiting_for_transport"):
        protocol.run_model(root, "transport", config=tmp_path / "config.json", runs_root=runs, delay=0)
    protocol.run_model(root, "transport", config=tmp_path / "config.json", runs_root=runs, delay=0)
    assert len(calls) == 4 and len(model.requests) == 3
    batch = protocol.read(runs / "transport/batch.json")
    assert len(batch["attempts"][cohort["episode_ids"][0]]) == 1
    assert len(batch["attempts"][cohort["episode_ids"][1]]) == 2


def test_crash_recovers_returned_shard_without_another_api_call(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    model = fake("crash", dataset, cohort)
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda *a, **k: model)
    original_write = protocol.write_json
    def interrupted_write(path, value):
        if path.name == "batch.json" and value["attempts"]:
            raise KeyboardInterrupt("test crash before batch update")
        original_write(path, value)
    monkeypatch.setattr(protocol, "write_json", interrupted_write)
    runs = tmp_path / "runs"
    with pytest.raises(KeyboardInterrupt):
        protocol.run_model(root, "crash", config=tmp_path / "config.json", runs_root=runs, delay=0)
    assert len(model.requests) == 1
    monkeypatch.setattr(protocol, "write_json", original_write)
    protocol.run_model(root, "crash", config=tmp_path / "config.json", runs_root=runs, delay=0)
    assert len(model.requests) == 3
    batch = protocol.read(runs / "crash/batch.json")
    assert len(batch["attempts"][cohort["episode_ids"][0]]) == 1


def test_execution_does_not_read_targets_until_evaluation(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    model = fake("isolation", dataset, cohort)
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda *a, **k: model)
    (dataset / "targets.jsonl").write_text("invalid evaluator-only secret")
    runs = tmp_path / "runs"
    protocol.run_model(root, "isolation", config=tmp_path / "config.json", runs_root=runs,
                       delay=0, max_new_cases=1)
    assert len(model.requests) == 1
    assert "evaluator-only secret" not in json.dumps(model.requests)
    with pytest.raises(ValueError, match="dataset file changed"):
        protocol.verify(root, evaluator=True)


def test_changed_config_refuses_resume_without_calls(tmp_path, monkeypatch):
    dataset, root, cohort = cohort_fixture(tmp_path)
    model = fake("changed", dataset, cohort)
    monkeypatch.setattr(protocol.RetryingModel, "from_config", lambda *a, **k: model)
    runs = tmp_path / "runs"
    protocol.run_model(root, "changed", config=tmp_path / "config.json", runs_root=runs,
                       delay=0, max_new_cases=1)
    model.config["temperature"] = 1
    with pytest.raises(ValueError, match="saved cohort, config or code differs"):
        protocol.run_model(root, "changed", config=tmp_path / "config.json", runs_root=runs, delay=0)
    assert len(model.requests) == 1
