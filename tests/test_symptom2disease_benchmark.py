import csv
import json
from pathlib import Path

import pytest

from ama.data import validate_dataset
from ama.model import Model, ModelError
from ama.importers.symptom2disease import LABELS
from scripts.symptom2disease_benchmark import (
    RetryingModel, compare_models, main, prepare_benchmark, run_model, select_rows,
)
from fakes import FakeModel, decision


def _source(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "text"])
        writer.writeheader()
        for label in LABELS:
            for index in range(6):
                writer.writerow({"label": label, "text": f"Unique symptoms {label} {index}"})
        writer.writerow({"label": LABELS[0], "text": f"Unique symptoms {LABELS[0]} 0"})
    return path


def _model(name: str, replies: list) -> FakeModel:
    model = FakeModel(*replies)
    model.name = name
    model.config = {"model": name, "temperature": 0, "max_tokens": 1024}
    model.timeout = 30
    model.last_error_kind = None
    return model


def test_fixed_sample_is_reproducible_and_refuses_changed_source(tmp_path):
    source = _source(tmp_path / "raw.csv")
    rows, extras, duplicates = select_rows(source)
    assert len(rows) == len(set(rows)) == 100
    assert len(extras) == 4 and duplicates == 1
    root = tmp_path / "benchmark"
    first = prepare_benchmark(source, root)
    assert first["selected_csv_rows"] == rows
    assert first["duplicates_excluded_before_sampling"] == 1
    assert sorted(first["label_counts"].values()) == [4] * 20 + [5] * 4
    assert validate_dataset(root / "dataset") == []
    assert prepare_benchmark(source, root) == first
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        prepare_benchmark(source, root)


def test_invalid_model_answer_is_frozen_and_comparison_is_paired(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    benchmark = prepare_benchmark(_source(tmp_path / "raw.csv"), root)
    first = _model("first", [decision(answer="Acne", citations=["symptoms"])
                             for _ in range(100)])
    # Both an invalid Decision and a returned assistant message rejected by
    # Agent must remain frozen after a second invocation of the same batch.
    second = _model("second", ["not JSON", {"role": "assistant", "content": None}] + [
        decision(answer={"disease": "Acne"}, citations=["symptoms"]) for _ in range(98)])
    models = {"first": first, "second": second}
    monkeypatch.setattr(RetryingModel, "from_config", lambda name, *args, **kwargs: models[name])
    for name in models:
        run_model(root, name, config=tmp_path / "models.json", timeout=30, delay=0)
    assert len(first.requests) == len(second.requests) == 100
    assert run_model(root, "second", config=tmp_path / "models.json", timeout=30, delay=0)
    assert len(second.requests) == 100
    batch = json.loads((root / "runs/second/batch.json").read_text())
    assert batch["status"] == "completed_with_invalid_answers"
    assert len(batch["attempts"][benchmark["episode_ids"][0]]) == 1
    assert len(batch["attempts"][benchmark["episode_ids"][1]]) == 1
    assert batch["attempts"][benchmark["episode_ids"][1]][0] == {
        "run": batch["runs"][benchmark["episode_ids"][1]],
        "termination": "api_error", "error_kind": None}
    report = compare_models(root, ["first", "second"])
    assert report["results"]["second"]["terminations"] == {
        "completed": 98, "invalid_decision": 1, "api_error": 1}
    assert report["results"]["second"]["accuracy"]["den"] == 100
    assert report["results"]["second"]["answer_object_format"] == {
        "num": 98, "den": 100, "value": 0.98}
    assert report["results"]["second"]["answer_object_format"] != report[
        "results"]["second"]["accuracy"]
    assert report["results"]["first"]["answer_object_format"]["num"] == 0
    assert report["results"]["first"]["accuracy"]["num"] > 0
    assert sum(report["paired"]["first_vs_second"].values()) == 100
    messages = root / "runs/second/merged/messages.json"
    content = json.loads(messages.read_text())
    content[benchmark["episode_ids"][0]][0]["content"] = "tampered prompt"
    messages.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="system prompt"):
        compare_models(root, ["first", "second"])


def test_exhausted_transport_can_resume_without_requerying_terminal_cases(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    benchmark = prepare_benchmark(_source(tmp_path / "raw.csv"), root)
    class TransportOnce(FakeModel):
        def __init__(self):
            super().__init__()
            self.name = "transport"
            self.config = {"model": "transport", "temperature": 0}
            self.timeout = 30
            self.last_error_kind = None
            self.count = 0

        def complete(self, history, tools=None):
            self.count += 1
            self.requests.append((history, tools))
            if self.count == 2:
                self.last_error_kind = "transport"
                raise ModelError("model connection failed: TimeoutError")
            self.last_error_kind = None
            return {"role": "assistant", "content": decision(answer={
                "disease": "Acne"}, citations=["symptoms"])}

    model = TransportOnce()
    monkeypatch.setattr(RetryingModel, "from_config", lambda *args, **kwargs: model)
    with pytest.raises(RuntimeError, match="waiting_for_transport"):
        run_model(root, "transport", config=tmp_path / "models.json", timeout=30, delay=0)
    batch = json.loads((root / "runs/transport/batch.json").read_text())
    assert batch["status"] == "waiting_for_transport"
    assert len(batch["runs"]) == 2 and model.count == 2
    run_model(root, "transport", config=tmp_path / "models.json", timeout=30, delay=0)
    assert model.count == 101  # First case was frozen; only failed case was retried.
    batch = json.loads((root / "runs/transport/batch.json").read_text())
    assert len(batch["attempts"][benchmark["episode_ids"][0]]) == 1
    assert len(batch["attempts"][benchmark["episode_ids"][1]]) == 2
    assert compare_models(root, ["transport"])["results"]["transport"]["accuracy"]["den"] == 100


def test_transport_retry_is_bounded_and_does_not_retry_permanent_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fixture")
    monkeypatch.setattr("scripts.symptom2disease_benchmark.time.sleep", lambda _: None)
    model = RetryingModel("fixture", {"base_url": "https://example.test/v1",
                                    "model": "fixture", "api_key_env": "TEST_API_KEY"})
    model.retry_log = tmp_path / "retry.jsonl"
    model.max_attempts = 3
    attempts = []
    def transient(self, history, tools=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise ModelError("HTTP 429", http_status=429)
        return {"role": "assistant", "content": "received answer"}
    monkeypatch.setattr(Model, "complete", transient)
    assert model.complete([])["content"] == "received answer"
    assert len(attempts) == 3 and len(model.retry_log.read_text().splitlines()) == 2
    attempts.clear()
    def malformed_api(self, history, tools=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelError("model endpoint returned invalid JSON")
        return {"role": "assistant", "content": "received answer"}
    monkeypatch.setattr(Model, "complete", malformed_api)
    assert model.complete([])["content"] == "received answer"
    assert len(attempts) == 2
    attempts.clear()
    def permanent(self, history, tools=None):
        attempts.append(1)
        raise ModelError("HTTP 401", http_status=401)
    monkeypatch.setattr(Model, "complete", permanent)
    with pytest.raises(ModelError):
        model.complete([])
    assert len(attempts) == 1 and model.last_error_kind == "permanent_api"


def test_429_then_invalid_reply_is_frozen(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    benchmark = prepare_benchmark(_source(tmp_path / "raw.csv"), root)
    monkeypatch.setenv("TEST_API_KEY", "fixture")
    monkeypatch.setattr("scripts.symptom2disease_benchmark.time.sleep", lambda _: None)
    model = RetryingModel("retry-invalid", {"base_url": "https://example.test/v1",
                                           "model": "fixture", "api_key_env": "TEST_API_KEY"},
                          timeout=30)
    monkeypatch.setattr(RetryingModel, "from_config", lambda *args, **kwargs: model)
    calls = []
    def reply(self, history, tools=None):
        calls.append(1)
        if len(calls) == 1:
            raise ModelError("HTTP 429", http_status=429)
        if len(calls) == 2:
            return {"role": "assistant", "content": None}
        return {"role": "assistant", "content": decision(answer={
            "disease": "Acne"}, citations=["symptoms"])}
    monkeypatch.setattr(Model, "complete", reply)
    run_model(root, "retry-invalid", timeout=30, delay=0)
    assert model.last_error_kind is None
    batch = json.loads((root / "runs/retry-invalid/batch.json").read_text())
    first = benchmark["episode_ids"][0]
    assert batch["attempts"][first][0]["termination"] == "api_error"
    assert batch["attempts"][first][0]["error_kind"] is None
    assert len(calls) == 101
    run_model(root, "retry-invalid", timeout=30, delay=0)
    assert len(calls) == 101
    assert compare_models(root, ["retry-invalid"])["results"]["retry-invalid"][
        "accuracy"]["den"] == 100


def test_crash_before_batch_write_recovers_frozen_shard(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    benchmark = prepare_benchmark(_source(tmp_path / "raw.csv"), root)
    model = _model("crash", ["not JSON"] + [decision(answer={
        "disease": "Acne"}, citations=["symptoms"]) for _ in range(99)])
    monkeypatch.setattr(RetryingModel, "from_config", lambda *args, **kwargs: model)
    from scripts import symptom2disease_benchmark as protocol
    real_write = protocol.write_json
    def crash_on_case_record(path, value):
        if path.name == "batch.json" and value["attempts"]:
            raise KeyboardInterrupt("simulated process exit at batch write")
        real_write(path, value)
    monkeypatch.setattr(protocol, "write_json", crash_on_case_record)
    with pytest.raises(KeyboardInterrupt, match="simulated process exit"):
        run_model(root, "crash", timeout=30, delay=0)
    assert len(model.requests) == 1
    saved = json.loads((root / "runs/crash/batch.json").read_text())
    assert saved["attempts"] == {}
    monkeypatch.setattr(protocol, "write_json", real_write)
    run_model(root, "crash", timeout=30, delay=0)
    assert len(model.requests) == 100
    batch = json.loads((root / "runs/crash/batch.json").read_text())
    first = benchmark["episode_ids"][0]
    assert len(batch["attempts"][first]) == 1
    assert batch["attempts"][first][0]["termination"] == "invalid_decision"
    assert compare_models(root, ["crash"])["results"]["crash"]["accuracy"]["den"] == 100


def test_one_command_prepares_scores_compares_and_reruns_without_new_calls(tmp_path, monkeypatch):
    source, root = _source(tmp_path / "raw.csv"), tmp_path / "benchmark"
    model = _model("one", [decision(answer={"disease": "Acne"}, citations=["symptoms"])
                           for _ in range(100)])
    monkeypatch.setattr(RetryingModel, "from_config", lambda *args, **kwargs: model)
    args = ["--source", str(source), "--out", str(root), "--model", "one",
            "--delay", "0", "--timeout", "30"]
    assert main(args) == 0
    assert len(model.requests) == 100
    report = json.loads((root / "comparison.json").read_text())
    assert report["results"]["one"]["accuracy"]["den"] == 100
    assert main(args) == 0
    assert len(model.requests) == 100
