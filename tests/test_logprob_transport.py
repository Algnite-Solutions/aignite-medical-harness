import json
import urllib.request

import pytest

from ama.cli import main
from ama.model import Model


def config(**extra):
    return {"base_url": "https://example.test/v1", "model": "fixture", "api_key_env": "TEST_KEY", **extra}


@pytest.mark.parametrize("value", ["true", 1, None])
def test_invalid_logprobs(value, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "fixture-key")
    with pytest.raises(ValueError, match="logprobs must"):
        Model("fixture", config(logprobs=value))


@pytest.mark.parametrize("value", [-1, 21, True, "3", 2.5])
def test_invalid_top_logprobs(value, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "fixture-key")
    with pytest.raises(ValueError, match="top_logprobs must"):
        Model("fixture", config(logprobs=True, top_logprobs=value))


def test_top_logprobs_requires_request(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "fixture-key")
    with pytest.raises(ValueError, match="requires"):
        Model("fixture", config(top_logprobs=3))


@pytest.mark.parametrize("returned", [None, {"content": [{"token": "{", "bytes": [123], "logprob": -0.4, "top_logprobs": []}]}])
def test_standard_cli_retains_missing_and_partial_probabilities(tmp_path, monkeypatch, returned):
    monkeypatch.setenv("TEST_KEY", "fixture-key")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(json.dumps({"schema": "ama-dataset", "name": "synthetic", "splits": {"test": ["synthetic"]}}))
    (dataset / "episodes.jsonl").write_text(json.dumps({"id": "synthetic", "turns": [{"id": "t1", "evidence": [{"id": "e1", "text": "Fictional marker ALPHA BRAVO CHARLIE."}]}]}) + "\n")
    (dataset / "instructions.txt").write_text("Synthetic transport check.")
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"models": {"fixture": config(logprobs=True, top_logprobs=3, reasoning_effort="low", max_tokens=128)}}))
    final = json.dumps({"turn_id": "t1", "answer": {"text": "ALPHA BRAVO CHARLIE"}, "citations": ["e1"], "reasoning_summary": "The marker is in the evidence."})
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return json.dumps({"model": "fixture", "choices": [{"message": {"role": "assistant", "content": final}, "finish_reason": "stop", "logprobs": returned}], "usage": {"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60}}).encode()
    calls = []
    def respond(request, timeout):
        payload = json.loads(request.data)
        assert payload["logprobs"] is True and payload["top_logprobs"] == 3 and payload["reasoning_effort"] == "low"
        calls.append(payload)
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    runs = tmp_path / "runs"
    assert main(["run", str(dataset), "--model", "fixture", "--config", str(cfg), "--runs-root", str(runs)]) == 0
    run = next(runs.iterdir())
    events = [json.loads(line) for line in (run / "diagnostics.jsonl").read_text().splitlines()]
    meta = next(event["response_metadata"] for event in events if event["kind"] == "model_call")
    assert meta["logprobs"] == returned and meta["logprob_content"] == final
    assert "fixture-key" not in (run / "diagnostics.jsonl").read_text()
    assert len(calls) == 1
    assert main(["eval", str(run)]) == 0
    assert len(calls) == 1
