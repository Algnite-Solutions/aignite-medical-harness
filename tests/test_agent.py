"""Deterministic protocol and replay-controller coverage."""
import json

from ama.agent import Interaction, run_episode
from ama.data import Episode
from ama.model import ModelError, ScriptedModel
from ama.recorder import Budget, Recorder, TraceConfig


EPISODE = {"id": "case", "turns": [
    {"id": "t1", "observation": "History", "evidence": [{"id": "e1", "type": "note", "text": "pain"}]},
    {"id": "t2", "observation": "Labs", "evidence": [{"id": "e2", "type": "lab", "text": "WBC 14"}]},
]}


def run(tmp_path, actions, *, protocol=None, episode=None, per_turn=5,
        interaction=None, model=None, retries=0, files=None):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "dataset.json").write_text('{"schema":"ama-dataset","name":"test","splits":{}}')
    (source / "episodes.jsonl").write_text(json.dumps(episode or EPISODE) + "\n")
    for name, content in (files or {}).items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ep = Episode.model_validate(episode or EPISODE)
    recorder = Recorder(tmp_path / "runs", "test", "scripted", source, [ep.id], TraceConfig(),
                        Budget(30, per_turn, 60, retries))
    summary = run_episode(ep, model or ScriptedModel("script", actions), recorder, protocol=protocol,
                          interaction=interaction)
    recorder.finalize([{k: v for k, v in summary.items() if k != "rows"}],
                      {"n_episodes": 1})
    return summary, recorder


def decision(turn, answer, citations=()):
    return {"turn_id": turn, "answer": answer, "citations": list(citations)}


def test_direct_shows_evidence_and_accepts_cumulative_citation(tmp_path):
    summary, recorder = run(tmp_path, [decision("t1", {"diagnosis": "pain"}, ["e1"]),
                                       decision("t2", {"diagnosis": "infection"}, ["e1", "e2"])])
    assert summary["turns_decided"] == 2
    events = (recorder.run_dir / "events.jsonl").read_text()
    assert "read_evidence" not in events
    assert summary["rows"][1]["decision"]["answer"]["diagnosis"] == "infection"


def test_direct_future_citation_repaired(tmp_path):
    summary, _ = run(tmp_path, [decision("t1", "wrong", ["e2"]), decision("t1", "ok", ["e1"]),
                                decision("t2", "ok", ["e2"])])
    assert summary["rows"][0]["steps"] == 2
    assert summary["rows"][0]["decision"]["answer"] == "ok"


def test_protocol_extension_can_supply_tools(tmp_path):
    from ama.protocols import DirectDecisionProtocol, StepResult
    from ama.model import Message

    class ExampleExtension(DirectDecisionProtocol):
        name = "example_extension"
        tools = [{"type": "function", "function": {"name": "example_tool",
                  "parameters": {"type": "object", "properties": {}}}}]

        def step(self, item, turn, visible, root):
            if item == "tool request":
                return StepResult(messages=[Message(role="user", content="tool result")],
                                  event={"action": "example_tool"})
            return super().step(item, turn, visible, root)

    class ToolAwareModel(ScriptedModel):
        def next(self, messages, tools=None):
            assert tools == ExampleExtension.tools
            return super().next(messages, tools)

    model = ToolAwareModel("extension", ["tool request", decision("t1", "ok"), decision("t2", "ok")])
    summary, recorder = run(tmp_path, [], protocol=ExampleExtension(), model=model)
    assert summary["turns_decided"] == 2
    assert summary["rows"][0]["steps"] == 2
    assert "example_tool" in (recorder.run_dir / "events.jsonl").read_text()


def test_abstain_is_null_answer_without_citations(tmp_path):
    summary, _ = run(tmp_path, [decision("t1", None), decision("t2", None)])
    assert summary["turns_decided"] == 2
    assert summary["rows"][0]["decision"]["answer"] is None


def test_invalid_json_hits_turn_budget_but_replay_advances(tmp_path):
    summary, _ = run(tmp_path, ["not json", decision("t2", "ok")], per_turn=1)
    assert summary["rows"][0]["decision"] is None
    assert summary["rows"][1]["decision"]["answer"] == "ok"


class StopAfterTurn(Interaction):
    def wait_next(self):
        return False


def test_operator_stop(tmp_path):
    summary, _ = run(tmp_path, [decision("t1", "ok")], interaction=StopAfterTurn())
    assert summary["termination"] == "operator_stopped" and len(summary["rows"]) == 1


class CaptureModel:
    name = "capture"
    last_usage = None

    def __init__(self, response, fail_once=False):
        self.response = response
        self.fail_once = fail_once
        self.calls = 0
        self.messages = []

    def next(self, messages, tools=None):
        self.calls += 1
        self.messages = messages[:]
        if self.fail_once and self.calls == 1:
            raise ModelError("temporary failure", kind="network")
        return self.response


def test_model_retry_respects_budget(tmp_path):
    one = {"id": "case", "turns": [EPISODE["turns"][0]]}
    model = CaptureModel(decision("t1", "ok", ["e1"]), fail_once=True)
    summary, recorder = run(tmp_path, [], episode=one, model=model, retries=1)
    assert summary["turns_decided"] == 1 and model.calls == 2
    assert "model_retry" in (recorder.run_dir / "events.jsonl").read_text()


def test_direct_image_is_sent_but_binary_not_logged(tmp_path):
    one = {"id": "case", "turns": [{"id": "t1", "evidence": [
        {"id": "scan", "type": "image", "file": "assets/scan.jpg"}]}]}
    model = CaptureModel(decision("t1", {"caption": "x"}, ["scan"]))
    summary, recorder = run(tmp_path, [], episode=one, model=model,
                            files={"assets/scan.jpg": b"synthetic-image-bytes"})
    assert summary["turns_decided"] == 1
    assert any(p["type"] == "image_url" for p in model.messages[1].content)
    assert "c3ludGhldGljLWltYWdlLWJ5dGVz" not in (recorder.run_dir / "events.jsonl").read_text()
