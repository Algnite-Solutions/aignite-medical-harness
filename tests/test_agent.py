"""The ONE loop: visibility, read-before-cite, abstain, budgets, retries, usage,
teacher-forced continuation, operator stop."""
import json
from pathlib import Path

from ama.agent import Interaction, OperatorStopped, run_episode
from ama.data import Episode
from ama.model import ListEvidence, ModelError, ReadEvidence, SubmitDecision
from ama.recorder import Budget, Recorder, TraceConfig

EP = Episode.model_validate({
    "episode_id": "ep1", "subject_id": "s1", "metadata": {},
    "turns": [
        {"turn_id": "t1", "time": "2026-01-01T00:00:00+00:00", "message": "m1",
         "evidence": [{"evidence_id": "a1", "kind": "lab", "text": "肌酐 1.2", "artifact": None,
                       "source": "s", "metadata": {}}]},
        {"turn_id": "t2", "time": "2026-01-02T00:00:00+00:00", "message": "m2",
         "evidence": [{"evidence_id": "b1", "kind": "note", "text": "病程记录", "artifact": None,
                       "source": "s", "metadata": {}}]},
        {"turn_id": "t3", "time": "2026-01-03T00:00:00+00:00", "message": "m3", "evidence": []},
    ],
})


class Scripted:
    def __init__(self, actions):
        self.name = "scripted-test"
        self.last_usage = None
        self._it = iter(actions)

    def next(self, messages):
        return next(self._it)


def submit(turn, **kw):
    d = {"turn_id": turn, "state": {}, "action": None, "citations": [], "abstain": False, "note": ""}
    d.update(kw)
    return SubmitDecision(type="submit_decision", decision=d)


def run(actions, budget=None, trace=None, interaction=None):
    budget = budget or Budget(max_model_calls=40, per_turn_model_calls=15,
                              deadline_seconds=60, max_retries=0)
    recorder = Recorder(Path("/tmp/ama-test-runs"), "test", "scripted-test",
                        Path("datasets"), ["ep1"], trace or TraceConfig(), budget)
    result = run_episode(EP, Scripted(actions), recorder, interaction=interaction)
    events = [json.loads(l) for l in (recorder.run_dir / "events.jsonl").read_text().splitlines()]
    decisions = [json.loads(l) for l in (recorder.run_dir / "decisions.jsonl").read_text().splitlines()]
    recorder.finalize([], {"n_episodes": 1})
    return result, events, decisions


def test_world_message_discloses_turn_id():
    from ama.agent import world_message
    msg = world_message(0, 3, EP.turns[0], ["a1"])
    assert "turn_id=t1" in msg  # the model must be told the exact string it must echo


def test_progressive_visibility_and_read_before_cite():
    result, events, _ = run([
        ListEvidence(),  # t1: only a1
        ReadEvidence(evidence_id="a1"),
        submit("t1"),
        ReadEvidence(evidence_id="b1"),  # t2: b1 released, a1 still visible
        ListEvidence(),
        submit("t2", citations=["a1"]),  # citing evidence read in an EARLIER turn is fine
        submit("t3"),
    ])
    assert result["termination"] == "completed" and result["turns_decided"] == 3
    lists = [e for e in events if e["type"] == "step" and e["action"]["type"] == "list_evidence"]
    ids1 = [x["evidence_id"] for x in lists[0]["observation"]["data"]["evidence"]]
    ids2 = [x["evidence_id"] for x in lists[1]["observation"]["data"]["evidence"]]
    assert ids1 == ["a1"] and ids2 == ["a1", "b1"]


def test_unreleased_evidence_invisible_with_internal_tag():
    result, events, _ = run([
        ReadEvidence(evidence_id="b1"),  # not yet released at t1
        ReadEvidence(evidence_id="a1"),
        submit("t1"),
        submit("t2"),
        submit("t3"),
    ])
    reads = [e for e in events if e["type"] == "step" and e["action"]["type"] == "read_evidence"]
    assert reads[0]["observation"]["ok"] is False
    assert reads[0]["violation"] == "future"
    assert "violation" not in json.dumps(reads[0]["observation"])  # never shown to the model


def test_visible_but_unread_citation_rejected_then_accepted():
    result, events, _ = run([
        ReadEvidence(evidence_id="a1"),
        submit("t1", citations=["a1"]),
        submit("t2", citations=["b1"]),  # visible but unread -> rejected, consumed
        ReadEvidence(evidence_id="b1"),
        submit("t2", citations=["b1"]),
        submit("t3"),
    ])
    t2_steps = [e for e in events if e["type"] == "step" and e.get("turn_id") == "t2"
                and e["action"]["type"] == "submit_decision"]
    assert t2_steps[0]["observation"]["ok"] is False and "not read" in t2_steps[0]["observation"]["error"]
    assert t2_steps[1]["observation"]["ok"] is True
    assert result["turns_decided"] == 3


def test_abstain_and_turn_mismatch_validation():
    result, events, _ = run([
        submit("t1", abstain=True, state={"x": 1}),  # abstain with state -> rejected
        submit("t1", abstain=True),
        submit("t9"),  # wrong turn id -> rejected
        submit("t2"),
        submit("t3"),
    ])
    steps = [e for e in events if e["type"] == "step" and e["action"]["type"] == "submit_decision"]
    assert "abstain" in steps[0]["observation"]["error"]        # abstain + state -> rejected
    assert steps[1]["observation"]["ok"] is True                # clean abstain accepted, ends t1
    assert "does not match" in steps[2]["observation"]["error"]  # wrong turn id rejected in t2


def test_per_turn_budget_records_and_continues():
    result, _, decisions = run([
        ListEvidence(),
        submit("t1"),
        submit("t2"),
        submit("t3"),
    ], budget=Budget(max_model_calls=40, per_turn_model_calls=1, deadline_seconds=60, max_retries=0))
    # t2's single call was a plain submit that got rejected? no: per_turn=1 -> one call, decision accepted
    assert result["termination"] == "completed"


def test_episode_budget_exhaustion():
    result, _, _ = run([
        ListEvidence(), ReadEvidence(evidence_id="a1"), submit("t1"),
    ], budget=Budget(max_model_calls=2, per_turn_model_calls=10, deadline_seconds=60, max_retries=0))
    assert result["termination"] == "budget_exceeded"
    assert result["model_calls"] == 2


def test_model_error_turn_continues_teacher_forced():
    class FlakyT2(Scripted):
        def next(self, messages):
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "第 2/3 轮" in last_user:
                raise ModelError("injected", kind="injected")
            return super().next(messages)

    budget = Budget(max_model_calls=40, per_turn_model_calls=15, deadline_seconds=60, max_retries=0)
    recorder = Recorder(Path("/tmp/ama-test-runs"), "t2", "flaky", Path("datasets"), ["ep1"],
                        TraceConfig(), budget)
    result = run_episode(EP, FlakyT2([submit("t1"), submit("t3")]), recorder)
    assert result["termination"] == "completed" and result["turns_decided"] == 2
    rows = {r["turn_id"]: r for r in recorder.decisions}
    assert rows["t2"]["termination"] == "model_error" and rows["t2"]["decision"] is None


def test_usage_accumulates_per_call():
    class UsageModel(Scripted):
        def __init__(self, actions):
            super().__init__(actions)
            self._n = 0

        def next(self, messages):
            out = super().next(messages)
            self._n += 1
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            return out

    budget = Budget(max_model_calls=40, per_turn_model_calls=15, deadline_seconds=60, max_retries=0)
    recorder = Recorder(Path("/tmp/ama-test-runs"), "u", "usage", Path("datasets"), ["ep1"],
                        TraceConfig(), budget)
    result = run_episode(EP, UsageModel([ListEvidence(), submit("t1"), submit("t2"), submit("t3")]), recorder)
    assert result["usage"]["total_tokens"] == 4 * 11
    calls = [e for e in (json.loads(l) for l in
                         (recorder.run_dir / "events.jsonl").read_text().splitlines())
             if e["type"] == "model_call"]
    assert [c["call_id"] for c in calls] == ["1", "2", "3", "4"]
    assert all(c["latency_ms"] is not None for c in calls)


def test_retry_cannot_exceed_budget():
    class AlwaysErr:
        name = "err"
        last_usage = None

        def next(self, messages):
            raise ModelError("boom", kind="http_error")

    budget = Budget(max_model_calls=2, per_turn_model_calls=10, deadline_seconds=60, max_retries=5)
    recorder = Recorder(Path("/tmp/ama-test-runs"), "r", "err", Path("datasets"), ["ep1"],
                        TraceConfig(), budget)
    result = run_episode(EP, AlwaysErr(), recorder)
    assert result["termination"] == "budget_exceeded"
    assert result["model_calls"] == 2


def test_operator_stop_preserves_run():
    class StopAfterFirst(Interaction):
        def wait_next(self) -> bool:
            return False

    result, _, decisions = run([submit("t1")], interaction=StopAfterFirst())
    assert result["termination"] == "operator_stopped"
    assert len(decisions) == 1  # artifacts complete and flushed


def test_invalid_action_gets_visible_feedback():
    result, events, _ = run([
        "not json",
        submit("t1"), submit("t2"), submit("t3"),
    ])
    invalid = [e for e in events if e["type"] == "step" and e.get("action") is None]
    assert invalid and "invalid action" in invalid[0]["observation"]["error"]
    end = [e for e in events if e["type"] == "episode_end"][0]
    contents = json.dumps(end["messages"])
    assert "INVALID ACTION" in contents


def test_trace_redaction():
    result, events, _ = run([
        ReadEvidence(evidence_id="a1"), submit("t1"), submit("t2"), submit("t3"),
    ], trace=TraceConfig(save_model_context=False, save_evidence_text=False))
    reads = [e for e in events if e["type"] == "step" and e["action"]["type"] == "read_evidence"]
    assert reads[0]["observation"]["data"]["text"].startswith("<redacted")
    end = [e for e in events if e["type"] == "episode_end"][0]
    assert isinstance(end["messages"], str) and end["messages"].startswith("<redacted")
