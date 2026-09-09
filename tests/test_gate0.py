"""Gate 0 measurement fixes: per-call usage accumulation, retry-vs-budget,
read-before-cite (env level covered in test_environment), manifest provenance."""
import json

from medical_harness.agent import run_question
from medical_harness.environment import TimelineEnvironment
from medical_harness.model import ModelError
from medical_harness.runner import run as run_harness
from medical_harness.schemas import (
    Answer, AnswerClaim, BudgetConfig, ListEvidenceAction, Question, StrategyConfig, SubmitAnswerAction,
)
from conftest import DATA
from datetime import datetime


class UsageModel:
    """Emits actions while reporting fake per-call token usage."""

    def __init__(self, actions, per_call_usage):
        self.name = "usage-fake"
        self.last_usage = None
        self._actions = list(actions)
        self._per_call = per_call_usage
        self._n = 0

    def next(self, messages):
        if not self._actions:
            raise ModelError("exhausted")
        self._n += 1
        self.last_usage = {"prompt_tokens": self._per_call, "completion_tokens": 1,
                           "total_tokens": self._per_call + 1}
        item = self._actions.pop(0)
        return item() if callable(item) else item


def test_usage_accumulates_across_all_calls(inputs):
    model = UsageModel([
        ListEvidenceAction(),
        SubmitAnswerAction(answer=Answer(
            question_id="q1", claims=[AnswerClaim(key="medication", value="aspirin 100mg daily")],
        )),
    ], per_call_usage=100)
    env = TimelineEnvironment(inputs["case_alpha"], as_of=datetime.fromisoformat("2024-03-01T12:00:00+00:00"))
    events = []
    res = run_question(env, model, Question(question_id="q1", text="t"),
                       BudgetConfig(max_steps=5, max_model_calls=5, max_retries=0),
                       lambda t, **f: events.append({"type": t, **f}), StrategyConfig())
    assert res.termination == "completed"
    assert res.usage == {"prompt_tokens": 200, "completion_tokens": 2, "total_tokens": 202}  # 2 calls summed
    calls = [e for e in events if e["type"] == "model_call"]
    assert [c["call_id"] for c in calls] == ["1", "2"]
    assert all(c["latency_ms"] is not None and c["model"] == "usage-fake" for c in calls)
    end = [e for e in events if e["type"] == "question_end"][0]
    assert end["usage"]["total_tokens"] == 202


class AlwaysErrorModel:
    def __init__(self):
        self.name = "always-error"

    def next(self, messages):
        raise ModelError("boom", kind="http_error")


def test_retries_cannot_exceed_model_call_budget(inputs):
    env = TimelineEnvironment(inputs["case_alpha"], as_of=datetime.fromisoformat("2024-03-01T12:00:00+00:00"))
    events = []
    res = run_question(env, AlwaysErrorModel(), Question(question_id="q1", text="t"),
                       BudgetConfig(max_steps=10, max_model_calls=2, max_retries=5, deadline_seconds=30),
                       lambda t, **f: events.append({"type": t, **f}), StrategyConfig())
    assert res.termination == "budget_exceeded"  # not model_error: budget stopped the retry loop
    assert res.model_calls == 2
    assert sum(1 for e in events if e["type"] == "model_retry") == 2  # bounded: budget stops further retries


def test_manifest_records_git_and_dependencies(tmp_path):
    cfg = {
        "name": "manifest-check",
        "data_dir": str(DATA / "inputs"),
        "model": {"type": "scripted", "script_dir": str(DATA / "scripts")},
        "budget": {"max_steps": 6, "max_model_calls": 6, "deadline_seconds": 60},
        "cases": ["case_beta"],
        "experiment": {"question": "manifest provenance check", "primary_metric": "field_correct"},
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    run_dir = run_harness(p, tmp_path / "runs")
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["code_version"]["vcs"] == "git"
    assert resolved["code_version"]["git_commit"] and len(resolved["code_version"]["git_commit"]) == 12
    assert "pydantic" in resolved["dependencies"]
    assert resolved["experiment"]["primary_metric"] == "field_correct"
    first_event = json.loads((run_dir / "events.jsonl").read_text().splitlines()[0])
    assert first_event["run_id"] == run_dir.name
