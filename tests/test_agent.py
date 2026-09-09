"""Loop tests: budgets, invalid actions, model errors/timeouts, tool crashes,
message pairing, and complete logging."""
from medical_harness.agent import run_question
from medical_harness.environment import TimelineEnvironment
from medical_harness.model import ModelError, ModelTimeoutError
from medical_harness.schemas import (
    Answer,
    AnswerClaim,
    BudgetConfig,
    ListEvidenceAction,
    Question,
    StrategyConfig,
    SubmitAnswerAction,
)
from datetime import datetime


class FixedModel:
    """Emits a fixed sequence; callable entries are invoked (may raise)."""

    def __init__(self, items):
        self.name = "fixed"
        self.items = list(items)

    def next(self, messages):
        if not self.items:
            raise ModelError("script exhausted", kind="exhausted")
        item = self.items.pop(0)
        return item() if callable(item) else item


def make_env(inputs, as_of="2024-03-01T12:00:00+00:00"):
    return TimelineEnvironment(inputs["case_alpha"], as_of=datetime.fromisoformat(as_of))


def run(env, model, qid="q1", budget=None):
    events = []

    def log(type_, **fields):
        events.append({"type": type_, **fields})

    res = run_question(
        env, model, Question(question_id=qid, text="t"),
        budget or BudgetConfig(max_steps=5, max_model_calls=5, max_retries=0, deadline_seconds=30),
        log, StrategyConfig(),
    )
    return res, events


def submit(qid="q1", **kw):
    return SubmitAnswerAction(answer=Answer(question_id=qid, **kw))


def test_budget_exceeded_terminates_and_logs(inputs):
    model = FixedModel([lambda: ListEvidenceAction()] * 50)
    res, events = run(make_env(inputs), model, budget=BudgetConfig(max_steps=3, max_model_calls=10, max_retries=0))
    assert res.termination == "budget_exceeded" and res.termination_detail == "max_steps"
    assert res.steps == 3 and res.answer is None
    assert sum(1 for e in events if e["type"] == "step") == 3
    end = [e for e in events if e["type"] == "question_end"][0]
    assert end["termination"] == "budget_exceeded" and end["answer"] is None


def test_model_call_budget_exceeded(inputs):
    model = FixedModel([lambda: ListEvidenceAction()] * 50)
    res, _ = run(make_env(inputs), model,
                 budget=BudgetConfig(max_steps=10, max_model_calls=2, max_retries=0))
    assert res.termination == "budget_exceeded" and res.termination_detail == "max_model_calls"


def test_invalid_actions_are_visible_errors_and_do_not_hang(inputs):
    model = FixedModel([
        "this is not json",
        '{"type": "unknown_tool"}',
        submit(claims=[AnswerClaim(key="medication", value="aspirin 100mg daily")]),
    ])
    res, events = run(make_env(inputs), model, budget=BudgetConfig(max_steps=6, max_model_calls=6, max_retries=0))
    assert res.termination == "completed"
    steps = [e for e in events if e["type"] == "step"]
    assert steps[0]["observation"]["ok"] is False and "invalid action" in steps[0]["observation"]["error"]
    assert steps[1]["observation"]["ok"] is False and "invalid action" in steps[1]["observation"]["error"]
    assert steps[2]["observation"]["ok"] is True


def test_model_error_terminates_with_complete_log(inputs):
    res, events = run(make_env(inputs), FixedModel([]))
    assert res.termination == "model_error" and "exhausted" in res.termination_detail
    end = [e for e in events if e["type"] == "question_end"][0]
    assert end["termination"] == "model_error" and end["steps"] == 0


def test_model_timeout_terminates_and_retries_logged(inputs):
    def boom():
        raise ModelTimeoutError()

    res, events = run(make_env(inputs), FixedModel([boom, boom]),
                      budget=BudgetConfig(max_steps=3, max_model_calls=3, max_retries=1, deadline_seconds=30))
    assert res.termination == "model_error" and "timeout" in res.termination_detail
    assert any(e["type"] == "model_retry" for e in events)  # bounded retry, logged
    assert res.model_calls == 2  # 1 attempt + 1 retry


def test_tool_crash_terminates_as_tool_error(inputs, monkeypatch):
    env = make_env(inputs)

    def crash(action):
        raise RuntimeError("boom")

    monkeypatch.setattr(env, "execute", crash)
    res, events = run(env, FixedModel([ListEvidenceAction()]))
    assert res.termination == "tool_error" and "boom" in res.termination_detail
    assert [e for e in events if e["type"] == "question_end"][0]["termination"] == "tool_error"


def test_messages_keep_tool_call_pairing(inputs):
    model = FixedModel([
        ListEvidenceAction(),
        submit(claims=[AnswerClaim(key="medication", value="aspirin 100mg daily")]),
    ])
    res, events = run(make_env(inputs), model)
    assert res.termination == "completed"
    msgs = [e for e in events if e["type"] == "question_end"][0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            if m.get("tool_calls"):  # valid protocol pairing with matching ids
                assert msgs[i + 1]["role"] == "tool"
                assert msgs[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]
            else:
                assert msgs[i + 1]["role"] == "user"  # invalid-action feedback


def test_invalid_action_feedback_is_user_message(inputs):
    model = FixedModel([
        "garbage",
        submit(claims=[AnswerClaim(key="medication", value="aspirin 100mg daily")]),
    ])
    res, events = run(make_env(inputs), model)
    assert res.termination == "completed"
    msgs = [e for e in events if e["type"] == "question_end"][0]["messages"]
    assert msgs[2]["role"] == "assistant" and not msgs[2].get("tool_calls")
    assert msgs[3]["role"] == "user" and "INVALID ACTION" in msgs[3]["content"]


def test_submit_for_wrong_question_rejected_visibly(inputs):
    model = FixedModel([submit(qid="q9")])  # current question is q1
    res, events = run(make_env(inputs), model, qid="q1")
    assert res.termination == "model_error"  # script exhausted after rejection
    step = [e for e in events if e["type"] == "step"][0]
    assert step["observation"]["ok"] is False and "does not match" in step["observation"]["error"]
