"""The agent loop: build context -> model action -> validate -> execute -> log -> repeat.

One model call per step; tool-call messages always come in valid
(assistant action, tool observation) pairs. Invalid actions and tool
validation failures are returned to the model as visible errors and count
against the budget; internal tool failures terminate as tool_error.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from .environment import TimelineEnvironment, summarize_state
from .model import Message, ModelClient, ModelError
from .schemas import (
    Answer,
    BudgetConfig,
    Question,
    StrategyConfig,
    SubmitAnswerAction,
    ToolResult,
    parse_action,
)

LogFn = Callable[..., None]
AnnounceFn = Callable[[str], None]


class BudgetExceededError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _merge_usage(total: dict[str, Any] | None, usage: Any) -> dict[str, Any] | None:
    """Accumulate per-call usage (Gate 0): every API call counts, not just the last."""
    if not isinstance(usage, dict):
        return total
    base = total or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base[key] = base.get(key, 0) + (usage.get(key) or 0)
    return base


@dataclass
class Budget:
    cfg: BudgetConfig
    steps: int = 0
    model_calls: int = 0
    call_seq: int = 0
    started: float = field(default_factory=time.monotonic)

    def exceeded(self) -> str | None:
        if self.steps >= self.cfg.max_steps:
            return "max_steps"
        if self.model_calls >= self.cfg.max_model_calls:
            return "max_model_calls"
        if time.monotonic() - self.started > self.cfg.deadline_seconds:
            return "deadline"
        return None


@dataclass
class QuestionResult:
    termination: str  # completed | budget_exceeded | model_error | tool_error
    termination_detail: str = ""
    answer: Answer | None = None
    steps: int = 0
    model_calls: int = 0
    duration_ms: int = 0
    usage: dict[str, Any] | str = "unknown"


def build_messages(
    env: TimelineEnvironment,
    question: Question,
    strategy: StrategyConfig,
    prior_state_summary: str | None,
) -> list[Message]:
    tools = ["list_evidence", "read_evidence"]
    if env.allow_state_tools:
        tools += ["get_state", "propose_state_update"]
    tools.append("submit_answer")
    lines = [
        "你是病历证据整理 agent。只依据当前可见证据整理事实，不产生诊疗建议，不臆测。",
        f"患者：{env.patient_id}；截至时间 as_of：{env.as_of.isoformat()}。",
        "可见性规则：仅 recorded_time 不晚于 as_of 的该患者证据可见；其他患者的证据、以及尚未入库（recorded_time 晚于 as_of）的证据都不可见，即使其 event_time 更早。",
        f"可用工具：{', '.join(tools)}。每轮恰好调用一个工具（function calling）；最终必须以 submit_answer 结束。",
        "证据不足时提交 abstain=true 的答案并简述原因；引用必须来自已读取的可见证据。",
    ]
    if question.key:
        lines.append(
            f"答案结构要求：submit_answer 中 claim 的 key 必须为 '{question.key}'；"
            "value 必须是单个标量（数字不带单位，或纯文本字符串），不要用对象或数组。"
        )
    if strategy.state_tools_enabled:
        lines.append("状态规则：通过 propose_state_update 维护 claim；修订用新 claim 显式 supersedes 旧 claim；矛盾未被解决时保留两条，不得默认'最新即真'。")
    if prior_state_summary:
        lines.append("此前时间点维护的状态（仅供参考，证据以工具读取为准）：\n" + prior_state_summary)
    return [
        Message(role="system", content="\n".join(lines)),
        Message(role="user", content=question.text),
    ]


def _to_action(item: Any) -> tuple[Any | None, str | None]:
    """Coerce a model reply into a typed action; returns (action, error)."""
    try:
        if isinstance(item, str):
            return parse_action(json.loads(item)), None
        return parse_action(item), None
    except (ValidationError, json.JSONDecodeError) as exc:
        return None, f"invalid action: {exc}"


def run_question(
    env: TimelineEnvironment,
    model: ModelClient,
    question: Question,
    budget_cfg: BudgetConfig,
    log: LogFn,
    strategy: StrategyConfig,
    announce: AnnounceFn | None = None,
    prior_state_summary: str | None = None,
) -> QuestionResult:
    t0 = time.monotonic()
    budget = Budget(budget_cfg)
    messages = build_messages(env, question, strategy, prior_state_summary)
    log("question_start", question_text=question.text, model_context=[m.model_dump() for m in messages])

    termination: tuple[str, str] | None = None
    answer: Answer | None = None
    step = 0
    usage_total: dict[str, Any] | None = None

    while termination is None:
        why = budget.exceeded()
        if why:
            termination = ("budget_exceeded", why)
            break
        try:
            action_item = _call_model(model, messages, budget, log)
        except BudgetExceededError as exc:
            termination = ("budget_exceeded", exc.detail)
            break
        except ModelError as exc:
            termination = ("model_error", f"{exc.kind}: {exc}")
            break
        usage_total = _merge_usage(usage_total, getattr(model, "last_usage", None))

        parsed, parse_error = _to_action(action_item)
        if parsed is None:
            step += 1
            budget.steps = step
            messages.append(Message(role="assistant", content=str(action_item)[:500]))
            messages.append(Message(
                role="user",
                content=f"INVALID ACTION: {parse_error}\n请重新发出一个合法的工具调用（function call）。",
            ))
            log("step", step=step, action=None, raw=str(action_item)[:200],
                observation={"ok": False, "error": parse_error})
            if announce:
                announce(f"step {step} INVALID -> {parse_error}")
            continue

        call_id = f"call_{step + 1}"

        if isinstance(parsed, SubmitAnswerAction) and parsed.answer.question_id != question.question_id:
            step += 1
            budget.steps = step
            obs = {"ok": False, "error": f"answer.question_id '{parsed.answer.question_id}' does not match current question '{question.question_id}'"}
            messages.append(Message(
                role="assistant",
                content=json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False),
                tool_calls=[_tool_call(call_id, parsed)],
            ))
            messages.append(Message(role="tool", tool_call_id=call_id,
                                    content=json.dumps(obs, ensure_ascii=False)))
            log("step", step=step, action=parsed.model_dump(mode="json"), observation=obs)
            if announce:
                announce(f"step {step} {parsed.type} -> REJECTED (question mismatch)")
            continue

        try:
            result = env.execute(parsed)
        except Exception as exc:  # belt-and-suspenders: tool crash terminates as tool_error
            result = ToolResult(ok=False, error=f"tool crashed: {exc!r}", error_kind="internal")
        step += 1
        budget.steps = step
        log(
            "step",
            step=step,
            action=parsed.model_dump(mode="json"),
            observation=result.public(),
            violation=result.violation,
        )
        messages.append(Message(
            role="assistant",
            content=json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False),
            tool_calls=[_tool_call(call_id, parsed)],
        ))
        messages.append(Message(role="tool", tool_call_id=call_id,
                                content=json.dumps(result.public(), ensure_ascii=False)))
        if announce:
            status = "OK" if result.ok else f"ERR {result.error}"
            announce(f"step {step} {parsed.type} -> {status}")

        if result.error_kind == "internal":
            termination = ("tool_error", result.error or "internal")
            break
        if isinstance(parsed, SubmitAnswerAction) and result.ok:
            answer = parsed.answer
            termination = ("completed", "")
            break

    duration_ms = int((time.monotonic() - t0) * 1000)
    usage = usage_total if isinstance(usage_total, dict) else "unknown"
    log(
        "question_end",
        termination=termination[0],
        termination_detail=termination[1],
        answer=answer.model_dump(mode="json") if answer else None,
        steps=step,
        model_calls=budget.model_calls,
        duration_ms=duration_ms,
        usage=usage,
        messages=[m.model_dump(exclude_none=True) for m in messages],
    )
    return QuestionResult(
        termination=termination[0],
        termination_detail=termination[1],
        answer=answer,
        steps=step,
        model_calls=budget.model_calls,
        duration_ms=duration_ms,
        usage=usage,
    )


def _tool_call(call_id: str, action: Any) -> dict[str, Any]:
    """OpenAI-style tool_call envelope around an internal action."""
    args = action.model_dump(exclude={"type"}, mode="json")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": action.type, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _call_model(model: ModelClient, messages: list[Message], budget: Budget, log: LogFn) -> Any:
    last: ModelError | None = None
    for attempt in range(budget.cfg.max_retries + 1):
        why = budget.exceeded()  # Gate 0: retries also honor max_model_calls and deadline
        if why:
            raise BudgetExceededError(why)
        budget.model_calls += 1
        call_id = f"{budget.call_seq + 1}"
        budget.call_seq += 1
        t0 = time.monotonic()
        try:
            out = model.next(messages)
            log(
                "model_call",
                call_id=call_id,
                model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000),
                usage=getattr(model, "last_usage", None),
                attempt=attempt,
                status="ok",
            )
            return out
        except ModelError as exc:
            last = exc
            log(
                "model_call",
                call_id=call_id,
                model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000),
                usage=None,
                attempt=attempt,
                status=f"error:{exc.kind}",
            )
            if attempt < budget.cfg.max_retries:
                log("model_retry", attempt=attempt + 1, error=f"{exc.kind}: {exc}")
    raise last  # type: ignore[misc]
