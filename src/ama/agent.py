"""One replay controller; interaction protocols own all model-facing behavior."""
from __future__ import annotations

import time
from typing import Any, Callable

from .data import Decision, Episode, Evidence, Turn
from .model import Message, ModelClient, ModelError
from .protocols import DirectDecisionProtocol, InteractionProtocol
from .recorder import (Budget, Recorder, merge_usage, redact_content, redact_messages,
                       sanitize_multimodal_content)


class OperatorStopped(Exception):
    pass


class BudgetExceeded(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class Interaction:
    """No-op hooks; the interactive CLI subclasses this."""

    operator_wait_ms: int = 0

    def on_turn_start(self, index: int, total: int, turn: Turn, visible_meta: list[dict]) -> None: ...
    def wait_run(self) -> bool: return True
    def on_step(self, line: str) -> None: ...
    def on_message(self, message: Message) -> None: ...
    def on_turn_end(self, decision: Decision | None, note: str) -> None: ...
    def wait_next(self) -> bool: return True


def _call_model(model: ModelClient, messages: list[Message], tools: list[dict] | None,
                budget: Budget, turn_calls: int, log, episode_id: str, turn_id: str) -> Any:
    last: ModelError | None = None
    for attempt in range(budget.max_retries + 1):
        why = budget.check(turn_calls)
        if why:
            raise BudgetExceeded(why)
        budget.total_calls += 1
        budget.call_seq += 1
        started = time.monotonic()
        try:
            out = model.next(messages, tools=tools)
            log("model_call", episode_id=episode_id, turn_id=turn_id,
                call_id=str(budget.call_seq), model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - started) * 1000),
                usage=getattr(model, "last_usage", None), attempt=attempt, status="ok")
            return out
        except ModelError as exc:
            last = exc
            log("model_call", episode_id=episode_id, turn_id=turn_id,
                call_id=str(budget.call_seq), model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - started) * 1000),
                attempt=attempt, status=f"error:{exc.kind}")
            if attempt < budget.max_retries:
                log("model_retry", episode_id=episode_id, turn_id=turn_id,
                    attempt=attempt + 1, error=f"{exc.kind}: {exc}")
                if exc.kind == "http_error" and "HTTP 429" in str(exc):
                    time.sleep(30 * (attempt + 1))
    assert last is not None
    raise last


def run_episode(episode: Episode, model: ModelClient, recorder: Recorder,
                protocol: InteractionProtocol | None = None, instruction: str = "",
                interaction: Interaction | None = None,
                announce: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Replay fixed observations, collecting one accepted Decision per Turn."""
    driver = protocol or DirectDecisionProtocol()
    interaction = interaction or Interaction()
    budget = recorder.budget
    messages = [Message(role="system", content=driver.system_prompt(episode, instruction))]
    interaction.on_message(messages[0])
    visible: dict[str, Evidence] = {}
    usage_total: dict | None = None
    rows: list[dict[str, Any]] = []
    term = ("completed", "")

    recorder.log("episode_start", episode_id=episode.id, turns=[t.id for t in episode.turns],
                 model_context=[m.model_dump(exclude_none=True) for m in messages])
    try:
        for index, turn in enumerate(episode.turns):
            # 1. 释放本轮材料；此前材料继续可见。
            for ev in turn.evidence:
                visible[ev.id] = ev
            metadata = [{"id": ev.id, "type": ev.type} for ev in turn.evidence]
            recorder.log("turn_start", episode_id=episode.id, turn_id=turn.id,
                         available_at=turn.available_at.isoformat() if turn.available_at else None,
                         released=metadata)
            interaction.on_turn_start(index, len(episode.turns), turn, metadata)
            # 2. 将本轮材料打包成模型消息（含图像）。
            for message in driver.turn_messages(turn, index, len(episode.turns), recorder.dataset_dir, visible):
                messages.append(message)
                interaction.on_message(message)
            started = time.monotonic()
            turn_calls = steps = 0
            turn_usage: dict | None = None
            decision: Decision | None = None
            turn_term = ("completed", "")

            wait_started = time.monotonic()
            if not interaction.wait_run():
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait_started)

            while decision is None:
                # 3. 请求模型并解析答案；格式错误时带反馈重试，受预算限制。
                try:
                    item = _call_model(model, messages, driver.tools, budget, turn_calls,
                                       recorder.log, episode.id, turn.id)
                except BudgetExceeded as exc:
                    turn_term = ("budget_exceeded", exc.detail)
                    break
                except ModelError as exc:
                    turn_term = ("model_error", f"{exc.kind}: {exc}")
                    break
                turn_calls += 1
                turn_usage = merge_usage(turn_usage, getattr(model, "last_usage", None))
                usage_total = merge_usage(usage_total, getattr(model, "last_usage", None))
                result = driver.step(item, turn, visible, recorder.dataset_dir)
                steps += 1
                for message in result.messages:
                    messages.append(message)
                    interaction.on_message(message)
                recorder.log("step", episode_id=episode.id, turn_id=turn.id, step=steps,
                             action=result.event.get("action"),
                             observation=redact_content(result.event.get("observation"),
                                                        recorder.trace.save_evidence_text),
                             error=result.error)
                line = f"step {steps} {result.event.get('action')} -> " + (result.error or "OK")
                interaction.on_step(line)
                if announce:
                    announce(line)
                decision = result.decision

            # 4. 保存答案，再进入下一轮。这里不读取参考答案或计算分数。
            row = {"episode_id": episode.id, "turn_id": turn.id,
                   "termination": turn_term[0], "termination_detail": turn_term[1],
                   "decision": decision.model_dump(mode="json") if decision else None,
                   "model_calls": turn_calls, "steps": steps,
                   "usage": turn_usage if turn_usage is not None else "unknown",
                   "duration_ms": int((time.monotonic() - started) * 1000)}
            recorder.save_decision(row)
            rows.append(row)
            recorder.log("turn_end", episode_id=episode.id, turn_id=turn.id,
                         termination=turn_term[0], termination_detail=turn_term[1],
                         decision=redact_content(row["decision"], recorder.trace.save_evidence_text),
                         model_calls=turn_calls, steps=steps, usage=row["usage"],
                         duration_ms=row["duration_ms"])
            interaction.on_turn_end(decision, turn_term[1])
            if turn_term[0] == "budget_exceeded" and turn_term[1] in ("max_model_calls", "deadline"):
                term = turn_term
                break
            wait_started = time.monotonic()
            if not interaction.wait_next():
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait_started)
    except OperatorStopped:
        term = ("operator_stopped", "operator requested stop")
    except KeyboardInterrupt:
        term = ("operator_stopped", "keyboard interrupt")

    recorder.log("episode_end", episode_id=episode.id, termination=term[0],
                 termination_detail=term[1], total_model_calls=budget.total_calls,
                 usage=usage_total if usage_total is not None else "unknown",
                 operator_wait_ms=interaction.operator_wait_ms,
                 messages=redact_messages(sanitize_multimodal_content(
                     [m.model_dump(exclude_none=True) for m in messages]),
                     recorder.trace.save_model_context))
    return {"episode_id": episode.id, "termination": term[0],
            "termination_detail": term[1], "turns": len(episode.turns),
            "turns_decided": sum(row["decision"] is not None for row in rows),
            "model_calls": budget.total_calls,
            "usage": usage_total if usage_total is not None else "unknown",
            "operator_wait_ms": interaction.operator_wait_ms, "rows": rows}
