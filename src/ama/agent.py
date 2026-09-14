"""The ONE agent loop: every workload (single-turn, multi-turn, batch, interactive)
runs through run_episode(). Budget, usage, retries, trace and interactive hooks are
cross-cutting concerns of this single function — there is no second runner."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from .data import Decision, Episode, Turn
from .model import Message, ModelClient, ModelError, parse_action
from .recorder import Budget, Recorder, merge_usage, redact_content, redact_messages


class OperatorStopped(Exception):
    pass


class Interaction:
    """No-op hooks; the interactive CLI subclasses this. Same loop for batch."""

    operator_wait_ms: int = 0

    def on_turn_start(self, index: int, total: int, turn: Turn, visible_meta: list[dict]) -> None: ...

    def wait_run(self) -> bool:
        return True

    def on_step(self, line: str) -> None: ...

    def on_turn_end(self, decision: Decision | None, note: str) -> None: ...

    def wait_next(self) -> bool:
        return True


def system_prompt(episode: Episode, public_policy: dict | None) -> str:
    lines = [
        "你是研究运行器中的被测 agent，工作在 teacher-forced 回放环境：每轮会到达一条新的 world observation（user 消息），你的动作不会改变后续已记录的观察。",
        f"主体：{episode.subject_id}。共 {len(episode.turns)} 轮。",
        "每轮流程：可用 list_evidence / read_evidence 查看证据，然后恰好调用一次 submit_decision 提交本轮决策。",
        "可见性：第 N 轮可见 = 前N轮释放的全部证据；尚未释放的证据不可见（读取会失败）。",
        "decision.turn_id 必须逐字符等于本轮 world observation 标注的 turn_id（如 t1、t2）。",
        "引用约束：decision.citations 必须来自此前已成功 read_evidence 的证据；证据不足时 abstain=true（不得有 action/citations/state 内容）。",
    ]
    if public_policy:
        guidance = public_policy.get("guidance")
        if guidance:
            lines.append(str(guidance))
    return "\n".join(lines)


def world_message(index: int, total: int, turn: Turn, new_ids: list[str]) -> str:
    ids = ", ".join(new_ids) or "（无新增）"
    when = turn.time.isoformat() if turn.time else "时间未记录"
    return (f"[turn_id={turn.turn_id} | 第 {index + 1}/{total} 轮 world observation @ {when}]\n"
            f"{turn.message}\n本轮新增证据（已可读取）：{ids}")


def _tool_call(call_id: str, action: Any) -> dict[str, Any]:
    args = action.model_dump(exclude={"type"}, mode="json")
    return {"id": call_id, "type": "function",
            "function": {"name": action.type, "arguments": json.dumps(args, ensure_ascii=False)}}


def _to_action(item: Any) -> tuple[Any | None, str | None]:
    try:
        if isinstance(item, str):
            return parse_action(json.loads(item)), None
        return parse_action(item), None
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid action: {exc}"


def _call_model(model, messages, budget: Budget, turn_calls: int, log, episode_id: str, turn_id: str):
    last: ModelError | None = None
    for attempt in range(budget.max_retries + 1):
        why = budget.check(turn_calls)  # retries also honor call budgets and deadline
        if why:
            raise BudgetExceeded(why)
        budget.total_calls += 1
        budget.call_seq += 1
        t0 = time.monotonic()
        try:
            out = model.next(messages)
            log("model_call", episode_id=episode_id, turn_id=turn_id, call_id=str(budget.call_seq),
                model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000),
                usage=getattr(model, "last_usage", None), attempt=attempt, status="ok")
            return out
        except ModelError as exc:
            last = exc
            log("model_call", episode_id=episode_id, turn_id=turn_id, call_id=str(budget.call_seq),
                model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000), usage=None,
                attempt=attempt, status=f"error:{exc.kind}")
            if attempt < budget.max_retries:
                log("model_retry", episode_id=episode_id, turn_id=turn_id,
                    attempt=attempt + 1, error=f"{exc.kind}: {exc}")
    raise last


class BudgetExceeded(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def run_episode(
    episode: Episode,
    model: ModelClient,
    recorder: Recorder,
    public_policy: dict | None = None,
    interaction: Interaction | None = None,
    announce: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Returns the episode summary row (decisions already saved via recorder)."""
    interaction = interaction or Interaction()
    budget = recorder.budget
    messages: list[Message] = [Message(role="system", content=system_prompt(episode, public_policy))]
    released: dict[str, Any] = {}   # evidence_id -> evidence dict (cumulative visibility)
    read_ids: set[str] = set()
    usage_total: dict | None = None

    recorder.log("episode_start", episode_id=episode.episode_id, subject_id=episode.subject_id,
                 turns=[t.turn_id for t in episode.turns],
                 model_context=[m.model_dump(exclude_none=True) for m in messages])
    rows: list[dict[str, Any]] = []
    term = ("completed", "")
    prev_missed = False

    try:
        for idx, turn in enumerate(episode.turns):
            if term[0] != "completed":
                break
            for ev in turn.evidence:
                released[ev.evidence_id] = ev.model_dump(mode="json")
            new_meta = [{"evidence_id": e.evidence_id, "kind": e.kind, "source": e.source} for e in turn.evidence]
            recorder.log("turn_start", episode_id=episode.episode_id, turn_id=turn.turn_id,
                         time=turn.time.isoformat() if turn.time else None, released=new_meta)
            interaction.on_turn_start(idx, len(episode.turns), turn, new_meta)

            if prev_missed:
                messages.append(Message(role="user", content="（系统提示：上一轮未收到合法 submit_decision，已按真实轨迹推进到本轮。）"))
            messages.append(Message(role="user", content=world_message(idx, len(episode.turns), turn,
                                                                       [e.evidence_id for e in turn.evidence])))
            t0 = time.monotonic()
            turn_calls = 0
            step = 0
            decision: Decision | None = None
            turn_usage: dict | None = None
            turn_term = ("completed", "")

            wait0 = time.monotonic()
            if not interaction.wait_run():  # one operator approval per turn
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait0)

            while decision is None:
                try:
                    item = _call_model(model, messages, budget, turn_calls, recorder.log,
                                       episode.episode_id, turn.turn_id)
                except BudgetExceeded as exc:
                    turn_term = ("budget_exceeded", exc.detail)
                    break
                except ModelError as exc:
                    turn_term = ("model_error", f"{exc.kind}: {exc}")
                    break
                turn_calls += 1
                usage_total = merge_usage(usage_total, getattr(model, "last_usage", None))
                turn_usage = merge_usage(turn_usage, getattr(model, "last_usage", None))

                action, err = _to_action(item)
                if action is None:
                    step += 1
                    messages.append(Message(role="assistant", content=str(item)[:500]))
                    messages.append(Message(role="user",
                                            content=f"INVALID ACTION: {err}\n请重新发出一个合法工具调用（function call）。"))
                    recorder.log("step", episode_id=episode.episode_id, turn_id=turn.turn_id, step=step,
                                 action=None, raw=str(item)[:200], observation={"ok": False, "error": err})
                    line = f"step {step} INVALID -> {err}"
                    interaction.on_step(line)
                    if announce:
                        announce(line)
                    continue

                call_id = f"call_{step + 1}"
                obs = _execute(action, released, read_ids, turn)
                step += 1
                action_pub = {"type": action.type, "params": action.model_dump(exclude={"type"}, mode="json")}
                recorder.log("step", episode_id=episode.episode_id, turn_id=turn.turn_id, step=step,
                             action=action_pub,
                             observation=redact_content({k: v for k, v in obs.items() if k != "violation"},
                                                        recorder.trace.save_evidence_text),
                             violation=obs.get("violation"))
                messages.append(Message(role="assistant", content=json.dumps(action_pub, ensure_ascii=False),
                                        tool_calls=[_tool_call(call_id, action)]))
                obs_public = {k: v for k, v in obs.items() if k != "violation"}
                messages.append(Message(role="tool", tool_call_id=call_id,
                                        content=json.dumps(obs_public, ensure_ascii=False, default=str)))
                line = f"step {step} {action.type} -> " + ("OK" if obs.get("ok") else f"ERR {obs.get('error')}")
                interaction.on_step(line)
                if announce:
                    announce(line)

                if obs.get("error_kind") == "internal":
                    turn_term = ("tool_error", str(obs.get("error")))
                    break
                if action.type == "submit_decision" and obs.get("ok"):
                    decision = action.decision
                    break

            if decision is not None:
                prev_missed = False
            else:
                prev_missed = True

            row = {
                "episode_id": episode.episode_id,
                "turn_id": turn.turn_id,
                "termination": turn_term[0],
                "termination_detail": turn_term[1],
                "decision": decision.model_dump(mode="json") if decision else None,
                "read_ids": sorted(read_ids),
                "model_calls": turn_calls,
                "steps": step,
                "usage": turn_usage if isinstance(turn_usage, dict) else "unknown",
                "duration_ms": int((time.monotonic() - t0) * 1000),
            }
            recorder.save_decision(row)
            rows.append(row)
            recorder.log("turn_end", episode_id=episode.episode_id, turn_id=turn.turn_id,
                         termination=turn_term[0], termination_detail=turn_term[1],
                         decision=redact_content(row["decision"], recorder.trace.save_evidence_text),
                         model_calls=turn_calls, steps=step, usage=row["usage"],
                         duration_ms=row["duration_ms"])
            interaction.on_turn_end(decision, turn_term[1])

            if turn_term[0] == "budget_exceeded" and turn_term[1] in ("max_model_calls", "deadline"):
                term = ("budget_exceeded", turn_term[1])
                break
            if turn_term[0] in ("model_error", "tool_error"):
                recorder.log("episode_note", episode_id=episode.episode_id, turn_id=turn.turn_id,
                             note=f"turn failed ({turn_term[0]}); teacher-forced replay continues")

            wait0 = time.monotonic()
            if not interaction.wait_next():
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait0)
    except OperatorStopped:
        term = ("operator_stopped", "operator requested stop")
    except KeyboardInterrupt:
        term = ("operator_stopped", "keyboard interrupt")

    recorder.log("episode_end", episode_id=episode.episode_id, termination=term[0],
                 termination_detail=term[1], total_model_calls=budget.total_calls,
                 usage=usage_total if isinstance(usage_total, dict) else "unknown",
                 operator_wait_ms=interaction.operator_wait_ms,
                 messages=redact_messages([m.model_dump(exclude_none=True) for m in messages],
                                          recorder.trace.save_model_context))
    return {
        "episode_id": episode.episode_id,
        "termination": term[0],
        "termination_detail": term[1],
        "turns": len(episode.turns),
        "turns_decided": sum(1 for r in rows if r["decision"]),
        "model_calls": budget.total_calls,
        "usage": usage_total if isinstance(usage_total, dict) else "unknown",
        "operator_wait_ms": interaction.operator_wait_ms,
        "rows": rows,
    }


def _execute(action: Any, released: dict[str, Any], read_ids: set[str], turn: Turn) -> dict[str, Any]:
    """The three tools. Visibility = released-so-far; citations must come from read set."""
    try:
        if action.type == "list_evidence":
            items = [{"evidence_id": eid, "kind": e["kind"], "source": e["source"]}
                     for eid, e in released.items()]
            return {"ok": True, "data": {"evidence": items, "count": len(items)}}
        if action.type == "read_evidence":
            ev = released.get(action.evidence_id)
            if ev is None:
                return {"ok": False, "error": "evidence_id not found among visible evidence",
                        "error_kind": "validation", "violation": "future"}
            read_ids.add(action.evidence_id)
            return {"ok": True, "data": ev}
        if action.type == "submit_decision":
            d: Decision = action.decision
            if d.turn_id != turn.turn_id:
                return {"ok": False,
                        "error": f"decision.turn_id '{d.turn_id}' does not match current turn '{turn.turn_id}'",
                        "error_kind": "validation"}
            if d.abstain and (d.action is not None or d.citations or d.state):
                return {"ok": False, "error": "abstain decision must not include action/citations/state",
                        "error_kind": "validation"}
            for ref in d.citations:
                if ref not in released:
                    return {"ok": False, "error": f"citation '{ref}' is not visible evidence",
                            "error_kind": "validation", "violation": "future"}
                if ref not in read_ids:
                    return {"ok": False,
                            "error": f"citation '{ref}' visible but not read; call read_evidence before citing",
                            "error_kind": "validation"}
            return {"ok": True, "data": {"accepted": True}}
        return {"ok": False, "error": f"unsupported action {action.type}", "error_kind": "internal"}
    except Exception as exc:
        return {"ok": False, "error": f"internal error: {exc!r}", "error_kind": "internal"}
