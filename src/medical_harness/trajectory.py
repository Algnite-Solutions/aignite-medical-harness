"""Multi-turn clinical trajectory (v0.2): teacher-forced episode replay.

One patient = one episode = one persistent model conversation. Each turn
releases the next recorded observations (world message); the agent reads
evidence and submits exactly one TurnDecision; a deterministic workflow
validator scores state/action/transition; the episode then advances along the
RECORDED trajectory regardless of agent mistakes (replay, not closed loop).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .agent import BudgetExceededError
from .environment import EpisodeEnvironment
from .model import Message, ModelClient, ModelError, make_episode_model_factory
from .runner import EventLog, code_version, dependency_versions
from .schemas import (
    BudgetConfig,
    Episode,
    EpisodeGold,
    Evidence,
    StrategyConfig,
    SubmitTurnDecisionAction,
    ToolResult,
    TraceConfig,
    TurnDecision,
    WorkflowConfig,
    parse_action,
)
from .utils import redact_content, redact_messages
from .workflow import guidance_summary, load_workflow, validate_decision

SCORER_POLICY_VERSION = "0.2"


class OperatorStopped(Exception):
    pass


# ---------------------------------------------------------------- episode folder

class EpisodeBundle:
    def __init__(self, episode: Episode, workflow: WorkflowConfig, gold: EpisodeGold | None, folder: Path) -> None:
        self.episode = episode
        self.workflow = workflow
        self.gold = gold
        self.folder = folder

    def file_hashes(self) -> dict[str, str]:
        import hashlib

        out = {}
        for name in ("episode.json", "evidence.jsonl", "workflow.json", "gold.json"):
            p = self.folder / name
            if p.exists():
                out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out


def _safe_artifact_path(folder: Path, evidence: Evidence) -> str | None:
    """Reject absolute paths and `..` escapes; return resolved path or None on violation."""
    if evidence.artifact_path is None:
        return None
    raw = Path(evidence.artifact_path)
    if raw.is_absolute():
        return None
    resolved = (folder / raw).resolve()
    if folder.resolve() not in resolved.parents and resolved != folder.resolve():
        return None
    return str(resolved)


def load_episode_bundle(folder: Path, with_gold: bool = True) -> EpisodeBundle:
    folder = Path(folder)
    ep_raw = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    evidence = [
        Evidence.model_validate(json.loads(line))
        for line in (folder / ep_raw.get("evidence_file", "evidence.jsonl")).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode = Episode.model_validate({**ep_raw, "evidence": [e.model_dump(mode="json") for e in evidence]})
    workflow = load_workflow(folder / ep_raw.get("workflow_file", "workflow.json"))
    gold = None
    gold_path = folder / "gold.json"
    if with_gold and gold_path.exists():
        gold = EpisodeGold.model_validate(json.loads(gold_path.read_text(encoding="utf-8")))
    return EpisodeBundle(episode, workflow, gold, folder)


def validate_episode(folder: Path) -> list[str]:
    """Fully-offline structural validation; returns a list of errors (empty = ok)."""
    errors: list[str] = []
    folder = Path(folder)
    try:
        bundle = load_episode_bundle(folder)
    except Exception as exc:
        return [f"load failed: {exc}"]
    ep, wf = bundle.episode, bundle.workflow

    if not ep.turns:
        return ["episode has no turns"]
    turn_ids = [t.turn_id for t in ep.turns]
    if len(set(turn_ids)) != len(turn_ids):
        errors.append("duplicate turn ids")
    for a, b in zip(ep.turns, ep.turns[1:]):
        if b.as_of < a.as_of:
            errors.append(f"non-monotonic as_of: {b.turn_id} earlier than {a.turn_id}")
    ev_ids = [e.evidence_id for e in ep.evidence]
    if len(set(ev_ids)) != len(ev_ids):
        errors.append("duplicate evidence ids")
    ev_by_id = {e.evidence_id: e for e in ep.evidence}
    if not any(e.patient_id == ep.patient_id for e in ep.evidence):
        errors.append(f"no evidence for bound patient {ep.patient_id}")
    for t in ep.turns:
        for ref in t.release_evidence_ids:
            if ref not in ev_by_id:
                errors.append(f"turn {t.turn_id} releases unknown evidence '{ref}'")
            elif ev_by_id[ref].recorded_time > t.as_of:
                errors.append(f"turn {t.turn_id} releases '{ref}' with recorded_time after as_of")
    if wf.start_state() not in wf.states:
        errors.append(f"workflow initial_state '{wf.start_state()}' not in states")
    for tr in wf.transitions:
        if tr.from_state not in wf.states:
            errors.append(f"workflow transition from unknown state '{tr.from_state}'")
        if tr.to_state not in wf.states:
            errors.append(f"workflow transition to unknown state '{tr.to_state}'")
    if bundle.gold:
        turn_set = set(turn_ids)
        for tid in bundle.gold.turns:
            if tid not in turn_set:
                errors.append(f"gold references unknown turn '{tid}'")
        for tid, g in bundle.gold.turns.items():
            if g.expected_workflow_state and g.expected_workflow_state not in wf.states:
                errors.append(f"gold turn {tid} expects unknown workflow state '{g.expected_workflow_state}'")
    for e in ep.evidence:
        if e.artifact_path is not None:
            resolved = _safe_artifact_path(folder, e)
            if resolved is None:
                errors.append(f"evidence {e.evidence_id} has unsafe artifact_path '{e.artifact_path}'")
            elif not Path(resolved).exists():
                errors.append(f"evidence {e.evidence_id} artifact missing: '{e.artifact_path}'")
    return errors


# ---------------------------------------------------------------- interaction hooks

class Interaction:
    """No-op hooks; the interactive CLI subclasses this. One runner for batch and interactive."""

    operator_wait_ms: int = 0

    def on_turn_start(self, turn_index: int, total: int, turn, released_meta: list[dict]) -> None: ...

    def wait_run(self) -> bool:
        return True

    def on_step(self, line: str) -> None: ...

    def on_turn_end(self, decision: TurnDecision | None, public_result: dict | None, note: str) -> None: ...

    def wait_next(self) -> bool:
        return True


# ---------------------------------------------------------------- prompt helpers

def build_episode_messages(episode: Episode, workflow: WorkflowConfig, strategy: StrategyConfig) -> list[Message]:
    tools = ["list_evidence", "read_evidence"]
    if strategy.state_tools_enabled:
        tools += ["get_state", "propose_state_update"]
    tools.append("submit_turn_decision")
    lines = [
        "你是临床诊疗 agent，工作在 teacher-forced 轨迹回放环境：每轮会到达新的真实观察（world observation，作为 user 消息出现），你的动作不会改变已记录的患者未来。",
        f"患者：{episode.patient_id}。共 {len(episode.turns)} 轮。",
        "每轮流程：按需调用工具查看证据，然后恰好调用一次 submit_turn_decision 提交本轮决策。",
        f"可用工具：{', '.join(tools)}。",
        "引用约束：所有 evidence_refs 必须来自此前已成功 read_evidence 的证据；证据不足以更新判断时提交 abstain=true（不得包含 next_action 或 hypotheses）。",
        guidance_summary(workflow),
    ]
    if strategy.state_tools_enabled:
        lines.append("状态规则：可用 propose_state_update 维护跨轮保留的持久 claim 状态；修订用显式 supersedes；矛盾未解决时保留两条，不得默认'最新即真'。")
    return [Message(role="system", content="\n".join(lines))]


def world_message(turn_index: int, total: int, turn, released: list[Evidence]) -> str:
    ids = ", ".join(e.evidence_id for e in released) or "（无新增）"
    return (
        f"[第 {turn_index + 1}/{total} 轮 world observation @ {turn.as_of.isoformat()}]\n"
        f"{turn.world_message or turn.event}\n"
        f"本轮新增证据（已可读取）：{ids}"
    )


def _to_action(item: Any) -> tuple[Any | None, str | None]:
    try:
        if isinstance(item, str):
            return parse_action(json.loads(item)), None
        return parse_action(item), None
    except (ValidationError, json.JSONDecodeError) as exc:
        return None, f"invalid action: {exc}"


def _tool_call(call_id: str, action: Any) -> dict[str, Any]:
    args = action.model_dump(exclude={"type"}, mode="json")
    return {"id": call_id, "type": "function",
            "function": {"name": action.type, "arguments": json.dumps(args, ensure_ascii=False)}}


# ---------------------------------------------------------------- episode budget

class EpisodeBudget:
    def __init__(self, cfg: BudgetConfig) -> None:
        self.cfg = cfg
        self.total_calls = 0
        self.call_seq = 0
        self.started = time.monotonic()

    def check(self, turn_calls: int) -> str | None:
        if self.cfg.per_turn_model_calls is not None and turn_calls >= self.cfg.per_turn_model_calls:
            return "per_turn_model_calls"
        if self.total_calls >= self.cfg.max_model_calls:
            return "max_model_calls"
        if time.monotonic() - self.started > self.cfg.deadline_seconds:
            return "deadline"
        return None

    def exclude_wait(self, seconds: float) -> None:
        self.started += seconds  # operator waiting never eats the episode deadline


def _call_model(model, messages, budget: EpisodeBudget, turn_calls: int, log, episode_id, turn_id):
    last: ModelError | None = None
    retries = budget.cfg.max_retries
    for attempt in range(retries + 1):
        why = budget.check(turn_calls)  # retries also honor call budgets and deadline
        if why:
            raise BudgetExceededError(why)
        budget.total_calls += 1
        budget.call_seq += 1
        t0 = time.monotonic()
        try:
            out = model.next(messages)
            log("model_call", episode_id=episode_id, turn_id=turn_id,
                call_id=str(budget.call_seq), model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000),
                usage=getattr(model, "last_usage", None), attempt=attempt, status="ok")
            return out
        except ModelError as exc:
            last = exc
            log("model_call", episode_id=episode_id, turn_id=turn_id,
                call_id=str(budget.call_seq), model=getattr(model, "name", type(model).__name__),
                latency_ms=int((time.monotonic() - t0) * 1000), usage=None,
                attempt=attempt, status=f"error:{exc.kind}")
            if attempt < retries:
                log("model_retry", episode_id=episode_id, turn_id=turn_id,
                    attempt=attempt + 1, error=f"{exc.kind}: {exc}")
    raise last


# ---------------------------------------------------------------- run_episode

@dataclass
class TurnRow:
    episode_id: str
    turn_id: str
    as_of: str
    phase: str | None
    decision: dict | None = None
    validator_public: dict | None = None
    validator_full: dict | None = None
    termination: str = "completed"
    termination_detail: str = ""
    model_calls: int = 0
    steps: int = 0
    usage: dict | str = "unknown"
    duration_ms: int = 0
    read_ids: list[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    episode_id: str
    termination: str
    termination_detail: str
    rows: list[TurnRow]
    total_model_calls: int
    usage: dict | str
    new_snapshots: list = field(default_factory=list)


def _merge_usage(total: dict | None, usage: Any) -> dict | None:
    if not isinstance(usage, dict):
        return total
    base = total or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base[key] = base.get(key, 0) + (usage.get(key) or 0)
    return base


def run_episode(
    bundle: EpisodeBundle,
    model: ModelClient,
    budget_cfg: BudgetConfig,
    log: EventLog,
    strategy: StrategyConfig,
    trace: TraceConfig | None = None,
    interaction: Interaction | None = None,
    announce: Callable[[str], None] | None = None,
) -> EpisodeResult:
    trace = trace or TraceConfig()
    interaction = interaction or Interaction()
    episode, workflow = bundle.episode, bundle.workflow
    env = EpisodeEnvironment(episode, allow_state_tools=strategy.state_tools_enabled)
    messages = build_episode_messages(episode, workflow, strategy)
    budget = EpisodeBudget(budget_cfg)

    log("episode_start", episode_id=episode.episode_id, patient_id=episode.patient_id,
        mode=episode.mode, workflow_id=workflow.workflow_id, workflow_version=workflow.version,
        model_context=[m.model_dump(exclude_none=True) for m in messages])

    rows: list[TurnRow] = []
    episode_term = ("completed", "")
    usage_total: dict | None = None
    prev_state = workflow.start_state()
    prev_missed = False

    try:
        for idx, turn in enumerate(episode.turns):
            if episode_term[0] != "completed":
                break
            env.advance_turn(turn)
            released = [e for e in episode.evidence if e.evidence_id in turn.release_evidence_ids]
            released_meta = [
                {"evidence_id": e.evidence_id, "modality": e.modality, "recorded_time": e.recorded_time.isoformat()}
                for e in released
            ]
            log("turn_start", episode_id=episode.episode_id, turn_id=turn.turn_id,
                as_of=turn.as_of.isoformat(), phase=turn.phase, released=released_meta)
            interaction.on_turn_start(idx, len(episode.turns), turn, released_meta)

            if prev_missed:
                messages.append(Message(role="user", content="（系统提示：上一轮未收到合法 submit_turn_decision，已按真实轨迹推进到本轮。）"))
            messages.append(Message(role="user", content=world_message(idx, len(episode.turns), turn, released)))

            t0 = time.monotonic()
            row = TurnRow(episode_id=episode.episode_id, turn_id=turn.turn_id,
                          as_of=turn.as_of.isoformat(), phase=turn.phase)
            turn_calls = 0
            step = 0
            decision: TurnDecision | None = None
            turn_usage: dict | None = None
            turn_term = ("completed", "")

            # one operator approval per turn; tool calls then stream freely
            wait0 = time.monotonic()
            if not interaction.wait_run():
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait0)

            while decision is None:
                try:
                    action_item = _call_model(model, messages, budget, turn_calls, log,
                                              episode.episode_id, turn.turn_id)
                except BudgetExceededError as exc:
                    turn_term = ("budget_exceeded", exc.detail)
                    break
                except ModelError as exc:
                    turn_term = ("model_error", f"{exc.kind}: {exc}")
                    break
                turn_calls += 1
                usage_total = _merge_usage(usage_total, getattr(model, "last_usage", None))
                turn_usage = _merge_usage(turn_usage, getattr(model, "last_usage", None))

                parsed, parse_error = _to_action(action_item)
                if parsed is None:
                    step += 1
                    messages.append(Message(role="assistant", content=str(action_item)[:500]))
                    messages.append(Message(
                        role="user",
                        content=f"INVALID ACTION: {parse_error}\n请重新发出一个合法的工具调用（function call）。",
                    ))
                    log("step", episode_id=episode.episode_id, turn_id=turn.turn_id, step=step,
                        action=None, raw=str(action_item)[:200],
                        observation={"ok": False, "error": parse_error})
                    line = f"step {step} INVALID -> {parse_error}"
                    interaction.on_step(line)
                    if announce:
                        announce(line)
                    continue

                call_id = f"call_{step + 1}"
                try:
                    result = env.execute(parsed)
                except Exception as exc:
                    result = ToolResult(ok=False, error=f"tool crashed: {exc!r}", error_kind="internal")
                step += 1
                obs_public = result.public()
                log("step", episode_id=episode.episode_id, turn_id=turn.turn_id, step=step,
                    action=parsed.model_dump(mode="json"),
                    observation=redact_content(obs_public, trace.save_evidence_content)
                    if isinstance(obs_public, dict) else obs_public,
                    violation=result.violation)
                messages.append(Message(
                    role="assistant",
                    content=json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False),
                    tool_calls=[_tool_call(call_id, parsed)],
                ))
                messages.append(Message(role="tool", tool_call_id=call_id,
                                        content=json.dumps(obs_public, ensure_ascii=False)))
                line = f"step {step} {parsed.type} -> " + ("OK" if result.ok else f"ERR {result.error}")
                interaction.on_step(line)
                if announce:
                    announce(line)

                if result.error_kind == "internal":
                    turn_term = ("tool_error", result.error or "internal")
                    break
                if isinstance(parsed, SubmitTurnDecisionAction) and result.ok:
                    decision = parsed.decision
                    break

            validator_full = None
            if decision is not None:
                visible_tags = {tag for e in env._visible() for tag in e.tags}
                action_type = decision.next_action.action_type if decision.next_action else None
                tres = validate_decision(workflow, prev_state, decision.workflow_state, action_type, visible_tags)
                validator_full = {
                    "state_valid": tres.state_valid,
                    "action_valid": tres.action_valid,
                    "transition_valid": tres.transition_valid,
                    "violations": tres.violations,
                    "matched_rule_ids": tres.matched_rule_ids,  # evaluator-side; never shown to the model
                }
                prev_state = decision.workflow_state
                prev_missed = False
            else:
                prev_missed = True

            row.decision = decision.model_dump(mode="json") if decision else None
            row.validator_full = validator_full
            row.validator_public = (
                {k: validator_full[k] for k in ("state_valid", "action_valid", "transition_valid", "violations")}
                if validator_full else None
            )
            row.termination = turn_term[0]
            row.termination_detail = turn_term[1]
            row.model_calls = turn_calls
            row.steps = step
            row.usage = turn_usage if isinstance(turn_usage, dict) else "unknown"
            row.duration_ms = int((time.monotonic() - t0) * 1000)
            row.read_ids = sorted(env.read_ids)
            rows.append(row)

            log("turn_end", episode_id=episode.episode_id, turn_id=turn.turn_id,
                decision=redact_content(row.decision, trace.save_evidence_content),
                validator_public=row.validator_public, validator_full=validator_full,
                termination=turn_term[0], termination_detail=turn_term[1],
                model_calls=turn_calls, steps=step, usage=row.usage,
                duration_ms=row.duration_ms, read_ids=row.read_ids)
            interaction.on_turn_end(decision, row.validator_public, turn_term[1])

            if turn_term[0] == "budget_exceeded" and turn_term[1] in ("max_model_calls", "deadline"):
                episode_term = ("budget_exceeded", turn_term[1])  # episode-level exhaustion ends the run
                break
            if turn_term[0] in ("model_error", "tool_error"):
                # per-turn failure: recorded, then teacher-forced replay continues
                log("episode_note", episode_id=episode.episode_id, turn_id=turn.turn_id,
                    note=f"turn failed ({turn_term[0]}); teacher-forced replay continues")

            wait0 = time.monotonic()
            if not interaction.wait_next():
                raise OperatorStopped()
            budget.exclude_wait(time.monotonic() - wait0)
    except OperatorStopped:
        episode_term = ("operator_stopped", "operator requested stop")
    except KeyboardInterrupt:
        episode_term = ("operator_stopped", "keyboard interrupt")

    usage_out = usage_total if isinstance(usage_total, dict) else "unknown"
    log("episode_end", episode_id=episode.episode_id, termination=episode_term[0],
        termination_detail=episode_term[1], total_model_calls=budget.total_calls,
        usage=usage_out, operator_wait_ms=interaction.operator_wait_ms,
        violations=dict(env.violations),
        messages=redact_messages([m.model_dump(exclude_none=True) for m in messages], trace.save_model_context))
    return EpisodeResult(
        episode_id=episode.episode_id,
        termination=episode_term[0],
        termination_detail=episode_term[1],
        rows=rows,
        total_model_calls=budget.total_calls,
        usage=usage_out,
        new_snapshots=[s.model_dump(mode="json") for s in env.snapshots[1:]],
    )


# ---------------------------------------------------------------- scoring (evaluator side)

def _hypothesis_match(decision: dict, expected: dict[str, str]) -> tuple[int, int]:
    got = {h["label"]: h["status"] for h in (decision.get("state", {}) or {}).get("hypotheses") or []}
    return sum(1 for label, status in expected.items() if got.get(label) == status), len(expected)


def score_episode(rows: list[TurnRow], gold: EpisodeGold) -> dict[str, Any]:
    per_turn: list[dict[str, Any]] = []
    cum_ok = 0
    for i, row in enumerate(rows):
        g = gold.turns.get(row.turn_id)
        entry: dict[str, Any] = {
            "turn_id": row.turn_id,
            "submitted": row.decision is not None,
            "turn_termination": row.termination,
            "model_calls": row.model_calls,
            "usage": row.usage,
            "duration_ms": row.duration_ms,
        }
        if g is None:
            entry["scored"] = False
            per_turn.append(entry)
            continue
        d = row.decision or {}
        checks_num = checks_den = 0
        if g.expected_workflow_state is not None:
            checks_den += 1
            checks_num += int(d.get("workflow_state") == g.expected_workflow_state)
        if g.expected_stage is not None:
            checks_den += 1
            checks_num += int((d.get("state") or {}).get("clinical_stage") == g.expected_stage)
        if g.expected_hypotheses:
            n, den = _hypothesis_match(d, g.expected_hypotheses)
            checks_num += n
            checks_den += den
        entry["state_field_accuracy"] = {"num": checks_num, "den": checks_den}
        # unsubmitted turns count as invalid (failures stay in denominators)
        entry["state_transition_validity"] = bool((row.validator_full or {}).get("transition_valid", False))

        action_type = (d.get("next_action") or {}).get("action_type")
        if g.expect_abstain:
            entry["abstention"] = {"correct": bool(d.get("abstain"))}
            entry["next_action_accuracy"] = None
        elif g.allowed_actions:
            entry["next_action_accuracy"] = {"correct": action_type in g.allowed_actions, "action": action_type}
            entry["abstention"] = None
        else:
            entry["next_action_accuracy"] = None
            entry["abstention"] = None

        gv: list[str] = []
        if action_type and action_type in g.forbidden_actions:
            gv.append(f"forbidden_action:{action_type}")
        if row.validator_full and not row.validator_full.get("action_valid", True):
            gv.append("workflow_action_invalid")
        if row.validator_full and not row.validator_full.get("transition_valid", True):
            gv.append("workflow_transition_invalid")
        entry["guideline_violations"] = gv

        refs = list((d.get("next_action") or {}).get("evidence_refs") or [])
        refs += [r for h in ((d.get("state") or {}).get("hypotheses") or []) for r in h.get("evidence_refs", [])]
        entry["evidence_grounding"] = {"num": sum(1 for r in refs if r in row.read_ids), "den": len(refs)}
        if g.required_evidence_refs:
            entry["required_refs_covered"] = {
                "num": sum(1 for r in g.required_evidence_refs if r in refs),
                "den": len(g.required_evidence_refs),
            }

        turn_ok = (
            row.decision is not None
            and checks_den > 0
            and checks_num == checks_den
            and (entry["next_action_accuracy"] is None or entry["next_action_accuracy"]["correct"])
            and entry["state_transition_validity"] in (None, True)
            and not gv
        )
        cum_ok += int(turn_ok)
        entry["turn_ok"] = turn_ok
        entry["cumulative_success"] = {"num": cum_ok, "den": i + 1}
        entry["scored"] = True
        per_turn.append(entry)

    def _rate(key: str) -> dict[str, Any]:
        rows_with = [e for e in per_turn if e.get("scored") and e.get(key) is not None]
        if key in ("state_field_accuracy", "evidence_grounding", "required_refs_covered"):
            num = sum(e[key]["num"] for e in rows_with)
            den = sum(e[key]["den"] for e in rows_with)
        elif key == "abstention":
            den = len(rows_with)
            num = sum(1 for e in rows_with if e[key]["correct"])
        else:  # next_action_accuracy (dict) / state_transition_validity (bool)
            den = len(rows_with)
            num = sum(1 for e in rows_with if (e[key]["correct"] if isinstance(e[key], dict) else e[key] is True))
        return {"num": num, "den": den, "value": (num / den) if den else None}

    aggregate = {
        "state_field_accuracy": _rate("state_field_accuracy"),
        "state_transition_validity": _rate("state_transition_validity"),
        "next_action_accuracy": _rate("next_action_accuracy"),
        "abstention_accuracy": _rate("abstention"),
        "evidence_grounding": _rate("evidence_grounding"),
        "required_refs_covered": _rate("required_refs_covered"),
        "guideline_violation_count": sum(len(e.get("guideline_violations", [])) for e in per_turn if e.get("scored")),
        "trajectory_completion": {
            "num": sum(1 for r in rows if r.decision is not None),
            "den": len(rows),
            "value": (sum(1 for r in rows if r.decision is not None) / len(rows)) if rows else None,
        },
        "final_cumulative_success": per_turn[-1].get("cumulative_success") if per_turn else None,
    }
    return {"per_turn": per_turn, "aggregate": aggregate, "scorer_policy_version": SCORER_POLICY_VERSION}


# ---------------------------------------------------------------- batch orchestration

from pydantic import Field  # noqa: E402

from .schemas import RunConfig  # noqa: E402


class TrajectoryRunConfig(RunConfig):
    """RunConfig minus question-runner fields; episodes replace cases/data_dir."""

    data_dir: str = ""  # unused for trajectory runs; kept for config compatibility
    episodes: list[str] = Field(default_factory=list)


def run_trajectory(
    config_path: Path,
    runs_root: Path,
    interactive: bool = False,
    episode_dirs: list[Path] | None = None,
    verbose: bool = False,
) -> Path:
    import secrets
    from datetime import datetime, timezone

    from .model import make_episode_model_factory

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = TrajectoryRunConfig.model_validate(raw)
    base = Path(config_path).resolve().parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (base / path).resolve()

    dirs = episode_dirs or [_resolve(p) for p in cfg.episodes]
    script_dir = _resolve(cfg.model.script_dir) if cfg.model.script_dir else None
    model_factory = make_episode_model_factory(
        cfg.model, script_dir,
        state_tools_enabled=cfg.strategy.state_tools_enabled,
        request_timeout=cfg.budget.request_timeout_seconds,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root) / f"{stamp}-{cfg.name}-{secrets.token_hex(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    episode_info = []
    for d in dirs:
        bundle = load_episode_bundle(d, with_gold=False)  # run never touches gold
        episode_info.append({"episode_dir": str(d), "episode_id": bundle.episode.episode_id,
                             "files": bundle.file_hashes()})
    resolved = {
        "name": cfg.name,
        "config_path": str(Path(config_path).resolve()),
        "kind": "trajectory",
        "mode": "teacher_forced_replay",
        "episodes": episode_info,
        "script_dir": str(script_dir) if script_dir else None,
        "model": cfg.model.model_dump(),
        "strategy": cfg.strategy.model_dump(mode="json"),
        "budget": cfg.budget.model_dump(mode="json"),
        "trace": cfg.trace.model_dump(),
        "experiment": cfg.experiment,
        "code_version": code_version(),
        "dependencies": dependency_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")

    ops: dict[str, Any] = {"episodes": {}, "total_model_calls": 0, "usage": "unknown"}
    usage_all: dict | None = None

    with open(run_dir / "events.jsonl", "w", encoding="utf-8") as ev_fh, \
            open(run_dir / "turn_decisions.jsonl", "w", encoding="utf-8") as dec_fh, \
            open(run_dir / "state_snapshots.jsonl", "w", encoding="utf-8") as snap_fh:
        log = EventLog(ev_fh, run_id=run_dir.name)
        log("run_start", run_dir=str(run_dir), name=cfg.name,
            episodes=[e["episode_id"] for e in episode_info], interactive=interactive)

        for d in dirs:
            errs = validate_episode(d)
            if errs:
                log("episode_skipped", episode_dir=str(d), errors=errs)
                ops["episodes"][d.name] = {"termination": "invalid", "errors": errs}
                continue
            bundle = load_episode_bundle(d, with_gold=False)
            model = model_factory(bundle.episode.episode_id)
            interaction: Interaction = Interaction()
            if interactive:
                from .cli import CliInteraction
                interaction = CliInteraction(bundle.episode.episode_id, note_fn=lambda **kw: log("operator_note", **kw))
            announce = (lambda m: print(m, flush=True)) if (verbose or interactive) else None
            result = run_episode(bundle, model, cfg.budget, log, cfg.strategy,
                                 trace=cfg.trace, interaction=interaction, announce=announce)
            for row in result.rows:
                dec_fh.write(json.dumps(row.__dict__, ensure_ascii=False, default=str) + "\n")
            dec_fh.flush()
            for snap in result.new_snapshots:
                snap_fh.write(json.dumps(snap, ensure_ascii=False, default=str) + "\n")
            snap_fh.flush()
            usage_all = _merge_usage(usage_all, result.usage) if isinstance(result.usage, dict) else usage_all
            ops["episodes"][result.episode_id] = {
                "termination": result.termination,
                "termination_detail": result.termination_detail,
                "turns": len(result.rows),
                "model_calls": result.total_model_calls,
                "usage": result.usage,
                "operator_wait_ms": interaction.operator_wait_ms,
            }
            ops["total_model_calls"] += result.total_model_calls

        if isinstance(usage_all, dict):
            ops["usage"] = usage_all
        metrics = {"backend": cfg.model.type, "kind": "trajectory", "operational": ops}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log("run_end", operational=ops)

    _write_trajectory_report(run_dir, cfg, ops, scored=None)
    print(f"trajectory run complete: {run_dir}")
    return run_dir


def evaluate_trajectory(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    rows_by_episode: dict[str, list[TurnRow]] = {}
    for line in (run_dir / "turn_decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = TurnRow(
            episode_id=raw["episode_id"], turn_id=raw["turn_id"], as_of=raw["as_of"], phase=raw.get("phase"),
            decision=raw.get("decision"), validator_public=raw.get("validator_public"),
            validator_full=raw.get("validator_full"), termination=raw.get("termination", "completed"),
            termination_detail=raw.get("termination_detail", ""), model_calls=raw.get("model_calls", 0),
            steps=raw.get("steps", 0), usage=raw.get("usage", "unknown"),
            duration_ms=raw.get("duration_ms", 0), read_ids=raw.get("read_ids", []),
        )
        rows_by_episode.setdefault(raw["episode_id"], []).append(row)

    scored_episodes: dict[str, Any] = {}
    for info in resolved.get("episodes", []):
        eid = info["episode_id"]
        if eid not in rows_by_episode:
            continue
        bundle = load_episode_bundle(Path(info["episode_dir"]), with_gold=True)
        if bundle.gold is None:
            scored_episodes[eid] = {"scored": False, "reason": "no gold.json"}
            continue
        scored_episodes[eid] = score_episode(rows_by_episode[eid], bundle.gold)
        scored_episodes[eid]["gold_sha256"] = bundle.file_hashes().get("gold.json")
        scored_episodes[eid]["workflow_version"] = bundle.workflow.version

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["scored"] = scored_episodes
    metrics["scorer_policy_version"] = SCORER_POLICY_VERSION
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_trajectory_report(run_dir, resolved, metrics.get("operational", {}), scored=scored_episodes)
    return metrics


def _fmt_rate(r: dict | None) -> str:
    if not isinstance(r, dict) or not r.get("den"):
        return "N/A"
    return f"{r.get('num', 0)}/{r['den']}"


def _write_trajectory_report(run_dir: Path, cfg_like, ops: dict, scored: dict | None) -> None:
    model_name = cfg_like.get("model", {}).get("model") if isinstance(cfg_like, dict) else getattr(
        getattr(cfg_like, "model", None), "model", None)
    backend = cfg_like.get("model", {}).get("type") if isinstance(cfg_like, dict) else getattr(
        getattr(cfg_like, "model", None), "type", "scripted")
    if backend == "scripted":
        backend_note = "scripted 校准运行：满分只证明运行器与评分器工作，不代表真实模型效果"
    else:
        backend_note = f"{backend} 真实模型运行（{model_name or '?'}）；teacher-forced replay 评估，不是闭环模拟；小样本不构成基准结论"
    lines = [
        "# Trajectory run report",
        "",
        f"- run_dir: `{run_dir}`",
        f"- backend: **{backend}**（{backend_note}）",
        f"- mode: teacher_forced_replay（agent 动作被评分，不改变记录中的患者未来）",
        "",
        "## Episodes（运行概览）",
        "",
        "| episode | termination | turns | model calls | usage |",
        "|---|---|---|---|---|",
    ]
    for eid, e in ops.get("episodes", {}).items():
        usage = e.get("usage") if isinstance(e, dict) else None
        usage_s = usage.get("total_tokens") if isinstance(usage, dict) else (usage or "unknown")
        lines.append(f"| {eid} | {e.get('termination')} | {e.get('turns')} | {e.get('model_calls')} | {usage_s} |")
    if scored:
        lines += ["", "## 逐轮评分（evaluator-side，规则判分）", ""]
        for eid, sc in scored.items():
            if not sc.get("scored", True) and "per_turn" not in sc:
                lines.append(f"### {eid}: 未评分（{sc.get('reason', '?')}）")
                continue
            agg = sc["aggregate"]
            lines += [
                f"### {eid}",
                "",
                "| turn | 提交 | 终因 | 状态字段 | 迁移有效 | 动作正确 | 违规 | 累计成功 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for t in sc["per_turn"]:
                state = _fmt_rate(t.get("state_field_accuracy"))
                action = t.get("next_action_accuracy")
                action_s = "—" if action is None else ("✓" if action["correct"] else f"✗({action.get('action')})")
                trans = t.get("state_transition_validity")
                trans_s = "—" if trans is None else ("✓" if trans else "✗")
                cum = _fmt_rate(t.get("cumulative_success"))
                abst = t.get("abstention")
                abst_s = f"弃答:{'✓' if abst and abst['correct'] else '✗'}" if abst else ""
                lines.append(
                    f"| {t['turn_id']} | {'是' if t['submitted'] else '否'} | {t['turn_termination']} {abst_s} "
                    f"| {state} | {trans_s} | {action_s} | {len(t.get('guideline_violations', []))} | {cum} |"
                )
            lines += [
                "",
                f"- aggregate: state_field={_fmt_rate(agg['state_field_accuracy'])} "
                f"transition={_fmt_rate(agg['state_transition_validity'])} "
                f"next_action={_fmt_rate(agg['next_action_accuracy'])} "
                f"abstention={_fmt_rate(agg['abstention_accuracy'])} "
                f"grounding={_fmt_rate(agg['evidence_grounding'])} "
                f"violations={agg['guideline_violation_count']} "
                f"completion={_fmt_rate(agg['trajectory_completion'])}",
                f"- gold_sha256: {sc.get('gold_sha256', 'N/A')[:12] if sc.get('gold_sha256') else 'N/A'}; "
                f"workflow_version={sc.get('workflow_version', 'N/A')}; scorer_policy={sc.get('scorer_policy_version', SCORER_POLICY_VERSION)}",
                "",
            ]
    lines += [
        "## 口径",
        "",
        "- 未提交/失败的轮保留在分母；分母为 0 记 N/A。",
        "- replay 分数是 proposed-action correctness，不是闭环临床模拟成绩。",
        "- gold/workflow 判分细节（含 matched rule ids）仅存于 evaluator 侧，不进入模型上下文。",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
