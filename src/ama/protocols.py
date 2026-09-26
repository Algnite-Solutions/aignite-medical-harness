"""Independent model-facing interaction protocols for the common replay loop."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .data import Decision, Episode, Evidence, Turn
from .model import Message


@dataclass
class StepResult:
    decision: Decision | None = None
    messages: list[Message] = field(default_factory=list)
    event: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _content(item: Any) -> Any:
    if isinstance(item, dict) and ("content" in item or "tool_calls" in item):
        return item.get("content") or ""
    return item


def parse_decision(item: Any) -> Decision:
    """Find a Decision in text; providers may add fences or thinking tags."""
    value = _content(item)
    if isinstance(value, Decision):
        return value
    if isinstance(value, dict):
        return Decision.model_validate(value)
    if not isinstance(value, str):
        raise ValueError("response is not a JSON Decision")
    cleaned = re.sub(r"<think>.*?</think>", "", value, flags=re.I | re.S)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[index:])
            return Decision.model_validate(obj)
        except (ValueError, TypeError):
            continue
    raise ValueError("no valid Decision JSON object")


def _visible_decision(decision: Decision, turn: Turn, allowed: set[str]) -> str | None:
    if decision.turn_id != turn.id:
        return f"turn_id must be {turn.id!r}"
    if decision.answer is None and decision.citations:
        return "abstention (answer=null) must have no citations"
    missing = set(decision.citations) - allowed
    if missing:
        return f"citation not permitted: {sorted(missing)}"
    return None


def _file_part(evidence: Evidence, root: Path) -> dict[str, Any] | None:
    if not evidence.file:
        return None
    path = (root / evidence.file).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"unsafe or missing evidence file {evidence.file!r}")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _evidence_text(evidence: Evidence) -> str:
    label = f'<evidence id={json.dumps(evidence.id)}'
    if evidence.type:
        label += f' type={json.dumps(evidence.type)}'
    return f"{label}>\n{evidence.text or ''}\n</evidence>"


class InteractionProtocol(Protocol):
    """工具扩展接口：提供提示、每轮消息、工具定义和响应处理。

    扩展自行管理工具状态；step 返回消息继续执行，或返回 Decision 结束本轮。
    """

    name: str
    tools: list[dict[str, Any]] | None

    def system_prompt(self, episode: Episode, instruction: str) -> str: ...

    def turn_messages(self, turn: Turn, index: int, total: int, root: Path,
                      visible: dict[str, Evidence]) -> list[Message]: ...

    def step(self, item: Any, turn: Turn, visible: dict[str, Evidence],
             root: Path) -> StepResult: ...


class DirectDecisionProtocol:
    name = "direct_decision"
    tools: list[dict[str, Any]] | None = None

    def system_prompt(self, episode: Episode, instruction: str) -> str:
        return ("You are an evaluated decision-maker in a fixed evidence-release replay. "
                "Later observations are recorded and your answers do not change them. "
                f"This episode has {len(episode.turns)} stages. At each stage, use only evidence shown so far. "
                "Return exactly one JSON object with turn_id, answer, and citations. "
                "answer is the task-defined JSON value; use null to abstain. "
                "Cite only visible evidence IDs.\n" + instruction)

    def turn_messages(self, turn: Turn, index: int, total: int, root: Path,
                      visible: dict[str, Evidence]) -> list[Message]:
        label = f"[turn_id={turn.id} | stage {index + 1}/{total}]"
        if turn.available_at:
            label += f" available_at={turn.available_at.isoformat()}"
        body = "\n".join([label, turn.observation or "", "New evidence:",
                          *(_evidence_text(ev) for ev in turn.evidence)])
        images = [part for ev in turn.evidence if (part := _file_part(ev, root))]
        if images:
            return [Message(role="user", content=[{"type": "text", "text": body}, *images])]
        return [Message(role="user", content=body)]

    def step(self, item: Any, turn: Turn, visible: dict[str, Evidence],
             root: Path) -> StepResult:
        raw = _content(item)
        try:
            decision = parse_decision(item)
            error = _visible_decision(decision, turn, set(visible))
        except (ValueError, TypeError) as exc:
            decision, error = None, str(exc)
        messages = [Message(role="assistant", content=raw if isinstance(raw, str)
                            else json.dumps(raw, ensure_ascii=False, default=str))]
        if error:
            messages.append(Message(role="user", content=f"INVALID DECISION: {error}. Return one corrected JSON Decision only."))
            return StepResult(messages=messages, error=error, event={"action": "invalid_decision"})
        return StepResult(decision=decision, messages=messages, event={"action": "decision"})
