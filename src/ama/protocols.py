"""Independent model-facing interaction protocols for the common replay loop."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import Decision, Episode, Evidence, Turn
from .model import Message


EVIDENCE_TOOLS = [
    {"type": "function", "function": {"name": "list_evidence",
     "description": "List IDs and types of evidence visible through the current stage.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_evidence",
     "description": "Read one visible evidence item before citing it.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
]


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


class DirectDecisionProtocol:
    name = "direct_decision"
    version = "1"
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
             read: set[str], root: Path) -> StepResult:
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


class ToolAgentProtocol(DirectDecisionProtocol):
    name = "tool_agent"
    tools = EVIDENCE_TOOLS

    def system_prompt(self, episode: Episode, instruction: str) -> str:
        return ("You are an evaluated agent in a fixed evidence-release replay. "
                f"This episode has {len(episode.turns)} stages. Use list_evidence and read_evidence "
                "to inspect currently visible evidence. You may cite only IDs you have read. "
                "When ready, return exactly one JSON Decision with turn_id, answer, citations; "
                "answer=null means abstain. Do not call a submit tool.\n" + instruction)

    def turn_messages(self, turn: Turn, index: int, total: int, root: Path,
                      visible: dict[str, Evidence]) -> list[Message]:
        label = f"[turn_id={turn.id} | stage {index + 1}/{total}]"
        if turn.available_at:
            label += f" available_at={turn.available_at.isoformat()}"
        ids = ", ".join(ev.id for ev in turn.evidence) or "none"
        return [Message(role="user", content=f"{label}\n{turn.observation or ''}\nNew evidence IDs: {ids}")]

    def step(self, item: Any, turn: Turn, visible: dict[str, Evidence],
             read: set[str], root: Path) -> StepResult:
        call = None
        if isinstance(item, dict) and item.get("tool_calls"):
            call = item["tool_calls"][0]
        elif isinstance(item, dict) and item.get("type") in {"list_evidence", "read_evidence"}:
            call = {"id": "scripted_tool", "type": "function", "function": {
                "name": item["type"], "arguments": json.dumps({k: v for k, v in item.items() if k != "type"})}}
        if call is None:
            result = super().step(item, turn, visible, read, root)
            if result.decision:
                error = _visible_decision(result.decision, turn, read)
                if error:
                    result.decision = None
                    result.error = error
                    result.messages.append(Message(role="user", content=f"INVALID DECISION: {error}. Read evidence before citing."))
            return result
        fn = call.get("function") or {}
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be an object")
            if name == "list_evidence":
                observation: dict[str, Any] = {"evidence": [{"id": ev.id, "type": ev.type} for ev in visible.values()]}
            elif name == "read_evidence":
                evidence_id = args.get("id")
                if evidence_id not in visible:
                    raise ValueError(f"evidence {evidence_id!r} is not visible")
                ev = visible[evidence_id]
                observation = {"evidence": ev.model_dump(exclude_none=True)}
                read.add(evidence_id)
            else:
                raise ValueError(f"unknown tool {name!r}")
            error = None
        except (ValueError, TypeError) as exc:
            observation, error = {"error": str(exc)}, str(exc)
        call_id = call.get("id") or "call_1"
        messages = [Message(role="assistant", content="", tool_calls=[call]),
                    Message(role="tool", tool_call_id=call_id,
                            content=json.dumps(observation, ensure_ascii=False))]
        if not error and name == "read_evidence":
            part = _file_part(visible[args["id"]], root)
            if part:
                messages.append(Message(role="user", content=[
                    {"type": "text", "text": f"Image evidence id={args['id']}"}, part]))
        return StepResult(messages=messages, error=error,
                          event={"action": name, "arguments": args if not error else {},
                                 "observation": observation})


def get_protocol(name: str) -> DirectDecisionProtocol | ToolAgentProtocol:
    if name == "direct_decision":
        return DirectDecisionProtocol()
    if name == "tool_agent":
        return ToolAgentProtocol()
    raise ValueError(f"unknown protocol {name!r}")
