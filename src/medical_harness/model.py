"""Model clients.

- ScriptedModel: deterministic replay, offline default for tests/CI.
- OpenAICompatModel: any OpenAI-compatible /chat/completions endpoint
  (GLM etc.), stdlib-only (urllib), explicit function tools.

The API key never appears in code, configs, or logs: it is read from the
environment (see .env.example) and only ever placed in the Authorization
header of the request itself.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .schemas import ModelConfig, parse_action


class ModelError(Exception):
    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


class ModelTimeoutError(ModelError):
    def __init__(self, message: str = "model request timed out") -> None:
        super().__init__(message, kind="timeout")


class Message(BaseModel):
    role: str  # system | user | assistant | tool
    content: str
    tool_calls: list[dict[str, Any]] | None = None  # assistant-side function call
    tool_call_id: str | None = None  # tool-side reference


class ModelClient(Protocol):
    name: str

    def next(self, messages: list[Message]) -> Any:
        """Return the next action (typed model or raw str) or raise ModelError."""
        ...


# ---------------------------------------------------------------- scripted

class ScriptedModel:
    """Replays a pre-written action list; observations are ignored (deterministic).

    Entries may be typed actions or JSON strings; an empty queue raises
    ModelError('script exhausted') which the runner records as model_error.
    """

    def __init__(self, name: str, actions: list[Any]) -> None:
        self.name = name
        self.last_usage: dict[str, Any] | None = None
        self._queue: deque[Any] = deque(actions)

    def next(self, messages: list[Message]) -> Any:
        if not self._queue:
            raise ModelError(f"script exhausted for {self.name}", kind="exhausted")
        item = self._queue.popleft()
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return parse_action(item)
        return item


def load_scripts(script_dir: Path) -> dict[tuple[str, str], list[Any]]:
    """Load <script_dir>/<case_id>.json files: {"case_id", "by_question": {qid: [action, ...]}}."""
    scripts: dict[tuple[str, str], list[Any]] = {}
    for path in sorted(script_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        case_id = raw["case_id"]
        for question_id, actions in raw.get("by_question", {}).items():
            scripts[(case_id, question_id)] = [parse_action(a) for a in actions]
    return scripts


# ---------------------------------------------------------------- tool definitions

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "question_id": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "value"],
            },
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["question_id", "abstain"],
}


_TURN_DECISION_SCHEMA = {
    "type": "object",
    "description": "本轮诊疗决策：流程节点、诊断状态（鉴别诊断假设/分期）、下一临床动作。",
    "properties": {
        "turn_id": {"type": "string"},
        "workflow_state": {"type": "string", "description": "诊疗流程节点，如 initial_imaging/pathology/staging"},
        "state": {
            "type": "object",
            "properties": {
                "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
                    "label": {"type": "string"},
                    "status": {"type": "string", "enum": ["possible", "supported", "confirmed", "ruled_out"]},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                }, "required": ["label", "status"]}},
                "clinical_stage": {"type": ["string", "null"], "description": "疾病分期（如 TNM），不是流程节点"},
                "unresolved_questions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "next_action": {"type": ["object", "null"], "properties": {
            "action_type": {"type": "string"},
            "target": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        }, "required": ["action_type"]},
        "report": {"type": ["string", "null"], "description": "对本轮新证据的简要结构化理解"},
        "abstain": {"type": "boolean"},
    },
    "required": ["turn_id", "workflow_state", "abstain"],
}


def tool_definitions(state_tools_enabled: bool, episode: bool = False) -> list[dict[str, Any]]:
    defs = [
        {"type": "function", "function": {
            "name": "list_evidence",
            "description": "列出当前患者截至 as_of 可见证据的元数据（编号、时间、模态、定位；不含内容）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }},
        {"type": "function", "function": {
            "name": "read_evidence",
            "description": "读取一条可见证据的完整内容与来源定位。",
            "parameters": {"type": "object",
                           "properties": {"evidence_id": {"type": "string"}},
                           "required": ["evidence_id"]},
        }},
    ]
    if state_tools_enabled:
        defs.append({"type": "function", "function": {
            "name": "get_state",
            "description": "读取当前维护的患者状态（claim 集合与版本）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }})
        defs.append({"type": "function", "function": {
            "name": "propose_state_update",
            "description": "原子提交一批 claim（校验证据可见性与版本号后整体写入；修订用 supersedes 显式指向旧 claim）。",
            "parameters": {"type": "object",
                           "properties": {
                               "claims": {"type": "array", "items": {"type": "object", "properties": {
                                   "key": {"type": "string"},
                                   "value": {},
                                   "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                   "supersedes": {"type": "string"},
                                   "status": {"type": "string", "enum": ["active", "uncertain"]},
                               }, "required": ["key", "value"]}},
                               "expected_version": {"type": "integer"},
                           },
                           "required": ["claims", "expected_version"]},
        }})
    if episode:
        defs.append({"type": "function", "function": {
            "name": "submit_turn_decision",
            "description": "提交本轮最终诊疗决策（每轮恰好一次）。证据不足以更新判断时 abstain=true 且不带 next_action/hypotheses。",
            "parameters": {"type": "object",
                           "properties": {"decision": _TURN_DECISION_SCHEMA},
                           "required": ["decision"]},
        }})
    else:
        defs.append({"type": "function", "function": {
            "name": "submit_answer",
            "description": "提交本问题的最终答案；证据不足时提交 abstain=true。调用后本问题结束。",
            "parameters": {"type": "object",
                           "properties": {"answer": _ANSWER_SCHEMA},
                           "required": ["answer"]},
        }})
    return defs


# ---------------------------------------------------------------- OpenAI-compatible

class OpenAICompatModel:
    """Chat Completions client with explicit function tools (stdlib urllib).

    - One tool call per turn is expected; if the provider returns several,
      only the first is executed.
    - A text-only reply (no tool call) is returned as a raw string so the
      agent loop turns it into a visible 'invalid action' error.
    - Usage (token counts) is captured on `last_usage` for logging; costs
      are NOT computed (no price configuration -> report unknown).
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        state_tools_enabled: bool = True,
        request_timeout: float = 30.0,
        temperature: float = 0.0,
        episode_tools: bool = False,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.state_tools_enabled = state_tools_enabled
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.episode_tools = episode_tools
        self.last_usage: dict[str, Any] | None = None

    # -- wire format -------------------------------------------------------

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                wire.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = m.tool_calls
                wire.append(entry)
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                wire.append({"role": m.role, "content": m.content})
        return wire

    # -- request -----------------------------------------------------------

    def next(self, messages: list[Message]) -> Any:
        payload = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "tools": tool_definitions(self.state_tools_enabled, episode=self.episode_tools),
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError:
            raise ModelTimeoutError() from None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise ModelError(f"HTTP {exc.code}: {body}", kind="http_error") from None
        except urllib.error.URLError as exc:
            raise ModelError(f"network error: {exc.reason}", kind="network") from None
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid JSON response: {exc}", kind="bad_response") from None

        self.last_usage = data.get("usage")
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise ModelError(f"malformed response: {json.dumps(data, ensure_ascii=False)[:300]}", kind="bad_response") from None

        tool_calls = choice.get("tool_calls")
        if tool_calls:
            fn = tool_calls[0]["function"]
            args = fn.get("arguments") or "{}"
            try:
                payload_args = json.loads(args)
                payload_args["type"] = fn["name"]
                return parse_action(payload_args)
            except (json.JSONDecodeError, ValidationError):
                return args  # raw string -> agent shows a visible 'invalid action' error
        return choice.get("content") or ""


# ---------------------------------------------------------------- factory

def make_model_factory(
    model_cfg: ModelConfig,
    script_dir: Path | None,
    state_tools_enabled: bool = True,
    request_timeout: float = 30.0,
):
    """Return factory(case_id, question_id) -> ModelClient (fresh instance per question)."""
    if model_cfg.type == "scripted":
        if script_dir is None:
            raise ValueError("scripted model requires script_dir")
        scripts = load_scripts(script_dir)

        def scripted_factory(case_id: str, question_id: str) -> ScriptedModel:
            actions = scripts.get((case_id, question_id), [])
            return ScriptedModel(f"{case_id}/{question_id}", list(actions))

        return scripted_factory

    if model_cfg.type == "openai_compatible":
        if not model_cfg.base_url or not model_cfg.model:
            raise ValueError("openai_compatible model requires base_url and model in config")
        api_key = os.environ.get(model_cfg.api_key_env or "GLM_API_KEY", "")
        if not api_key:
            raise ValueError(f"missing API key: set {model_cfg.api_key_env} (see .env.example); refusing to run")

        def openai_factory(case_id: str, question_id: str) -> OpenAICompatModel:
            return OpenAICompatModel(
                name=f"{model_cfg.model}:{case_id}/{question_id}",
                base_url=model_cfg.base_url,
                model=model_cfg.model,
                api_key=api_key,
                state_tools_enabled=state_tools_enabled,
                request_timeout=request_timeout,
                temperature=model_cfg.temperature,
            )

        return openai_factory

    raise ValueError(f"unsupported model type '{model_cfg.type}': use 'scripted' or 'openai_compatible'")


def make_episode_model_factory(
    model_cfg: ModelConfig,
    script_dir: Path | None,
    state_tools_enabled: bool = True,
    request_timeout: float = 30.0,
):
    """Episode variant: one model client persists across all turns of an episode.

    Scripted files: <script_dir>/<episode_id>.json = {"episode_id", "actions": [...]}
    with a flat action list across turns (turns are separated by submit_turn_decision).
    """
    if model_cfg.type == "scripted":
        if script_dir is None:
            raise ValueError("scripted model requires script_dir")
        scripts: dict[str, list[Any]] = {}
        for path in sorted(script_dir.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            scripts[raw["episode_id"]] = [parse_action(a) for a in raw.get("actions", [])]

        def scripted_episode_factory(episode_id: str) -> ScriptedModel:
            return ScriptedModel(episode_id, list(scripts.get(episode_id, [])))

        return scripted_episode_factory

    if model_cfg.type == "openai_compatible":
        if not model_cfg.base_url or not model_cfg.model:
            raise ValueError("openai_compatible model requires base_url and model in config")
        api_key = os.environ.get(model_cfg.api_key_env or "GLM_API_KEY", "")
        if not api_key:
            raise ValueError(f"missing API key: set {model_cfg.api_key_env} (see .env.example); refusing to run")

        def openai_episode_factory(episode_id: str) -> OpenAICompatModel:
            return OpenAICompatModel(
                name=f"{model_cfg.model}:{episode_id}",
                base_url=model_cfg.base_url,
                model=model_cfg.model,
                api_key=api_key,
                state_tools_enabled=state_tools_enabled,
                request_timeout=request_timeout,
                temperature=model_cfg.temperature,
                episode_tools=True,
            )

        return openai_episode_factory

    raise ValueError(f"unsupported model type '{model_cfg.type}': use 'scripted' or 'openai_compatible'")
