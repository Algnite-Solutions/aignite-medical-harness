"""Model transports. Interaction protocols own prompts and response parsing."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class ModelError(Exception):
    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


class ModelTimeoutError(ModelError):
    def __init__(self, message: str = "model request timed out") -> None:
        super().__init__(message, kind="timeout")


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ModelClient(Protocol):
    name: str
    last_usage: dict[str, Any] | None

    def next(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Any: ...


# ---------------------------------------------------------------- scripted

class ScriptedModel:
    """Flat per-episode action list; observations ignored (deterministic, offline)."""

    def __init__(self, name: str, actions: list[Any]) -> None:
        self.name = name
        self.last_usage = None
        self._queue: deque[Any] = deque(actions)

    def next(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Any:
        if not self._queue:
            raise ModelError(f"script exhausted for {self.name}", kind="exhausted")
        item = self._queue.popleft()
        if isinstance(item, str):
            return item
        return item


def load_episode_scripts(script_dir: Path) -> dict[str, list[Any]]:
    """<script_dir>/<episode_id>.json = {"episode_id", "actions": [...]} ."""
    scripts: dict[str, list[Any]] = {}
    for path in sorted(Path(script_dir).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        scripts[raw["episode_id"]] = raw.get("actions", [])
    return scripts


# ---------------------------------------------------------------- OpenAI-compatible

class OpenAICompatModel:
    """OpenAI-compatible Chat Completions using tools or plain JSON text."""

    def __init__(self, name: str, base_url: str, model: str, api_key: str,
                 request_timeout: float = 30.0, temperature: float = 0.0,
                 wire: str = "openai",
                 max_tokens: int | None = None,
                 chat_template_kwargs: dict[str, Any] | None = None) -> None:
        if wire not in ("openai", "openai_compact_image"):
            raise ValueError(f"unknown wire format: {wire!r}")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.wire = wire
        self.max_tokens = max_tokens
        self.chat_template_kwargs = chat_template_kwargs
        self.last_usage: dict[str, Any] | None = None

    @staticmethod
    def _compact_image_history(messages: list[Message]) -> list[Message]:
        """Rewrite the transcript for gateways that reject image-bearing user messages
        after tool messages (they 400 on assistant(tool_call) -> tool -> user(image)).

        Everything up to and including the LAST image is folded into ONE user message:
        world observations, prior actions and their observations, image descriptors, and
        every successfully read image so far. Messages after the last image are kept
        verbatim (the gateway accepts tool pairs that FOLLOW an image user message).
        Evidence visibility checks stay in the protocol; only the wire transcript differs."""
        image_indexes = [i for i, m in enumerate(messages)
                         if m.role == "user" and isinstance(m.content, list)
                         and any(isinstance(part, dict) and part.get("type") == "image_url"
                                 for part in m.content)]
        if not image_indexes:
            return messages
        last = image_indexes[-1]
        head, tail = messages[: last + 1], messages[last + 1:]

        transcript: list[str] = []
        image_parts: list[dict[str, Any]] = []
        for m in head:
            if m.role == "system":
                continue  # preserved as separate system messages
            if m.role == "user":
                if isinstance(m.content, str):
                    transcript.append(m.content)
                else:
                    for part in m.content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            image_parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            transcript.append(str(part.get("text", "")))
            elif m.role == "assistant":
                if m.tool_calls:
                    for call in m.tool_calls:
                        fn = call.get("function", {})
                        transcript.append(f'Previous action: {fn.get("name")}({fn.get("arguments")})')
                elif m.content:
                    transcript.append(f"Assistant reply: {m.content}")
            elif m.role == "tool":
                transcript.append(f"Observation: {m.content}")
        transcript.append(
            "The evidence image(s) are attached. "
            "Return a JSON Decision for the current turn when ready.")
        merged = Message(role="user",
                         content=[{"type": "text", "text": "\n\n".join(t for t in transcript if t)}]
                         + image_parts)
        return ([m for m in head if m.role == "system"] + [merged] + list(tail))

    def _to_wire(self, messages: list[Message]) -> list[dict[str, Any]]:
        if self.wire == "openai_compact_image":
            messages = self._compact_image_history(messages)
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = m.tool_calls
                wire.append(entry)
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                wire.append({"role": m.role, "content": m.content})
        return wire

    def next(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Any:
        payload = {"model": self.model, "messages": self._to_wire(messages),
                   "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
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
        return choice


# ---------------------------------------------------------------- config + factory

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "scripted"
    script_dir: str = "scripts"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "GLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    context_window: int | None = None
    # wire: how the conversation is serialized for the provider.
    #   "openai"               — standard tool-call pairing (default)
    #   "openai_compact_image" — for gateways that reject assistant(tool_call) -> tool ->
    #                           user(image) sequences; completed read_evidence history is
    #                           compacted to system + merged instruction+image. Read-before-
    #                           cite gating still happens in the agent; only the provider
    #                           transcript differs (visible in the model config).
    wire: str = "openai"


DEFAULT_SCRIPTED = {"type": "scripted", "script_dir": "scripts"}


def load_model_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in (Path.home() / ".config/ama/models.json", Path("ama.json")):
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                configs.update(raw.get("models", {}))
            except (json.JSONDecodeError, OSError):
                continue
    return configs


def resolve_model_config(model_name: str) -> ModelConfig:
    """Resolve a model alias to its effective config (ama.json / user config merged).
    Used by the factory and by run provenance (manifest) so a historical run records
    the actual provider model and wire dialect, not just the mutable alias."""
    cfg_raw = load_model_configs().get(model_name)
    if model_name == "scripted" and cfg_raw is None:
        cfg_raw = DEFAULT_SCRIPTED
    if cfg_raw is None:
        raise ValueError(f"unknown model '{model_name}' (configure in ama.json or ~/.config/ama/models.json)")
    return ModelConfig.model_validate(cfg_raw)


def make_model_factory(model_name: str, request_timeout: float = 60.0):
    """factory(episode_id) -> ModelClient (one client per episode, fresh usage)."""
    cfg = resolve_model_config(model_name)

    if cfg.type == "scripted":
        script_dir = Path(cfg.script_dir)
        scripts = load_episode_scripts(script_dir)

        def scripted_factory(episode_id: str) -> ScriptedModel:
            return ScriptedModel(episode_id, list(scripts.get(episode_id, [])))

        return scripted_factory

    if cfg.type == "openai_compatible":
        if not cfg.base_url or not cfg.model:
            raise ValueError(f"model '{model_name}': openai_compatible requires base_url and model")
        api_key = os.environ.get(cfg.api_key_env or "GLM_API_KEY", "")
        if not api_key:
            raise ValueError(f"model '{model_name}': set {cfg.api_key_env} (see .env.example); refusing to run")

        def openai_factory(episode_id: str) -> OpenAICompatModel:
            return OpenAICompatModel(
                name=f"{cfg.model}:{episode_id}", base_url=cfg.base_url, model=cfg.model,
                api_key=api_key, request_timeout=request_timeout, temperature=cfg.temperature,
                wire=cfg.wire, max_tokens=cfg.max_tokens,
                chat_template_kwargs=cfg.chat_template_kwargs,
            )

        return openai_factory

    raise ValueError(f"model '{model_name}': unsupported type '{cfg.type}'")
