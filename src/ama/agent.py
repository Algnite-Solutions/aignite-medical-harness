"""最小循环：输入 → 模型 → 可选工具 → 最终回复。与数据集和评分无关。"""
from __future__ import annotations

import copy
import json
import time

from .model import ModelError, content, image
from .tools import Tool


class CallLimitExceeded(RuntimeError):
    pass


class Agent:
    def __init__(self, model, system: str = "", tools: list[Tool] | None = None, max_calls: int = 8,
                 on_event=None):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.model, self.max_calls = model, max_calls
        self.tools = {tool.name: tool for tool in (tools or [])}
        if len(self.tools) != len(tools or []):
            raise ValueError("duplicate tool names")
        self.history = [{"role": "system", "content": system}] if system else []
        self.calls: list[dict] = []
        self.failed = False
        self.on_event = on_event
        self.evidence_ids: set[str] = set()
        self.evidence_releases: list[dict] = []

    def _emit(self, kind: str, value=None) -> None:
        if self.on_event is not None:
            self.on_event(kind, value)

    def chat(self, message: str | list) -> str:
        if self.failed:
            raise RuntimeError("session has stopped; create a new Agent")
        self.history.append({"role": "user", "content": content(message)})
        definitions = [tool.definition() for tool in self.tools.values()]
        try:
            for _ in range(self.max_calls):
                # 1. 原样发出历史；记录每次调用，包括失败和中断。
                started = time.monotonic()
                call = {"message_index": len(self.history), "usage": None, "error": None}
                try:
                    self._emit("model_start")
                    reply = self.model.complete(self.history, tools=definitions or None)
                    if not isinstance(reply, dict):
                        raise ModelError("assistant message must be an object")
                    reply = copy.deepcopy(reply)
                    reply.setdefault("role", "assistant")
                    self.history.append(reply)
                except BaseException as exc:
                    call["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, ModelError) and exc.http_status is not None:
                        call["http_status"] = exc.http_status
                        if exc.rate_limit_headers:
                            call["rate_limit_headers"] = exc.rate_limit_headers
                    raise
                finally:
                    call["duration_ms"] = int((time.monotonic() - started) * 1000)
                    call["usage"] = getattr(self.model, "last_usage", None)
                    self.calls.append(call)

                # 2. 没有工具调用，就是本次 chat 的最终回答。
                if reply["role"] != "assistant":
                    raise ModelError("expected assistant role")
                requests = reply.get("tool_calls") or []
                if not requests:
                    if not isinstance(reply.get("content"), str):
                        raise ModelError("final reply must contain text")
                    return reply["content"]
                if not isinstance(requests, list):
                    raise ModelError("tool_calls must be a list")
                ids = [r.get("id") if isinstance(r, dict) else None for r in requests]
                if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                    raise ModelError("tool calls require unique non-empty IDs")

                # 3. 每个调用都先有对应 tool 消息，再附图像，最后回到模型。
                attachments = []
                for request in requests:
                    function = request.get("function") or {}
                    self._emit("tool_call", function)
                    try:
                        if request.get("type") != "function":
                            raise ValueError("unsupported tool call type")
                        name = function["name"]
                        if name not in self.tools:
                            raise ValueError(f"unknown tool: {name}")
                        result = self.tools[name].invoke(function.get("arguments", "{}"))
                        text = result.text or "Image result attached."
                        for path in result.images:
                            attachments.extend([f"Tool image: call_id={request['id']}, name={name}", image(path)])
                        if result.evidence_ids:
                            text += "\nCiteable evidence IDs: " + json.dumps(result.evidence_ids)
                            self.evidence_ids.update(result.evidence_ids)
                            self.evidence_releases.append({"tool_call_id": request["id"],
                                                           "name": name, "evidence_ids": result.evidence_ids})
                    except Exception as exc:
                        text = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
                    self.history.append({"role": "tool", "tool_call_id": request["id"], "content": text})
                    self._emit("tool_result", text)
                if attachments:
                    self.history.append({"role": "user", "content": content(attachments)})
            raise CallLimitExceeded(f"maximum model calls per chat reached: {self.max_calls}")
        except BaseException:
            self.failed = True
            raise
