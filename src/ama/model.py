"""模型接入：配置别名 → 原样消息 → API；图像只在发送时编码。"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class ModelError(RuntimeError):
    pass


def image(path: str | Path) -> dict:
    """历史中保留可读的路径，不放 base64。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"image not found: {path}")
    if mimetypes.guess_type(path.name)[0] not in IMAGE_TYPES:
        raise ValueError(f"unsupported image type: {path.name}")
    return {"type": "image", "path": str(path)}


def content(value: str | list) -> str | list[dict]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("message must be text or a list of text/image parts")
    parts = []
    for part in value:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, dict) and part.get("type") == "image":
            parts.append(image(part["path"]))
        elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(dict(part))
        else:
            raise ValueError("unsupported message part")
    return parts


def wire_messages(history: list[dict]) -> list[dict]:
    """只编码图像；不改变消息顺序、角色或工具调用关系。"""
    result = []
    for message in history:
        wire = dict(message)
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part["type"] == "image":
                    path = Path(part["path"])
                    mime = mimetypes.guess_type(path.name)[0]
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    parts.append({"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{encoded}"}})
                else:
                    parts.append(dict(part))
            wire["content"] = parts
        result.append(wire)
    return result


class Model:
    def __init__(self, name: str, config: dict, timeout: float = 60):
        allowed = {"base_url", "model", "api_key_env", "temperature", "max_tokens", "chat_template_kwargs"}
        if set(config) - allowed:
            raise ValueError(f"unsupported model options: {sorted(set(config) - allowed)}")
        for key in ("base_url", "model", "api_key_env"):
            if not config.get(key):
                raise ValueError(f"missing model option: {key}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.name, self.config, self.timeout = name, dict(config), timeout
        self.last_usage = None
        self._key = os.environ.get(config["api_key_env"])
        if not self._key:
            raise ValueError(f"missing API key environment variable: {config['api_key_env']}")

    @classmethod
    def from_config(cls, name: str, config_path: str | Path = "ama.json", timeout: float = 60):
        models = json.loads(Path(config_path).read_text(encoding="utf-8"))["models"]
        if name not in models:
            raise ValueError(f"unknown model alias: {name}")
        return cls(name, models[name], timeout)

    def complete(self, history: list[dict], tools: list[dict] | None = None) -> dict:
        self.last_usage = None
        payload = {"model": self.config["model"], "messages": wire_messages(history),
                   "temperature": self.config.get("temperature", 0)}
        for key in ("max_tokens", "chat_template_kwargs"):
            if key in self.config:
                payload[key] = self.config[key]
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            hint = ""
            if exc.code in {400, 415, 422} and any(m["role"] == "tool" for m in history) and any(
                    isinstance(m.get("content"), list) and any(
                        isinstance(part, dict) and part.get("type") == "image"
                        for part in m["content"]) for m in history):
                hint = "; request contains tools and images; check service support (history was not rewritten)"
            raise ModelError(f"model endpoint returned HTTP {exc.code}{hint}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"model connection failed: {type(exc).__name__}") from None
        except (ValueError, UnicodeError):
            raise ModelError("model endpoint returned invalid JSON") from None
        try:
            self.last_usage = data.get("usage")
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError()
            return message
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ModelError("model response has no assistant message") from None
