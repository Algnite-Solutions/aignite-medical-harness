"""工具是显式的 schema + Python 函数；不从模型文本执行代码。"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .model import image

_MODULE_CACHE: dict[Path, object] = {}


@dataclass
class ToolResult:
    text: str = ""
    images: list[str | Path] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable

    def definition(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def invoke(self, arguments: str) -> ToolResult:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        inspect.signature(self.handler).bind(**args)
        result = self.handler(**args)
        if not isinstance(result, ToolResult):
            result = ToolResult(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))
        if not isinstance(result.text, str):
            raise ValueError("ToolResult.text must be a string")
        if not isinstance(result.evidence_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in result.evidence_ids):
            raise ValueError("tool evidence IDs must be non-empty strings")
        # 在记为成功工具结果之前检查图像路径；错误可回传模型修正。
        return ToolResult(result.text, [image(path)["path"] for path in result.images],
                          list(dict.fromkeys(result.evidence_ids)))


def load_tools(path: str | Path | None, *, dataset_dir: Path | None = None,
               episode_id: str | None = None) -> list[Tool]:
    if path is None:
        return []
    resolved = Path(path).resolve()
    module = _MODULE_CACHE.get(resolved)
    if module is None:
        spec = importlib.util.spec_from_file_location("_ama_user_tools", resolved)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load tools file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[resolved] = module
    factory = getattr(module, "make_tools", None)
    tools = factory(dataset_dir, episode_id) if callable(factory) else getattr(module, "TOOLS", None)
    if not isinstance(tools, list) or not all(isinstance(t, Tool) for t in tools):
        raise ValueError("tools file must export TOOLS: list[Tool]")
    return tools
