"""Small shared helpers: content redaction policy (Gate 0 sensitive fields)."""
from __future__ import annotations

import hashlib
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def redact_content(obj: Any, keep: bool) -> Any:
    """When keep=False, replace text fields (content/report) with digests in trace
    copies. The model always sees the full content; only persisted traces are redacted."""
    if keep or obj is None:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("content", "report") and isinstance(v, str) and v:
                out[k] = f"<redacted sha256={digest(v)}>"
            else:
                out[k] = redact_content(v, keep)
        return out
    if isinstance(obj, list):
        return [redact_content(x, keep) for x in obj]
    return obj


def redact_messages(messages: list[Any], keep: bool) -> Any:
    """Redact the whole model-context block to a digest summary when keep=False."""
    if keep:
        return messages
    return f"<redacted model_context: {len(messages)} messages>"
