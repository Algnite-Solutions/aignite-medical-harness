"""Recorder: run directory, events/decisions JSONL, manifest (git + hashes + usage),
redaction, and the single report writer. Targets are NEVER opened here."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pydantic

SRC_ROOT = Path(__file__).resolve().parent


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def redact_content(obj: Any, keep: bool) -> Any:
    if keep or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: (f"<redacted sha256={digest(v)}>" if k in ("text", "message", "note") and isinstance(v, str) and v
                    else redact_content(v, keep))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_content(x, keep) for x in obj]
    return obj


def redact_messages(messages: list[Any], keep: bool) -> Any:
    if keep:
        return messages
    return f"<redacted model_context: {len(messages)} messages>"


def sanitize_multimodal_content(obj: Any) -> Any:
    """Remove inline binary payloads before content reaches any persistent log."""
    if isinstance(obj, dict):
        if obj.get("type") == "image_url" and isinstance(obj.get("image_url"), dict):
            url = obj["image_url"].get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                mime = url[5:].split(";", 1)[0] or "unknown"
                return {"type": "image_url", "image_url": {"url": f"<inline {mime} omitted>"}}
        return {k: sanitize_multimodal_content(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_multimodal_content(v) for v in obj]
    return obj


def code_version() -> dict[str, Any]:
    repo = SRC_ROOT.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                    text=True, timeout=10, check=True).stdout.strip())
        return {"vcs": "git", "git_commit": commit[:12], "git_dirty": dirty}
    except Exception:
        h = hashlib.sha256()
        for path in sorted(SRC_ROOT.rglob("*.py")):
            h.update(path.read_bytes())
        return {"vcs": "not-a-git-repo", "source_sha256": h.hexdigest()[:16]}


def dependency_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "pydantic": pydantic.__version__}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EventLog:
    def __init__(self, fh: TextIO, run_id: str) -> None:
        self._fh = fh
        self.run_id = run_id
        self.seq = 0

    def __call__(self, type_: str, **fields: Any) -> None:
        self.seq += 1
        row = {"seq": self.seq, "run_id": self.run_id,
               "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "type": type_, **fields}
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()


class TraceConfig:
    def __init__(self, save_model_context: bool = True, save_evidence_text: bool = True) -> None:
        self.save_model_context = save_model_context
        self.save_evidence_text = save_evidence_text


class Budget:
    def __init__(self, max_model_calls: int, per_turn_model_calls: int | None,
                 deadline_seconds: float, max_retries: int) -> None:
        self.max_model_calls = max_model_calls
        self.per_turn_model_calls = per_turn_model_calls
        self.deadline_seconds = deadline_seconds
        self.max_retries = max_retries
        self.total_calls = 0
        self.call_seq = 0
        self._start = time.monotonic()

    def check(self, turn_calls: int) -> str | None:
        if self.per_turn_model_calls is not None and turn_calls >= self.per_turn_model_calls:
            return "per_turn_model_calls"
        if self.total_calls >= self.max_model_calls:
            return "max_model_calls"
        if time.monotonic() - self._start > self.deadline_seconds:
            return "deadline"
        return None

    def exclude_wait(self, seconds: float) -> None:
        self._start += seconds  # operator waiting never eats the deadline


class Recorder:
    """One run directory: manifest (no targets), events.jsonl, decisions.jsonl, report."""

    def __init__(self, runs_root: Path, name: str, model_name: str, dataset_dir: Path,
                 episode_ids: list[str], trace: TraceConfig, budget: Budget,
                 experiment: dict[str, Any] | None = None, interactive: bool = False,
                 model_info: dict[str, Any] | None = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        import secrets

        self.run_dir = Path(runs_root) / f"{stamp}-{name}-{secrets.token_hex(2)}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.run_id = self.run_dir.name
        self.trace = trace
        self.budget = budget
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        # Hash only model-visible input; never read evaluator or provenance files.
        self.manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model_name,
            # resolved provider details keep historical runs auditable even if the alias
            # in ama.json is later repointed (the wire option materially changes what the
            # model sees); secrets (API keys) are never included
            "model_resolved": model_info or {"alias": model_name},
            "dataset_dir": str(self.dataset_dir),
            "episode_ids": episode_ids,
            "interactive": interactive,
            "code_version": code_version(),
            "dependencies": dependency_versions(),
            "dataset_hashes": {
                name: sha256_file(self.dataset_dir / name)
                for name in ("dataset.json", "episodes.jsonl")
            },
            "experiment": experiment,
            "trace": {"save_model_context": trace.save_model_context, "save_evidence_text": trace.save_evidence_text},
            "budget": {"max_model_calls": budget.max_model_calls,
                       "per_turn_model_calls": budget.per_turn_model_calls,
                       "deadline_seconds": budget.deadline_seconds,
                       "max_retries": budget.max_retries},
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._ev_fh = open(self.run_dir / "events.jsonl", "w", encoding="utf-8")
        self._dec_fh = open(self.run_dir / "decisions.jsonl", "w", encoding="utf-8")
        self.log = EventLog(self._ev_fh, self.run_id)
        self.decisions: list[dict[str, Any]] = []

    def save_decision(self, row: dict[str, Any]) -> None:
        self.decisions.append(row)
        self._dec_fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._dec_fh.flush()

    def finalize(self, episode_summaries: list[dict[str, Any]], operational: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps({"model": self.model_name, "operational": operational,
                        "episodes": episode_summaries}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        lines = [
            f"# AMA run — {self.run_id}",
            "",
            f"- model: **{self.model_name}**（scripted 满分只证明运行器与评分器工作，不代表真实模型效果）",
            f"- dataset: `{self.dataset_dir}` | episodes: {len(self.manifest['episode_ids'])} | interactive: {self.manifest['interactive']}",
            f"- git: {self.manifest['code_version'].get('git_commit', 'n/a')} dirty={self.manifest['code_version'].get('git_dirty', 'n/a')}",
            "",
            "## Episodes",
            "",
            "| episode | termination | turns | calls | tokens |",
            "|---|---|---|---|---|",
        ]
        for e in episode_summaries:
            tokens = e.get("usage", {}).get("total_tokens") if isinstance(e.get("usage"), dict) else e.get("usage")
            lines.append(f"| {e['episode_id']} | {e['termination']} | {e.get('turns_decided', 0)}/{e.get('turns', 0)} | {e.get('model_calls', 0)} | {tokens if tokens is not None else 'unknown'} |")
        lines += ["", "尚未评分：运行 `ama eval runs/<run-id>`（eval 才读取 targets.jsonl）。", ""]
        (self.run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        self.log("run_end", operational=operational)
        self._ev_fh.close()
        self._dec_fh.close()


def merge_usage(total: dict | None, usage: Any) -> dict | None:
    if not isinstance(usage, dict):
        return total
    base = total or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base[key] = base.get(key, 0) + (usage.get(key) or 0)
    return base
