"""AMA Dataset v0: the only data standard. Episode → Turn → Evidence → Decision."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def resolve_dataset_dir(value: str | Path) -> Path:
    """Resolve an existing path, or a dataset name below AMA_DATA_ROOT."""
    direct = Path(value).expanduser()
    if direct.exists():
        return direct.resolve()
    root = os.environ.get("AMA_DATA_ROOT")
    if root and not direct.is_absolute():
        root_path = Path(root).expanduser().resolve()
        candidate = (root_path / direct).resolve()
        if root_path in candidate.parents and candidate.exists():
            return candidate
    detail = f"dataset not found: {value}"
    if root:
        detail += f" (also checked below {Path(root).expanduser()})"
    else:
        detail += " (set AMA_DATA_ROOT to use a dataset name)"
    raise FileNotFoundError(detail)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Evidence(_Model):
    evidence_id: str
    kind: str
    text: str = ""
    artifact: str | None = None  # relative path inside the dataset dir
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Turn(_Model):
    turn_id: str
    time: datetime | None = None  # single-turn episodes may omit; multi-turn must be monotonic
    message: str
    evidence: list[Evidence] = Field(default_factory=list)


class Episode(_Model):
    episode_id: str
    subject_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)


class DecisionAction(_Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Decision(_Model):
    turn_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    action: DecisionAction | None = None
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    note: str = ""


class DatasetInfo(_Model):
    schema_version: str = Field(default="ama-dataset-v0", alias="schema")
    name: str
    version: str = "0.1"
    description: str = ""
    splits: dict[str, list[str]] = Field(default_factory=dict)
    scorer: str | None = None  # None => unscored runs (trace + operational metrics only)
    license: str = ""


class TargetRecord(_Model):
    """One line of targets.jsonl. Turn payloads are scorer-interpreted dicts."""

    model_config = ConfigDict(extra="allow")
    episode_id: str
    turns: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Policy(_Model):
    """policy.json: loader-enforced public/hidden split."""

    public: dict[str, Any] = Field(default_factory=dict)
    hidden: dict[str, Any] = Field(default_factory=dict)


class Dataset:
    def __init__(self, folder: Path, info: DatasetInfo, episodes: list[Episode],
                 targets: dict[str, TargetRecord], policy: Policy) -> None:
        self.folder = folder
        self.info = info
        self.episodes = episodes
        self.targets = targets
        self.policy = policy

    def episode(self, episode_id: str) -> Episode | None:
        return next((e for e in self.episodes if e.episode_id == episode_id), None)

    def select(self, split: str | None = None, episode_id: str | None = None) -> list[Episode]:
        if episode_id:
            ep = self.episode(episode_id)
            if ep is None:
                raise KeyError(f"episode not found: {episode_id}")
            return [ep]
        if split:
            ids = self.info.splits.get(split)
            if ids is None:
                raise KeyError(f"unknown split: {split} (have {list(self.info.splits)})")
            return [e for e in self.episodes if e.episode_id in set(ids)]
        return self.episodes


def _safe_artifact(dataset_dir: Path, artifact: str) -> bool:
    raw = Path(artifact)
    if raw.is_absolute():
        return False
    resolved = (dataset_dir / raw).resolve()
    return resolved == dataset_dir.resolve() or dataset_dir.resolve() in resolved.parents


def load_dataset(folder: Path, with_targets: bool = False) -> Dataset:
    """Load a dataset directory. `with_targets=False` (the run path) NEVER touches targets.jsonl."""
    folder = Path(folder)
    info = DatasetInfo.model_validate(json.loads((folder / "dataset.json").read_text(encoding="utf-8")))
    episodes = [
        Episode.model_validate(json.loads(line))
        for line in (folder / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    targets: dict[str, TargetRecord] = {}
    if with_targets:
        tpath = folder / "targets.jsonl"
        if tpath.exists():
            for line in tpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = TargetRecord.model_validate(json.loads(line))
                    targets[rec.episode_id] = rec
    policy = Policy()
    ppath = folder / "policy.json"
    if ppath.exists():
        policy = Policy.model_validate(json.loads(ppath.read_text(encoding="utf-8")))
    return Dataset(folder, info, episodes, targets, policy)


def validate_dataset(folder: Path) -> list[str]:
    """Fully-offline structural validation; empty list = OK."""
    folder = Path(folder)
    errors: list[str] = []
    if not (folder / "dataset.json").exists():
        return [f"missing dataset.json in {folder}"]
    try:
        ds = load_dataset(folder, with_targets=True)
    except Exception as exc:
        return [f"load failed: {exc}"]

    ids = [e.episode_id for e in ds.episodes]
    if len(set(ids)) != len(ids):
        errors.append("duplicate episode ids")
    if not ds.episodes:
        errors.append("no episodes")
    for ep in ds.episodes:
        if not ep.turns:
            errors.append(f"{ep.episode_id}: no turns")
            continue
        turn_ids = [t.turn_id for t in ep.turns]
        if len(set(turn_ids)) != len(turn_ids):
            errors.append(f"{ep.episode_id}: duplicate turn ids")
        if len(ep.turns) > 1:
            times = [t.time for t in ep.turns]
            if any(t is None for t in times):
                errors.append(f"{ep.episode_id}: multi-turn episode with null turn time")
            else:
                for a, b in zip(times, times[1:]):
                    if b < a:
                        errors.append(f"{ep.episode_id}: non-monotonic turn time at {b}")
        ev_ids = [e.evidence_id for t in ep.turns for e in t.evidence]
        if len(set(ev_ids)) != len(ev_ids):
            errors.append(f"{ep.episode_id}: duplicate evidence ids")
        for e in (ev for t in ep.turns for ev in t.evidence):
            if not e.text and not e.artifact:
                errors.append(f"{ep.episode_id}: evidence {e.evidence_id} has neither text nor artifact")
            if e.artifact is not None:
                if not _safe_artifact(folder, e.artifact):
                    errors.append(f"{ep.episode_id}: evidence {e.evidence_id} unsafe artifact path '{e.artifact}'")
                elif not (folder / e.artifact).exists():
                    errors.append(f"{ep.episode_id}: evidence {e.evidence_id} artifact missing '{e.artifact}'")
    for split, split_ids in ds.info.splits.items():
        known = set(ids)
        for sid in split_ids:
            if sid not in known:
                errors.append(f"split '{split}' references unknown episode '{sid}'")
    for tid in ds.targets:
        if tid not in set(ids):
            errors.append(f"targets reference unknown episode '{tid}'")
    return errors
