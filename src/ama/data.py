"""AMA Dataset: Episode → Turn → Evidence → Decision.

The run loader never reads evaluator-only targets, eval rules, or provenance.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def nonempty_id(cls, value: Any) -> Any:
        if isinstance(value, dict) and "id" in value and (
                not isinstance(value["id"], str) or not value["id"].strip()):
            raise ValueError("id must be a non-empty string")
        return value


class Evidence(_Model):
    id: str
    type: str | None = None
    text: str | None = None
    file: str | None = None

    @model_validator(mode="before")
    @classmethod
    def omit_missing_optional_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(key in value and (value[key] is None or
                                           isinstance(value[key], str) and not value[key].strip())
                                           for key in ("type", "text", "file")):
            raise ValueError("omit absent optional evidence fields instead of null or blank")
        return value

    @model_validator(mode="after")
    def has_content(self) -> "Evidence":
        if not (self.text and self.text.strip()) and not self.file:
            raise ValueError("evidence has neither text nor file")
        return self


class Turn(_Model):
    id: str
    available_at: datetime | None = None
    observation: str | None = None
    evidence: list[Evidence]

    @model_validator(mode="before")
    @classmethod
    def omit_missing_optional_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(key in value and (value[key] is None or
                                           isinstance(value[key], str) and not value[key].strip())
                                           for key in ("available_at", "observation")):
            raise ValueError("omit absent optional turn fields instead of null or blank")
        return value


class Episode(_Model):
    id: str
    turns: list[Turn]


class Decision(_Model):
    turn_id: str
    answer: Any
    citations: list[str]
    reasoning_summary: str = ""


class DatasetInfo(_Model):
    schema_name: str = Field(alias="schema")
    name: str
    splits: dict[str, list[str]]

    @model_validator(mode="after")
    def known_schema(self) -> "DatasetInfo":
        if self.schema_name != "ama-dataset":
            raise ValueError(f"unsupported dataset schema {self.schema_name!r}; expected ama-dataset")
        return self


class TargetRecord(_Model):
    """One line of targets.jsonl. Turn payloads are scorer-interpreted dicts."""

    model_config = ConfigDict(extra="forbid")
    id: str
    turns: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvalConfig(_Model):
    scorer: str = "unscored"
    rules: dict[str, Any] = Field(default_factory=dict)


class Dataset:
    def __init__(self, folder: Path, info: DatasetInfo, episodes: list[Episode],
                 targets: dict[str, TargetRecord], eval_config: EvalConfig | None) -> None:
        self.folder = folder
        self.info = info
        self.episodes = episodes
        self.targets = targets
        self.eval_config = eval_config

    def episode(self, episode_id: str) -> Episode | None:
        return next((e for e in self.episodes if e.id == episode_id), None)

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
            return [e for e in self.episodes if e.id in set(ids)]
        return self.episodes


def _safe_file(dataset_dir: Path, file: str) -> bool:
    raw = Path(file)
    if raw.is_absolute():
        return False
    resolved = (dataset_dir / raw).resolve()
    return resolved == dataset_dir.resolve() or dataset_dir.resolve() in resolved.parents


def load_dataset(folder: Path, with_targets: bool = False) -> Dataset:
    """The run path reads only dataset.json and episodes.jsonl."""
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
                    if rec.id in targets:
                        raise ValueError(f"duplicate target episode id: {rec.id}")
                    targets[rec.id] = rec
    eval_config = None
    if with_targets and (folder / "eval.json").exists():
        eval_config = EvalConfig.model_validate(json.loads((folder / "eval.json").read_text(encoding="utf-8")))
    return Dataset(folder, info, episodes, targets, eval_config)


def validate_visible_files(dataset: Dataset) -> None:
    """Run-safe structural validation; never opens evaluator or provenance files."""
    ids = [episode.id for episode in dataset.episodes]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("episode IDs must be unique and non-empty")
    for episode in dataset.episodes:
        if not episode.turns:
            raise ValueError(f"{episode.id}: no turns")
        turn_ids = [turn.id for turn in episode.turns]
        evidence_ids = [evidence.id for turn in episode.turns for evidence in turn.evidence]
        if len(turn_ids) != len(set(turn_ids)) or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"{episode.id}: duplicate turn or evidence IDs")
        times = [turn.available_at for turn in episode.turns if turn.available_at is not None]
        try:
            if any(later < earlier for earlier, later in zip(times, times[1:])):
                raise ValueError(f"{episode.id}: non-monotonic available_at")
        except TypeError as exc:
            raise ValueError(f"{episode.id}: inconsistent timestamp timezones") from exc
        for turn in episode.turns:
            for evidence in turn.evidence:
                if evidence.file and (not _safe_file(dataset.folder, evidence.file)
                                      or not (dataset.folder / evidence.file).is_file()):
                    raise ValueError(f"{episode.id}: unsafe or missing evidence file {evidence.file!r}")
    for split, members in dataset.info.splits.items():
        missing = set(members) - set(ids)
        if missing:
            raise ValueError(f"split {split!r} references unknown episodes: {sorted(missing)}")


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

    ids = [e.id for e in ds.episodes]
    if len(set(ids)) != len(ids):
        errors.append("duplicate episode ids")
    if not ds.episodes:
        errors.append("no episodes")
    for ep in ds.episodes:
        if not ep.turns:
            errors.append(f"{ep.id}: no turns")
            continue
        turn_ids = [t.id for t in ep.turns]
        if len(set(turn_ids)) != len(turn_ids):
            errors.append(f"{ep.id}: duplicate turn ids")
        times = [t.available_at for t in ep.turns]
        known_times = [t for t in times if t is not None]
        try:
            if any(b < a for a, b in zip(known_times, known_times[1:])):
                errors.append(f"{ep.id}: non-monotonic available_at")
        except TypeError:
            errors.append(f"{ep.id}: inconsistent available_at timezone awareness")
        ev_ids = [e.id for t in ep.turns for e in t.evidence]
        if len(set(ev_ids)) != len(ev_ids):
            errors.append(f"{ep.id}: duplicate evidence ids")
        for e in (ev for t in ep.turns for ev in t.evidence):
            if e.file is not None:
                if not _safe_file(folder, e.file):
                    errors.append(f"{ep.id}: evidence {e.id} unsafe file path '{e.file}'")
                elif not (folder / e.file).is_file():
                    errors.append(f"{ep.id}: evidence {e.id} file missing '{e.file}'")
    for split, split_ids in ds.info.splits.items():
        known = set(ids)
        for sid in split_ids:
            if sid not in known:
                errors.append(f"split '{split}' references unknown episode '{sid}'")
    for tid in ds.targets:
        if tid not in set(ids):
            errors.append(f"targets reference unknown episode '{tid}'")
    for ep in ds.episodes:
        target = ds.targets.get(ep.id)
        if target:
            known_turns = {turn.id for turn in ep.turns}
            for turn_id in target.turns:
                if turn_id not in known_turns:
                    errors.append(f"{ep.id}: target references unknown turn '{turn_id}'")
    return errors
