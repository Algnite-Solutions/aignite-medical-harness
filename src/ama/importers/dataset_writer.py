"""Shared dataset writer for the built-in source-specific importers.

Importers may construct rich intermediate records; only the small model-visible
contract is emitted to episodes.jsonl. Source locators go to provenance.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_dataset(out: Path, *, name: str, splits: dict[str, list[str]],
                     episodes: list[dict[str, Any]], targets: list[dict[str, Any]],
                     scorer: str = "unscored", rules: dict[str, Any] | None = None,
                     instruction: str = "", retain_times: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    visible: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    target_by_id = {t.get("episode_id", t.get("id")): t for t in targets}
    for episode in episodes:
        eid = episode.get("episode_id", episode.get("id"))
        item: dict[str, Any] = {"id": eid, "turns": []}
        trace: dict[str, Any] = {"id": eid, "subject_id": episode.get("subject_id"),
                                  "metadata": episode.get("metadata", {}), "turns": []}
        if "reference" in target_by_id.get(eid, {}):
            trace["reference"] = target_by_id[eid]["reference"]
        for turn in episode["turns"]:
            tid = turn.get("turn_id", turn.get("id"))
            stage: dict[str, Any] = {"id": tid, "evidence": []}
            observation = turn.get("message", turn.get("observation"))
            if observation:
                stage["observation"] = observation
            timestamp = turn.get("time", turn.get("available_at"))
            if retain_times and timestamp:
                stage["available_at"] = timestamp
            trace_turn: dict[str, Any] = {"id": tid, "evidence": []}
            if retain_times and timestamp:
                trace_turn["source_time"] = timestamp
            for evidence in turn.get("evidence", []):
                vid = evidence.get("evidence_id", evidence.get("id"))
                block: dict[str, Any] = {"id": vid}
                kind = evidence.get("kind", evidence.get("type"))
                if kind:
                    block["type"] = kind
                if evidence.get("text"):
                    block["text"] = evidence["text"]
                artifact = evidence.get("artifact", evidence.get("file"))
                if artifact:
                    block["file"] = artifact
                stage["evidence"].append(block)
                trace_turn["evidence"].append({"id": vid, "source": evidence.get("source"),
                                                "metadata": evidence.get("metadata", {})})
            item["turns"].append(stage)
            trace["turns"].append(trace_turn)
        visible.append(item)
        provenance.append(trace)

    converted_targets = []
    for target in targets:
        turn_targets = {}
        for tid, payload in target.get("turns", {}).items():
            value = dict(payload)
            if "state" in value:
                value["answer"] = value.pop("state")
            turn_targets[tid] = value
        converted_targets.append({"id": target.get("episode_id", target.get("id")),
                                  "turns": turn_targets})

    (out / "dataset.json").write_text(json.dumps({"schema": "ama-dataset", "name": name,
                                                    "splits": splits}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    files = [("episodes.jsonl", visible), ("provenance.jsonl", provenance)]
    if converted_targets:
        files.append(("targets.jsonl", converted_targets))
    for filename, rows in files:
        (out / filename).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                                    + ("\n" if rows else ""), encoding="utf-8")
    (out / "eval.json").write_text(json.dumps({"scorer": scorer, "rules": rules or {}},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    if instruction:
        (out / "instructions.txt").write_text(instruction, encoding="utf-8")
