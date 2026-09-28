"""Append-only experiment records. Messages keep paths; never encode images here."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Recorder:
    def __init__(self, root: Path, *, dataset, episodes, mode, model, instruction, system,
                 max_calls, tools_path=None):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        self.run_dir = Path(root).resolve() / run_id
        self.run_dir.mkdir(parents=True)
        # Only visible inputs: never even hash targets, eval rules or provenance here.
        files = {"dataset.json", "episodes.jsonl"}
        files.update(e.file for ep in episodes for t in ep.turns for e in t.evidence if e.file)
        # Dataset-side files read by trusted tools are declared without executing dataset code.
        tool_data_manifest = dataset.folder / "tool_data_files.json"
        if tools_path and tool_data_manifest.exists():
            files.add("tool_data_files.json")
            for name in json.loads(tool_data_manifest.read_text(encoding="utf-8")):
                if not isinstance(name, str):
                    raise ValueError(f"unsafe tool data file: {name!r}")
                candidate = (dataset.folder / name).resolve()
                if dataset.folder.resolve() not in candidate.parents \
                        or not candidate.is_file():
                    raise ValueError(f"unsafe tool data file: {name!r}")
                files.add(name)
        self.manifest = {
            "run_id": run_id, "mode": mode, "dataset_dir": str(dataset.folder.resolve()),
            "episode_ids": [ep.id for ep in episodes],
            "expected_turns": {ep.id: [t.id for t in ep.turns] for ep in episodes},
            "dataset_sha256": {f: sha256_file(dataset.folder / f) for f in sorted(files)},
            "model": model.name, "model_config": model.config, "timeout": model.timeout,
            "max_calls": max_calls, "instruction": instruction, "system": system,
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "tools": {"file": str(Path(tools_path).resolve()), "sha256": sha256_file(Path(tools_path))}
            if tools_path else None,
            "termination": "running",
        }
        if mode == "run":
            self.manifest["output_contract_version"] = 2
        write_json(self.run_dir / "manifest.json", self.manifest)
        (self.run_dir / "events.jsonl").touch()
        if mode == "run":
            (self.run_dir / "decisions.jsonl").touch()

    def append(self, filename, value):
        with (self.run_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def event(self, kind, **fields):
        self.append("events.jsonl", {"kind": kind, **fields})

    def history(self, agent, start, *, source, episode_id, turn_id):
        for index in range(start, len(agent.history)):
            message = agent.history[index]
            origin = source if index == start and message["role"] == "user" else {
                "assistant": "model", "system": "system", "tool": "tool", "user": "tool",
            }.get(message["role"], "model")
            self.event("message", episode_id=episode_id, turn_id=turn_id,
                       index=index, source=origin, message=message)

    def finish(self, termination):
        self.manifest["termination"] = termination
        write_json(self.run_dir / "manifest.json", self.manifest)
