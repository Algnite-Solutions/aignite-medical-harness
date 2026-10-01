"""Run configuration, atomic conversation snapshots, and append-only diagnostics."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Recorder:
    def __init__(self, root: Path, *, dataset, episodes, mode, model,
                 max_calls, tools_path=None):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        self.run_dir = Path(root).resolve() / run_id
        self.run_dir.mkdir(parents=True)
        self.manifest = {
            "run_id": run_id, "mode": mode, "dataset_dir": str(dataset.folder.resolve()),
            "episode_ids": [ep.id for ep in episodes],
            "expected_turns": {ep.id: [t.id for t in ep.turns] for ep in episodes},
            "model": model.name, "model_config": model.config, "timeout": model.timeout,
            "max_calls": max_calls,
            "tools": str(Path(tools_path).resolve()) if tools_path else None,
            "termination": "running",
        }
        if mode == "run":
            self.manifest["output_contract_version"] = 2
        self.manifest["log_schema_version"] = 3
        write_json(self.run_dir / "manifest.json", self.manifest)
        self.messages: dict[str, list[dict]] = {}
        write_json(self.run_dir / "messages.json", self.messages)
        (self.run_dir / "diagnostics.jsonl").touch()
        if mode == "run":
            (self.run_dir / "decisions.jsonl").touch()

    def append(self, filename, value):
        with (self.run_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def event(self, kind, **fields):
        self.append("diagnostics.jsonl", {"kind": kind, **fields})

    def history(self, agent, *, episode_id):
        self.messages[episode_id] = agent.history
        write_json(self.run_dir / "messages.json", self.messages)

    def finish(self, termination):
        self.manifest["termination"] = termination
        write_json(self.run_dir / "manifest.json", self.manifest)
