"""Paced, resumable runs for a fixed MIMIC-CDM open benchmark view."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ama.cli import _eval_run, _load_dotenv
from ama.data import load_dataset
from ama.model import Model, ModelError
from ama.recorder import write_json
from ama.runner import execute


class PacedModel(Model):
    """Retry transport failures only; never revise a model answer."""

    def complete(self, history, tools=None):
        for attempt in range(6):
            time.sleep(self.request_delay)
            try:
                return super().complete(history, tools)
            except ModelError as exc:
                connection = str(exc).startswith("model connection failed:")
                if exc.http_status not in {429, 500, 502, 503, 504} and not connection:
                    raise
                max_attempts = 6 if exc.http_status == 429 else 3
                retry_after = (exc.rate_limit_headers or {}).get("retry-after")
                try:
                    wait = max(0, min(float(retry_after), 240)) if retry_after else None
                except ValueError:
                    wait = None
                if wait is None:
                    wait = (min(30 * 2 ** attempt, 240) if exc.http_status == 429
                            else min(5 * 2 ** attempt, 60))
                with self.retry_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": time.time(), "attempt": attempt + 1,
                                             "wait_seconds": wait if attempt + 1 < max_attempts else 0,
                                             "http_status": exc.http_status,
                                             "connection_error": connection}) + "\n")
                if attempt + 1 == max_attempts:
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(path: Path) -> dict:
    rows = [json.loads(line) for line in (path / "decisions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one decision: {path}")
    return rows[0]


def merge_batch(root: Path, batch: dict) -> Path:
    """Create a standard run from saved one-case runs; never edit the originals."""
    ids = batch["episode_ids"]
    paths = {case_id: Path(batch["runs"][case_id]) for case_id in ids}
    rows = [_decision(paths[case_id]) for case_id in ids]
    if [row["episode_id"] for row in rows] != ids:
        raise ValueError("case order differs from the selected cohort")
    manifests = [_read(paths[case_id] / "manifest.json") for case_id in ids]
    keys = ("mode", "dataset_dir", "model", "model_config", "timeout", "max_calls",
            "tools", "output_contract_version", "log_schema_version")
    if any(any(manifest[key] != manifests[0][key] for key in keys)
           for manifest in manifests[1:]):
        raise ValueError("shard configurations differ")
    merged = root / "merged"
    merged.mkdir(exist_ok=True)
    manifest = dict(manifests[0])
    manifest.update(run_id=root.name, episode_ids=ids,
                    expected_turns={case_id: ["t1"] for case_id in ids},
                    termination="completed" if all(row["termination"] == "completed" for row in rows)
                    else "completed_with_errors")
    write_json(merged / "manifest.json", manifest)
    histories = {}
    for case_id in ids:
        one = _read(paths[case_id] / "messages.json")
        if set(one) != {case_id}:
            raise ValueError(f"wrong message episode in {paths[case_id]}")
        histories[case_id] = one[case_id]
    write_json(merged / "messages.json", histories)
    with (merged / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (merged / "diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for case_id in ids:
            stream.write((paths[case_id] / "diagnostics.jsonl").read_text(encoding="utf-8"))
    _eval_run(merged)
    return merged


def run_batch(dataset: Path, model_name: str, out: Path, *, config: Path = Path("ama.json"),
              tools: Path | None = None, timeout: float = 120, max_calls: int = 24,
              delay: float = 3) -> Path:
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    _load_dotenv()
    model = PacedModel.from_config(model_name, config, timeout=timeout)
    model.request_delay = delay
    model.retry_log = out / "transport_retries.jsonl"
    source = load_dataset(dataset)
    ids = [ep.id for ep in source.episodes]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("this experiment requires the fixed 100-case cohort")
    settings = {"dataset_dir": str(dataset), "episode_ids": ids, "model": model_name,
                "model_config": model.config, "timeout": timeout, "max_calls": max_calls,
                "tools": str(tools.resolve()) if tools else None, "delay": delay}
    out.mkdir(parents=True, exist_ok=True)
    record = out / "batch.json"
    if record.exists():
        batch = _read(record)
        if any(batch.get(key) != value for key, value in settings.items()):
            raise ValueError("resume settings differ from the saved batch")
    else:
        batch = {**settings, "status": "running", "started_at": time.time(),
                 "runs": {}, "attempts": {}}
        write_json(record, batch)
    batch["status"] = "running"
    write_json(record, batch)
    try:
        for index, case_id in enumerate(ids, 1):
            previous = batch["runs"].get(case_id)
            if previous and _decision(Path(previous))["termination"] == "completed":
                continue
            run = execute(dataset, model, episode_ids=[case_id], runs_root=out / "shards",
                          tools_path=tools, max_calls=max_calls)
            row = _decision(run)
            batch["runs"][case_id] = str(run)
            batch["attempts"].setdefault(case_id, []).append(str(run))
            write_json(record, batch)
            done = sum(_decision(Path(path))["termination"] == "completed"
                       for path in batch["runs"].values())
            bar = "#" * (30 * len(batch["runs"]) // len(ids))
            print(f"[{bar:<30}] {index}/{len(ids)} | completed={done} | "
                  f"{case_id}: {row['termination']}", flush=True)
    except BaseException:
        batch["status"] = "interrupted"
        write_json(record, batch)
        raise
    merged = merge_batch(out, batch)
    batch["status"] = "completed" if _read(merged / "manifest.json")["termination"] == "completed" \
        else "completed_with_errors"
    batch["merged_run"] = str(merged)
    batch["finished_at"] = time.time()
    write_json(record, batch)
    print(f"batch: {batch['status']} | merged: {merged}", flush=True)
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("ama.json"))
    parser.add_argument("--tools", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--delay", type=float, default=3)
    args = parser.parse_args(argv)
    run_batch(args.dataset, args.model, args.out, config=args.config,
              tools=args.tools, timeout=args.timeout, max_calls=args.max_calls,
              delay=args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
