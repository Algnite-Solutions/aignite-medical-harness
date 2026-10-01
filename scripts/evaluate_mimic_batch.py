"""Run a reproducible MIMIC sample, or check its progress without API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ama.cli import _eval_run, _load_dotenv
from ama.data import load_dataset, validate_dataset
from ama.decisions import output_metrics
from ama.importers.mimic_cdm import dataset_instructions, score_mimic_cdm
from ama.model import Model, ModelError
from ama.recorder import write_json
from ama.runner import execute


class PacedModel(Model):
    """Retry only HTTP 429 transport failures; never repair a model answer."""

    def complete(self, history, tools=None):
        for attempt in range(6):
            time.sleep(self.request_delay)
            try:
                return super().complete(history, tools)
            except ModelError as exc:
                if not str(exc).startswith("model endpoint returned HTTP 429"):
                    raise
                wait = min(30 * 2 ** attempt, 240)
                with self.retry_log.open("a") as stream:
                    stream.write(json.dumps({"time": time.time(), "http_status": 429,
                                             "attempt": attempt + 1,
                                             "wait_seconds": wait if attempt < 5 else 0}) + "\n")
                if attempt == 5:
                    raise
                time.sleep(wait)


def rows_at(root):
    rows = []
    for path in sorted(root.glob("shards/*/decisions.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrently appended final line may still be incomplete.
    return rows


def progress(root):
    batch = json.loads((root / "batch.json").read_text())
    rows = rows_at(root)
    n, total = len(rows), batch["count"]
    elapsed = time.time() - batch["started_at"]
    eta = elapsed * (total - n) / n if n else None
    errors = sum(r["termination"] != "completed" for r in rows)
    bar = "#" * int(30 * n / total) + "-" * (30 - int(30 * n / total))
    eta_text = f"{eta / 60:.1f}m" if eta is not None else "unknown"
    retries = root / "transport_retries.jsonl"
    retry_count = len(retries.read_text().splitlines()) if retries.exists() else 0
    return f"[{bar}] {n}/{total} | errors={errors} | 429s={retry_count} | elapsed={elapsed / 60:.1f}m | ETA={eta_text} | {batch['status']}"


def run(args):
    if args.after:
        print(f"Waiting for prior batch: {args.after.resolve()}", flush=True)
        while True:
            prior_path = args.after / "batch.json"
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                if prior["status"] == "completed":
                    break
                if prior["status"] != "running":
                    raise ValueError(f"prior batch ended with status {prior['status']}")
            time.sleep(15)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = load_dataset(args.source)
    if args.retry_errors_from:
        previous = json.loads((args.retry_errors_from / "batch.json").read_text())
        if previous["status"] != "completed":
            raise ValueError("retry source batch has not completed")
        bad = {row["episode_id"] for row in rows_at(args.retry_errors_from)
               if row["termination"] != "completed"}
        ids = [episode_id for episode_id in previous["episode_ids"] if episode_id in bad]
        if not ids:
            raise ValueError("retry source batch has no failed cases")
        args.count = len(ids)
    elif args.ids_from:
        previous = json.loads((args.ids_from / "batch.json").read_text())
        ids = previous["episode_ids"][:args.count]
        if len(ids) < args.count or len(set(ids)) != len(ids):
            raise ValueError("prior batch has insufficient unique case IDs")
        if not set(ids) <= {e.id for e in source.episodes}:
            raise ValueError("prior sample includes cases absent from this dataset variant")
    else:
        ids = random.Random(args.seed).sample(sorted(e.id for e in source.episodes), args.count)
    selected = set(ids)
    folder = root / "dataset"
    folder.mkdir()
    write_json(folder / "dataset.json", {"schema": "ama-dataset", "name": f"mimic_cdm_{args.variant}",
                                        "splits": {"all": ids}})
    for name in ("episodes.jsonl", "targets.jsonl"):
        with (folder / name).open("w") as output:
            for line in (args.source / name).read_text().splitlines():
                if line.strip() and json.loads(line)["id"] in selected:
                    output.write(line + "\n")
    for ep in source.episodes:
        if ep.id in selected and any(e.file for t in ep.turns for e in t.evidence):
            raise ValueError("This batch expects inline evidence only")
    if args.variant == "interactive":
        with (folder / "cases.jsonl").open("w") as output:
            for line in (args.source / "cases.jsonl").read_text().splitlines():
                if line.strip() and json.loads(line)["hadm_id"] in selected:
                    output.write(line + "\n")
        (folder / "lab_mapping.json").write_text((args.source / "lab_mapping.json").read_text())
        write_json(folder / "tool_data_files.json", ["cases.jsonl", "lab_mapping.json"])
    (folder / "eval.json").write_text((args.source / "eval.json").read_text())
    (folder / "instructions.txt").write_text(dataset_instructions(interactive=args.variant == "interactive"))
    errors = validate_dataset(folder)
    if errors:
        raise ValueError(errors)
    _load_dotenv()
    models = [PacedModel.from_config(args.model, args.config, timeout=args.timeout) for _ in range(args.workers)]
    for model in models:
        model.request_delay = args.delay
        model.retry_log = root / "transport_retries.jsonl"
    write_json(root / "batch.json", {"source": str(args.source.resolve()), "count": args.count,
        "seed": args.seed,
        "sampling": (f"retry failed IDs from {args.retry_errors_from}" if args.retry_errors_from else
                     f"first {args.count} IDs from {args.ids_from}" if args.ids_from else
                     "uniform without replacement from sorted episode IDs"),
        "variant": args.variant, "ids_from": str(args.ids_from.resolve()) if args.ids_from else None,
        "retry_errors_from": (str(args.retry_errors_from.resolve())
                              if args.retry_errors_from else None),
        "episode_ids": ids, "model": args.model, "model_config": models[0].config,
        "workers": args.workers, "timeout": args.timeout, "request_delay": args.delay,
        "transport_retry_policy": "HTTP 429 only; max 6 attempts; exponential backoff 30–240 seconds; separately logged",
        "started_at": time.time(),
        "output_contract_version": 2, "status": "running"})
    print(f"batch: {root}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, folder, model, episode_ids=ids[i::args.workers],
                               runs_root=root / "shards", max_calls=(args.max_calls or
                                   (24 if args.variant == "interactive" else 1)),
                               tools_path=(Path("src/ama/importers/mimic_cdm_tools.py")
                                           if args.variant == "interactive" else None))
                   for i, model in enumerate(models)]
        while not all(f.done() for f in futures):
            print(progress(root), flush=True)
            time.sleep(15)
        paths = [f.result() for f in futures]
    for path in paths:
        _eval_run(path)
    dataset = load_dataset(folder, with_targets=True)
    rows = rows_at(root)
    if len(rows) != args.count or len({(r["episode_id"], r["turn_id"]) for r in rows}) != args.count:
        raise ValueError("Expected exactly one unique decision row per selected case")
    grouped = defaultdict(list)
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(Counter)
    for row in rows:
        grouped[row["episode_id"]].append(row)
        target = dataset.targets[row["episode_id"]].turns[row["turn_id"]]["answer"]["diagnosis"]
        answer = (row.get("decision") or {}).get("answer") or {}
        predicted = answer.get("diagnosis", "<missing>") if isinstance(answer, dict) else "<invalid>"
        if isinstance(predicted, str):
            predicted = predicted.strip().casefold()
        breakdown[target]["total"] += 1
        breakdown[target]["correct"] += int(predicted == target)
        confusion[target][str(predicted)] += 1
    scored = score_mimic_cdm(dataset, grouped)
    scored["aggregate"]["output_quality"] = output_metrics(rows, 2)
    result = {"scored": scored, "per_diagnosis": dict(breakdown), "confusion_matrix": dict(confusion),
              "terminations": dict(Counter(r["termination"] for r in rows)),
              "model_calls": sum(r["model_calls"] for r in rows),
              "total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0) for r in rows),
              "median_duration_ms": statistics.median(r["duration_ms"] for r in rows)}
    write_json(root / "metrics.json", result)
    batch = json.loads((root / "batch.json").read_text())
    batch.update(status="completed", finished_at=time.time(), shard_runs=[str(p) for p in paths])
    write_json(root / "batch.json", batch)
    print(progress(root), flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "scored"}, indent=2))
    print(json.dumps(scored["aggregate"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--variant", choices=["full_info", "interactive"], default="full_info")
    parser.add_argument("--ids-from", type=Path, help="reuse IDs from a prior batch in order")
    parser.add_argument("--retry-errors-from", type=Path,
                        help="rerun only failed cases from a completed batch")
    parser.add_argument("--after", type=Path, help="wait for a prior batch to finish before starting")
    parser.add_argument("--out", type=Path, default=Path("runs") / ("deepseek-full-info-500-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--config", type=Path, default=Path("ama.json"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    if args.check:
        while True:
            if not (args.check / "batch.json").exists():
                print(f"Waiting for batch to start: {args.check.resolve()} "
                      "(the checker does not start an evaluation)", flush=True)
                if not args.watch:
                    break
                time.sleep(15)
                continue
            print(progress(args.check), flush=True)
            if not args.watch or json.loads((args.check / "batch.json").read_text())["status"] != "running":
                break
            time.sleep(15)
    else:
        if not args.source or args.count < 1 or args.workers < 1 or args.workers > args.count:
            parser.error("provide --source, positive --count and --workers <= count")
        try:
            run(args)
        except BaseException as exc:
            path = args.out / "batch.json"
            if path.exists():
                batch = json.loads(path.read_text())
                batch.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_json(path, batch)
            raise
