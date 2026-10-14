"""Prepare and run a reproducible 100-case Symptom2Disease AMA benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from ama.cli import _load_dotenv
from ama.data import load_dataset, validate_dataset
from ama.importers.symptom2disease import INSTRUCTIONS, LABELS, SCORER, import_symptom2disease
from ama.model import Model, ModelError
from ama.recorder import write_json
from ama.runner import DECISION_PROMPT, execute
from scripts.mimic_cdm_batch import merge_batch

PROTOCOL = "symptom2disease_closed24_seed42_v1"
SEED = 42
COHORT_SIZE = 100
DATA_FILES = ("dataset.json", "episodes.jsonl", "instructions.txt", "eval.json", "targets.jsonl")
VISIBLE_FILES = DATA_FILES[:3]
FROZEN_TERMINATIONS = {"completed", "invalid_decision", "call_limit"}


def _frozen_attempt(attempt: dict[str, Any]) -> bool:
    """An API error after a returned assistant message is an invalid model output."""
    return (attempt["termination"] in FROZEN_TERMINATIONS or
            (attempt["termination"] == "api_error" and attempt["error_kind"] is None))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_rows(source: Path) -> tuple[list[int], list[str], int]:
    """Seed-42 balanced sample, excluding repeated symptom texts before sampling."""
    groups: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, str] = {}
    duplicates = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"label", "text"} <= set(reader.fieldnames):
            raise ValueError("Symptom2Disease source must contain label and text columns")
        for number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"malformed CSV row {number}")
            label, symptom = (row.get("label") or "").strip(), (row.get("text") or "").strip()
            if not label or not symptom:
                continue
            if label not in LABELS:
                raise ValueError(f"unknown disease label at CSV row {number}: {label!r}")
            key = " ".join(symptom.split()).casefold()
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"conflicting labels for repeated text at CSV row {number}")
                duplicates += 1
                continue
            seen[key] = label
            groups[label].append(number)
    if set(groups) != set(LABELS) or any(len(groups[label]) < 5 for label in LABELS):
        raise ValueError("source needs all 24 labels with at least five unique texts each")
    rng = random.Random(SEED)
    extra_labels = sorted(rng.sample(sorted(LABELS), 4))
    rows = sorted(number for label in sorted(LABELS)
                  for number in rng.sample(sorted(groups[label]), 4 + (label in extra_labels)))
    if len(rows) != COHORT_SIZE:
        raise AssertionError("sampling did not produce 100 cases")
    return rows, extra_labels, duplicates


def verify_benchmark(root: Path, *, evaluator: bool = False) -> dict[str, Any]:
    """The run phase verifies visible files; only evaluation checks hidden targets."""
    benchmark = _read(root / "benchmark.json")
    if benchmark.get("protocol") != PROTOCOL or len(benchmark.get("episode_ids", [])) != COHORT_SIZE:
        raise ValueError("wrong or incomplete Symptom2Disease benchmark protocol")
    dataset_dir = root / "dataset"
    for name in DATA_FILES if evaluator else VISIBLE_FILES:
        if _sha256(dataset_dir / name) != benchmark["dataset_sha256"][name]:
            raise ValueError(f"benchmark dataset changed: {name}")
    dataset = load_dataset(dataset_dir)
    if [episode.id for episode in dataset.episodes] != benchmark["episode_ids"] or \
            dataset.info.splits.get("all") != benchmark["episode_ids"]:
        raise ValueError("benchmark Episode IDs or order changed")
    if evaluator:
        problems = validate_dataset(dataset_dir)
        if problems:
            raise ValueError("invalid benchmark dataset: " + "; ".join(problems))
        scored = load_dataset(dataset_dir, with_targets=True)
        if scored.eval_config is None or scored.eval_config.scorer != SCORER or \
                set(scored.targets) != set(benchmark["episode_ids"]):
            raise ValueError("benchmark scorer or target coverage changed")
    return benchmark


def prepare_benchmark(source: Path, root: Path) -> dict[str, Any]:
    """Create the selected AMA dataset and frozen cohort manifest atomically."""
    source, root = Path(source).resolve(), Path(root).resolve()
    if root.exists():
        benchmark = verify_benchmark(root)
        if not source.is_file() or _sha256(source) != benchmark["source_sha256"]:
            raise ValueError("source changed since the fixed benchmark was prepared")
        return benchmark
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError(f"expected a local Symptom2Disease CSV: {source}")
    rows, extra_labels, duplicates = select_rows(source)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ama-s2d-benchmark-", dir=root.parent) as temp:
        stage = Path(temp) / "benchmark"
        stage.mkdir()
        report = import_symptom2disease(source, stage / "dataset", rows=rows)
        benchmark = {
            "protocol": PROTOCOL, "seed": SEED, "source_file": source.name,
            "source_sha256": report["source_sha256"],
            "sampling": "four unique texts per label, plus one from four seeded labels",
            "duplicate_policy": "keep first CSV row per whitespace-normalized, casefolded text",
            "duplicates_excluded_before_sampling": duplicates,
            "extra_labels": extra_labels, "selected_csv_rows": rows,
            "episode_ids": [item["id"] for item in report["selected_source_rows"]],
            "label_counts": report["label_counts_in_selected"],
            "scorer": SCORER,
            "dataset_sha256": {name: _sha256(stage / "dataset" / name) for name in DATA_FILES},
        }
        write_json(stage / "benchmark.json", benchmark)
        verify_benchmark(stage, evaluator=True)
        if root.exists():
            raise FileExistsError(f"benchmark output already exists: {root}")
        stage.rename(root)
    return benchmark


def _retryable(exc: ModelError) -> bool:
    return exc.http_status in {408, 429} or \
        exc.http_status is not None and 500 <= exc.http_status <= 599 or \
        str(exc).startswith(("model connection failed:",
                             "model endpoint returned invalid JSON",
                             "model response has no assistant message"))


class RetryingModel(Model):
    """Retry only transient transport errors; a received answer is never revised."""

    request_delay: float = 0
    max_attempts: int = 3
    retry_log: Path
    last_error_kind: str | None = None

    def complete(self, history: list[dict[str, Any]], tools=None) -> dict:
        self.last_error_kind = None
        for attempt in range(self.max_attempts):
            if self.request_delay:
                time.sleep(self.request_delay)
            try:
                reply = super().complete(history, tools)
                self.last_error_kind = None
                return reply
            except ModelError as exc:
                transient = _retryable(exc)
                self.last_error_kind = "transport" if transient else "permanent_api"
                if not transient:
                    raise
                retry_after = (exc.rate_limit_headers or {}).get("retry-after")
                try:
                    requested_wait = max(0, min(float(retry_after), 60)) if retry_after else None
                except ValueError:
                    requested_wait = None
                wait = requested_wait if requested_wait is not None else min(2 ** attempt, 30)
                with self.retry_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"attempt": attempt + 1, "http_status": exc.http_status,
                                             "wait_seconds": wait if attempt + 1 < self.max_attempts else 0,
                                             "connection_error": str(exc).startswith("model connection failed:")}) + "\n")
                if attempt + 1 == self.max_attempts:
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")


def _one_decision(run: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one recorded Turn: {run}")
    return rows[0]


def _recover_shards(out: Path, batch: dict[str, Any]) -> None:
    """Attach persisted one-case runs left outside batch.json by a crash."""
    shards = out / "shards"
    if not shards.is_dir():
        return
    known = {attempt["run"] for attempts in batch["attempts"].values()
             for attempt in attempts}
    changed = False
    for run in sorted((path for path in shards.iterdir() if path.is_dir()),
                      key=lambda path: (path.stat().st_mtime_ns, path.name)):
        if str(run) in known:
            continue
        manifest_path, decisions_path = run / "manifest.json", run / "decisions.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"orphan shard has no manifest: {run}")
        manifest = _read(manifest_path)
        ids = manifest.get("episode_ids")
        if not isinstance(ids, list) or len(ids) != 1 or ids[0] not in batch["episode_ids"] or \
                manifest.get("dataset_dir") != batch["dataset_dir"] or \
                manifest.get("model") != batch["model"] or \
                manifest.get("model_config") != batch["model_config"] or \
                manifest.get("timeout") != batch["timeout"] or \
                manifest.get("max_calls") != batch["max_calls"] or \
                manifest.get("expected_turns") != {ids[0]: ["t1"]} or \
                manifest.get("output_contract_version") != 2 or \
                manifest.get("mode") != "run" or manifest.get("tools") is not None:
            raise ValueError(f"orphan shard does not match batch settings: {run}")
        episode_id = ids[0]
        if not decisions_path.is_file() or not decisions_path.stat().st_size:
            raise ValueError(f"orphan shard has no Decision; inspect before retrying: {run}")
        row = _one_decision(run)
        if row["episode_id"] != episode_id or row["turn_id"] != "t1":
            raise ValueError(f"orphan shard Decision does not match Episode: {run}")
        history = _read(run / "messages.json").get(episode_id)
        if not isinstance(history, list) or len(history) < 2:
            raise ValueError(f"orphan shard has incomplete message history: {run}")
        returned_message = len(history) > 2
        if row["termination"] in FROZEN_TERMINATIONS or returned_message:
            kind = None
        elif row["termination"] == "api_error":
            diagnostics = [json.loads(line) for line in (run / "diagnostics.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            calls = [entry for entry in diagnostics if entry.get("kind") == "model_call"]
            http_status = calls[-1].get("http_status") if calls else None
            kind = "transport" if _retryable(ModelError(row.get("error") or "",
                                                        http_status=http_status)) else "permanent_api"
        else:
            kind = "runner_error"
        previous = batch["attempts"].get(episode_id, [])
        if any(_frozen_attempt(attempt) for attempt in previous):
            raise ValueError(f"orphan shard follows an already frozen answer: {run}")
        attempt = {"run": str(run), "termination": row["termination"], "error_kind": kind}
        batch["attempts"].setdefault(episode_id, []).append(attempt)
        batch["runs"][episode_id] = str(run)
        changed = True
    if changed:
        write_json(out / "batch.json", batch)


def run_model(root: Path, model_name: str, *, config: Path = Path("ama.json"),
              timeout: float = 300, delay: float = 1, max_attempts: int = 3,
              runs_root: Path | None = None) -> Path:
    """Resume transport failures; freeze completed and invalid model responses."""
    root, config = Path(root).resolve(), Path(config).resolve()
    benchmark = verify_benchmark(root)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_name) or model_name in {".", ".."}:
        raise ValueError("model alias is not a safe directory name")
    if timeout <= 0 or delay < 0 or max_attempts < 1:
        raise ValueError("timeout, delay or max_attempts out of range")
    _load_dotenv()
    model = RetryingModel.from_config(model_name, config, timeout=timeout)
    out = (Path(runs_root).resolve() if runs_root is not None else root / "runs") / model_name
    settings = {
        "dataset_dir": str(root / "dataset"),
        "protocol": PROTOCOL, "episode_ids": benchmark["episode_ids"],
        "visible_sha256": {name: benchmark["dataset_sha256"][name] for name in VISIBLE_FILES},
        "model": model_name, "model_config": model.config, "timeout": timeout,
        "max_calls": 1, "delay": delay, "max_attempts": max_attempts,
        "runner_sha256": _sha256(Path(__file__)),
    }
    out.mkdir(parents=True, exist_ok=True)
    record = out / "batch.json"
    if record.exists():
        batch = _read(record)
        if any(batch.get(key) != value for key, value in settings.items()):
            raise ValueError(f"saved batch settings differ: {record}")
        if batch["status"] in {"completed", "completed_with_invalid_answers"}:
            return out / "merged"
        if batch["status"] in {"permanent_api_error", "runner_error"}:
            raise ValueError("permanent API or runner error; use a new run directory after correction")
    else:
        batch = {**settings, "status": "running", "runs": {}, "attempts": {}}
        write_json(record, batch)
    _recover_shards(out, batch)
    model.request_delay, model.max_attempts = delay, max_attempts
    model.retry_log = out / "transport_retries.jsonl"
    batch["status"] = "running"
    write_json(record, batch)
    try:
        for index, episode_id in enumerate(benchmark["episode_ids"], start=1):
            previous = batch["runs"].get(episode_id)
            if previous:
                last = batch["attempts"][episode_id][-1]
                if _frozen_attempt(last):
                    continue
                if last["error_kind"] in {"permanent_api", "runner_error"}:
                    raise ValueError(f"{model_name}/{episode_id}: prior permanent failure; "
                                     "use a new run directory after correction")
            run = execute(root / "dataset", model, episode_ids=[episode_id],
                          runs_root=out / "shards", max_calls=1)
            row = _one_decision(run)
            batch["runs"][episode_id] = str(run)
            attempt = {
                "run": str(run), "termination": row["termination"],
                "error_kind": model.last_error_kind,
            }
            batch["attempts"].setdefault(episode_id, []).append(attempt)
            write_json(record, batch)
            print(f"[{index}/{COHORT_SIZE}] {model_name} {episode_id}: {row['termination']}", flush=True)
            if _frozen_attempt(attempt):
                continue
            if row["termination"] == "api_error":
                batch["status"] = ("waiting_for_transport" if model.last_error_kind == "transport"
                                   else "permanent_api_error")
            elif row["termination"] in {"interrupted", "not_executed"}:
                batch["status"] = "interrupted"
            else:
                batch["status"] = "runner_error"
            write_json(record, batch)
            raise RuntimeError(f"{model_name}/{episode_id}: {row['termination']}; "
                               f"batch status {batch['status']}")
        verify_benchmark(root, evaluator=True)
        merged = merge_batch(out, batch)
        attempts = [batch["attempts"][episode_id][-1] for episode_id in benchmark["episode_ids"]]
        batch["status"] = ("completed_with_invalid_answers" if any(
            attempt["termination"] != "completed" for attempt in attempts)
                           else "completed")
        batch["merged_run"] = str(merged)
        write_json(record, batch)
        return merged
    except BaseException:
        if batch["status"] == "running":
            batch["status"] = "interrupted"
            write_json(record, batch)
        raise


def _paired(left: dict[str, Any], right: dict[str, Any], ids: list[str]) -> dict[str, int]:
    counts = Counter()
    for episode_id in ids:
        a, b = left[episode_id]["correct"], right[episode_id]["correct"]
        counts["both_correct" if a and b else "left_only" if a else "right_only" if b
               else "neither"] += 1
    return {name: counts[name] for name in ("both_correct", "left_only", "right_only", "neither")}


def compare_models(root: Path, names: list[str], *, runs_root: Path | None = None) -> dict[str, Any]:
    """Verify matched runs before reporting scores and paired case outcomes."""
    root = Path(root).resolve()
    benchmark = verify_benchmark(root, evaluator=True)
    ids = benchmark["episode_ids"]
    if not names or len(names) != len(set(names)):
        raise ValueError("comparison needs distinct model aliases")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in {".", ".."} for name in names):
        raise ValueError("model alias is not a safe directory name")
    results: dict[str, Any] = {}
    request_settings = None
    for name in names:
        out = (Path(runs_root).resolve() if runs_root is not None else root / "runs") / name
        batch, manifest = _read(out / "batch.json"), _read(out / "merged" / "manifest.json")
        metrics = _read(out / "merged" / "metrics.json")
        if batch["status"] not in {"completed", "completed_with_invalid_answers"} or \
                batch["episode_ids"] != ids or manifest["episode_ids"] != ids or \
                batch["visible_sha256"] != {name: benchmark["dataset_sha256"][name]
                                            for name in VISIBLE_FILES} or \
                manifest["dataset_dir"] != str(root / "dataset") or \
                manifest["mode"] != "run" or manifest["tools"] is not None or \
                manifest["model"] != name or manifest["model_config"] != batch["model_config"] or \
                manifest["expected_turns"] != {episode_id: ["t1"] for episode_id in ids} or \
                metrics["run_id"] != manifest["run_id"] or metrics["model"] != name or \
                metrics["scorer"] != SCORER:
            raise ValueError(f"run, cohort or scorer mismatch: {name}")
        settings = (manifest["timeout"], manifest["max_calls"],
                    manifest["model_config"].get("temperature", 0),
                    manifest["model_config"].get("max_tokens"))
        if request_settings is None:
            request_settings = settings
        elif settings != request_settings:
            raise ValueError(f"request settings differ across models: {name}")
        histories = _read(out / "merged" / "messages.json")
        if set(histories) != set(ids) or any(
                not histories[episode_id] or histories[episode_id][0] != {
                    "role": "system", "content": INSTRUCTIONS + "\n\n" + DECISION_PROMPT}
                for episode_id in ids):
            raise ValueError(f"actual system prompt or message coverage differs: {name}")
        decisions = [json.loads(line) for line in (out / "merged" / "decisions.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        per = metrics["scored"]["per_episode"]
        if len(decisions) != COHORT_SIZE or [row["episode_id"] for row in decisions] != ids or \
                set(per) != set(ids) or \
                metrics["scored"]["aggregate"]["disease_accuracy"]["den"] != COHORT_SIZE:
            raise ValueError(f"decision or evaluation coverage differs: {name}")
        if any(row["termination"] != batch["attempts"][episode_id][-1]["termination"] or
               per[episode_id]["termination"] != row["termination"]
               for episode_id, row in zip(ids, decisions)):
            raise ValueError(f"attempt, decision or score disagrees: {name}")
        terminations = Counter(row["termination"] for row in decisions)
        if any(not _frozen_attempt(batch["attempts"][episode_id][-1]) or
               batch["attempts"][episode_id][-1]["run"] != batch["runs"][episode_id]
               for episode_id in ids):
            raise ValueError(f"unresolved transport or run failures: {name}")
        results[name] = {
            "model": manifest["model_config"]["model"],
            "model_config": manifest["model_config"],
            "accuracy": metrics["scored"]["aggregate"]["disease_accuracy"],
            "valid_choice": metrics["scored"]["aggregate"]["valid_choice"],
            "completion": metrics["scored"]["aggregate"]["completion"],
            "answer_object_format": metrics["scored"]["aggregate"]["answer_object_format"],
            "output_quality": metrics["scored"]["aggregate"]["output_quality"],
            "terminations": dict(sorted(terminations.items())),
            "reported_tokens": sum((row.get("usage") or {}).get("total_tokens", 0)
                                   for row in decisions),
            "per_episode": per,
        }
    paired = {f"{left}_vs_{right}": _paired(results[left]["per_episode"],
                                             results[right]["per_episode"], ids)
              for left, right in combinations(names, 2)}
    return {"protocol": PROTOCOL, "source_sha256": benchmark["source_sha256"],
            "episode_ids": ids, "request_settings": {"timeout": request_settings[0],
            "max_calls": request_settings[1], "temperature": request_settings[2],
            "max_tokens": request_settings[3]}, "results": results, "paired": paired}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use scripts/run_symptom2disease.sh --help for the raw/processed workflow. "
        "With --benchmark, --out contains run records only; running a prepared cohort "
        "does not require the raw CSV. Without --benchmark, the legacy combined layout is used."))
    parser.add_argument("--source", type=Path, help="raw label,text CSV; required to prepare a cohort")
    parser.add_argument("--benchmark", type=Path, help="prepared cohort root containing benchmark.json and dataset/")
    parser.add_argument("--out", type=Path, help="run output directory, or combined cohort/run root without --benchmark")
    parser.add_argument("--model", action="append", dest="models", help="ama.json model alias; repeat for multiple models")
    parser.add_argument("--config", type=Path, default=Path("ama.json"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="prepare and validate without API calls")
    mode.add_argument("--compare-only", action="store_true", help="compare completed runs without API calls")
    mode.add_argument("--run-only", action="store_true", help="run and score models without writing a comparison")
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.models:
        parser.error("provide at least one --model or use --prepare-only")
    if args.benchmark is None and args.out is None:
        parser.error("provide --benchmark or --out")
    if not args.prepare_only and args.out is None:
        parser.error("provide --out for run records")
    if args.prepare_only and args.source is None:
        parser.error("--prepare-only requires --source")
    if args.benchmark is None and args.source is None:
        parser.error("provide --source or an existing --benchmark")
    root = (args.benchmark or args.out).resolve()
    if args.source is not None and not args.compare_only:
        prepare_benchmark(args.source, root)
    else:
        verify_benchmark(root)
    if args.prepare_only:
        print(f"prepared: {root / 'dataset'}")
        return 0
    runs_root = args.out.resolve() if args.benchmark is not None else None
    if not args.compare_only:
        for name in args.models:
            run_model(root, name, config=args.config, timeout=args.timeout,
                      delay=args.delay, max_attempts=args.max_attempts, runs_root=runs_root)
    if args.run_only:
        return 0
    report = compare_models(root, args.models, runs_root=runs_root)
    destination = (runs_root or root) / "comparison.json"
    if destination.exists() and _read(destination) != report:
        raise ValueError("existing comparison differs; use a new output directory")
    write_json(destination, report)
    print(f"comparison: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
