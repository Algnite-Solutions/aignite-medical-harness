"""Fixed test-cohort VQA-RAD comparison using native AMA runs and evaluation.

Reuses the existing transport retry/shard recovery helpers and batch merger.
Answers (including invalid ones) are frozen; only transport failures may retry.
Preparation/evaluation may read targets; model execution verifies visible inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ama.cli import _eval_run, _load_dotenv
from ama.data import load_dataset, validate_dataset
from ama.importers.vqa_rad import REVISION, _sha
from ama.importers.vqa_rad_scoring import SCORER
from ama.model import content
from ama.recorder import write_json
from ama.runner import DECISION_PROMPT, execute, render
from scripts.mimic_cdm_batch import merge_batch
from scripts.symptom2disease_benchmark import (
    RetryingModel, _frozen_attempt, _one_decision, _recover_shards,
)

PROTOCOL = "vqa_rad_test_seeded_pr12_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "scripts/vqa_rad_benchmark.py", "scripts/symptom2disease_benchmark.py",
    "scripts/mimic_cdm_batch.py", "src/ama/runner.py", "src/ama/agent.py",
    "src/ama/model.py", "src/ama/data.py", "src/ama/recorder.py", "src/ama/cli.py",
    "src/ama/decisions.py", "src/ama/scorer.py", "src/ama/importers/vqa_rad_scoring.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare(dataset_dir: Path, root: Path, *, count: int = 100, seed: int = 42) -> dict:
    """Freeze selected test IDs and all input/target hashes without resampling."""
    dataset_dir, root = dataset_dir.resolve(), root.resolve()
    if (root / "cohort.json").exists():
        cohort = verify(root, evaluator=True)
        require((cohort["dataset_dir"], cohort["count"], cohort["seed"]) ==
                (str(dataset_dir), count, seed), "existing cohort settings differ")
        return cohort
    require(not root.exists(), "cohort directory already exists without a manifest")
    require(not validate_dataset(dataset_dir), "invalid AMA source dataset")
    dataset = load_dataset(dataset_dir, with_targets=True)
    require(dataset.eval_config is not None and dataset.eval_config.scorer == SCORER,
            "expected VQA-RAD scorer")
    report = read(dataset_dir / "import_report.json")
    require(report["source"]["revision"] == REVISION, "wrong VQA-RAD revision")
    candidates = dataset.info.splits.get("test", [])
    require(len(set(candidates)) == len(candidates) and 0 < count <= len(candidates),
            "invalid source test IDs or cohort count")
    positions = sorted(random.Random(seed).sample(range(len(candidates)), count))
    ids = [candidates[index] for index in positions]
    by_id = {episode.id: episode for episode in dataset.episodes}
    images = set()
    for episode_id in ids:
        require(episode_id in dataset.targets and len(by_id[episode_id].turns) == 1,
                "selected case missing target or not single-turn")
        for evidence in by_id[episode_id].turns[0].evidence:
            if evidence.file:
                images.add(evidence.file)
    visible = ["dataset.json", "episodes.jsonl", "instructions.txt", *sorted(images)]
    cohort = {
        "protocol": PROTOCOL, "dataset_dir": str(dataset_dir), "revision": REVISION,
        "count": count, "seed": seed, "source_test_count": len(candidates),
        "selection": "sorted(random.Random(seed).sample(range(source_test_count), count)); source test order",
        "selected_test_indices": positions, "episode_ids": ids,
        "visible_sha256": {name: _sha(dataset_dir / name) for name in visible},
        "evaluator_sha256": {name: _sha(dataset_dir / name) for name in
                             ("targets.jsonl", "eval.json", "import_report.json")},
        "scorer": SCORER,
        "question_type_counts": dict(Counter(dataset.targets[episode_id].turns["t1"]["question_type"]
                                             for episode_id in ids)),
    }
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "cohort.json", cohort)
    verify(root, evaluator=True)
    return cohort


def verify(root: Path, *, evaluator: bool = False) -> dict:
    cohort = read(root / "cohort.json")
    require(cohort["protocol"] == PROTOCOL and cohort["revision"] == REVISION
            and cohort["scorer"] == SCORER, "wrong cohort protocol")
    dataset_dir = Path(cohort["dataset_dir"])
    files = {**cohort["visible_sha256"], **(cohort["evaluator_sha256"] if evaluator else {})}
    for name, digest in files.items():
        path = dataset_dir / name
        require(dataset_dir.resolve() in path.resolve().parents and _sha(path) == digest,
                f"dataset file changed: {name}")
    dataset = load_dataset(dataset_dir, with_targets=evaluator)
    candidates = dataset.info.splits["test"]
    require(len(candidates) == cohort["source_test_count"], "source test count changed")
    indices = sorted(random.Random(cohort["seed"]).sample(range(len(candidates)), cohort["count"]))
    require(indices == cohort["selected_test_indices"] and
            [candidates[index] for index in indices] == cohort["episode_ids"], "cohort selection changed")
    if evaluator:
        require(not validate_dataset(dataset_dir) and dataset.eval_config.scorer == SCORER
                and all(episode_id in dataset.targets for episode_id in cohort["episode_ids"]),
                "invalid evaluation dataset")
    return cohort


def _settings(root: Path, cohort: dict, model, delay: float, max_attempts: int) -> dict:
    return {
        "protocol": PROTOCOL, "cohort_sha256": _sha(root / "cohort.json"),
        "dataset_dir": cohort["dataset_dir"], "episode_ids": cohort["episode_ids"],
        "visible_sha256": cohort["visible_sha256"], "model": model.name,
        "model_config": model.config, "timeout": model.timeout, "max_calls": 1,
        "delay": delay, "max_attempts": max_attempts,
        "code_sha256": {name: _sha(REPO_ROOT / name) for name in CODE_FILES},
    }


def run_model(root: Path, model_name: str, *, config: Path, runs_root: Path,
              timeout: float = 180, delay: float = 1, max_attempts: int = 3,
              max_new_cases: int | None = None) -> Path | None:
    """Resume persisted shards; returned answers are never regenerated."""
    root, runs_root = root.resolve(), runs_root.resolve()
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name)), "unsafe model alias")
    require(timeout > 0 and delay >= 0 and max_attempts > 0
            and (max_new_cases is None or max_new_cases > 0), "invalid request controls")
    cohort = verify(root)
    model = RetryingModel.from_config(model_name, config, timeout=timeout)
    settings = _settings(root, cohort, model, delay, max_attempts)
    out = runs_root / model_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another worker holds the batch lock: {model_name}") from None
        record = out / "batch.json"
        if record.exists():
            batch = read(record)
            require(all(batch.get(key) == value for key, value in settings.items()),
                    "saved cohort, config or code differs; inspect before starting a new batch")
            if batch["status"] in {"permanent_api_error", "runner_error"}:
                raise RuntimeError(f"{model_name}: saved permanent failure requires inspection")
        else:
            batch = {**settings, "status": "prepared", "runs": {}, "attempts": {}}
            write_json(record, batch)
        _recover_shards(out, batch)
        model.request_delay, model.max_attempts = delay, max_attempts
        model.retry_log = out / "transport_retries.jsonl"
        batch["status"] = "running"
        write_json(record, batch)
        new_cases = 0
        try:
            for index, episode_id in enumerate(cohort["episode_ids"], 1):
                attempts = batch["attempts"].get(episode_id, [])
                if attempts and _frozen_attempt(attempts[-1]):
                    continue
                if max_new_cases is not None and new_cases >= max_new_cases:
                    batch["status"] = "partial"
                    write_json(record, batch)
                    return None
                run = execute(Path(cohort["dataset_dir"]), model, episode_ids=[episode_id],
                              runs_root=out / "shards", max_calls=1)
                row = _one_decision(run)
                attempt = {"run": str(run), "termination": row["termination"],
                           "error_kind": model.last_error_kind}
                batch["runs"][episode_id] = str(run)
                batch["attempts"].setdefault(episode_id, []).append(attempt)
                write_json(record, batch)
                new_cases += 1
                print(f"[{index}/{cohort['count']}] {model_name}: {row['termination']}", flush=True)
                if _frozen_attempt(attempt):
                    continue
                batch["status"] = ("waiting_for_transport" if model.last_error_kind == "transport"
                                   else "permanent_api_error" if model.last_error_kind == "permanent_api"
                                   else "interrupted" if row["termination"] in {"interrupted", "not_executed"}
                                   else "runner_error")
                write_json(record, batch)
                raise RuntimeError(f"{model_name}: {batch['status']}")
            verify(root, evaluator=True)
            merged = merge_batch(out, batch)
            batch["status"] = "completed" if all(
                batch["attempts"][episode_id][-1]["termination"] == "completed"
                for episode_id in cohort["episode_ids"]) else "completed_with_invalid_answers"
            batch["merged_run"] = str(merged)
            write_json(record, batch)
            return merged
        except BaseException:
            if batch["status"] == "running":
                batch["status"] = "interrupted"
                write_json(record, batch)
            raise


def compare(root: Path, names: list[str], runs_root: Path) -> dict:
    """Re-evaluate complete native runs and verify identical actual input messages."""
    cohort = verify(root, evaluator=True)
    dataset = load_dataset(Path(cohort["dataset_dir"]))
    by_id = {episode.id: episode for episode in dataset.episodes}
    system = (dataset.folder / "instructions.txt").read_text() + "\n\n" + DECISION_PROMPT
    require(bool(names) and len(names) == len(set(names)), "distinct model aliases required")
    results, paired, shared = {}, {}, None
    by_model = {}
    for name in names:
        require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)), "unsafe model alias")
        out = runs_root / name
        batch = read(out / "batch.json")
        require(batch["status"] in {"completed", "completed_with_invalid_answers"}
                and batch["episode_ids"] == cohort["episode_ids"]
                and batch["cohort_sha256"] == _sha(root / "cohort.json")
                and batch["visible_sha256"] == cohort["visible_sha256"], "incomplete or mismatched batch")
        require(batch["code_sha256"] == {path: _sha(REPO_ROOT / path) for path in CODE_FILES},
                "evaluation code differs from the recorded batch")
        require(all(_frozen_attempt(batch["attempts"][episode_id][-1])
                    and batch["attempts"][episode_id][-1]["run"] == batch["runs"][episode_id]
                    for episode_id in cohort["episode_ids"]), "unresolved attempt")
        merged = out / "merged"
        manifest, histories = read(merged / "manifest.json"), read(merged / "messages.json")
        require(manifest["episode_ids"] == cohort["episode_ids"]
                and manifest["model"] == name and manifest["model_config"] == batch["model_config"]
                and manifest["dataset_dir"] == cohort["dataset_dir"]
                and manifest["timeout"] == batch["timeout"] and manifest["max_calls"] == 1
                and manifest["tools"] is None and manifest["mode"] == "run"
                and manifest["output_contract_version"] == 2
                and manifest["expected_turns"] == {episode_id: ["t1"] for episode_id in cohort["episode_ids"]},
                "native manifest differs from batch")
        settings = (batch["timeout"], batch["max_calls"], batch["model_config"].get("temperature", 0),
                    batch["delay"], batch["max_attempts"])
        require(shared is None or settings == shared, "shared request controls differ")
        shared = settings
        require(set(histories) == set(cohort["episode_ids"]), "history coverage differs")
        for episode_id in cohort["episode_ids"]:
            turn = by_id[episode_id].turns[0]
            evidence_ids = sorted(ev.id for ev in turn.evidence)
            message = render(turn, dataset.folder)
            message.append(f"Final Decision turn_id: {turn.id}. Currently citeable evidence IDs: "
                           + json.dumps(evidence_ids))
            expected = [{"role": "system", "content": system},
                        {"role": "user", "content": content(message)}]
            require(histories[episode_id][:2] == expected, "actual model input differs from frozen cohort")
        rows = [json.loads(line) for line in (merged / "decisions.jsonl").read_text().splitlines() if line]
        require([row["episode_id"] for row in rows] == cohort["episode_ids"]
                and all(row["turn_id"] == "t1" and row["termination"] ==
                        batch["attempts"][row["episode_id"]][-1]["termination"] for row in rows),
                "decision coverage or attempt termination differs")
        _eval_run(merged)
        scored = read(merged / "metrics.json")["scored"]
        require(scored["scorer"] == SCORER and set(scored["per_episode"]) == set(cohort["episode_ids"])
                and scored["aggregate"]["accuracy_heuristic"]["den"] == cohort["count"],
                "scorer or denominator mismatch")
        by_model[name] = scored["per_episode"]
        retries = out / "transport_retries.jsonl"
        results[name] = {
            "model_config": batch["model_config"], "aggregate": scored["aggregate"],
            "by_question_type": scored["by_question_type"],
            "terminations": dict(Counter(row["termination"] for row in rows)),
            "reported_tokens": sum((row.get("usage") or {}).get("total_tokens", 0) for row in rows),
            "duration_ms": sum(row.get("duration_ms", 0) for row in rows),
            "transport_retry_events": len(retries.read_text().splitlines()) if retries.exists() else 0,
            "merged_run": str(merged),
        }
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            differences = [by_model[left][episode_id]["accuracy_heuristic"] -
                           by_model[right][episode_id]["accuracy_heuristic"]
                           for episode_id in cohort["episode_ids"]]
            paired[f"{left}_vs_{right}"] = {"left_higher": sum(d > 1e-12 for d in differences),
                                             "right_higher": sum(d < -1e-12 for d in differences),
                                             "ties": sum(abs(d) <= 1e-12 for d in differences),
                                             "mean_difference": sum(differences) / len(differences)}
    return {"protocol": PROTOCOL, "count": cohort["count"], "seed": cohort["seed"],
            "cohort_sha256": _sha(root / "cohort.json"), "results": results, "paired": paired,
            "shared_controls": {"timeout": shared[0], "max_calls": shared[1], "temperature": shared[2],
                                "delay": shared[3], "max_attempts": shared[4]},
            "caveat": "PR #12 configurations retained: GLM has no explicit max_tokens; Qwen/DeepSeek use 8192. Provider thinking defaults retained. Preliminary API comparison, not fully matched decoding."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--config", type=Path, default=Path("ama.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.benchmark.resolve()
    if args.prepare_only:
        if args.dataset is None:
            parser.error("--prepare-only requires --dataset")
        cohort = prepare(args.dataset, root, count=args.count, seed=args.seed)
        print(f"prepared {cohort['count']} test episodes: {root}")
        return 0
    if not args.model or len(args.model) != len(set(args.model)) or args.runs_root is None or args.workers < 1:
        parser.error("provide distinct --model aliases, --runs-root and positive --workers")
    runs_root = args.runs_root.resolve()
    if not args.compare_only:
        _load_dotenv(args.env_file)
        errors = []
        with ThreadPoolExecutor(max_workers=min(args.workers, len(args.model))) as pool:
            futures = {pool.submit(run_model, root, name, config=args.config, runs_root=runs_root,
                                   timeout=args.timeout, delay=args.delay, max_attempts=args.max_attempts,
                                   max_new_cases=args.max_new_cases): name for name in args.model}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{futures[future]}: {type(exc).__name__}: {exc}")
        if errors:
            for error in errors:
                print(error, flush=True)
            return 1
        if args.max_new_cases is not None:
            return 0
    report = compare(root, args.model, runs_root)
    write_json(runs_root / "comparison.json", report)
    print(f"comparison: {runs_root / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
