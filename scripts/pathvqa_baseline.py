"""Prepare and run a fixed, stratified PathVQA PR cohort through native AMA."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    # Hash JSON text consistently across Windows CRLF and Linux LF checkouts.
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def prepare(dataset: Path, out: Path, size: int = 100, seed: int = 42) -> dict:
    info = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    report = json.loads((dataset / "import_report.json").read_text(encoding="utf-8"))
    pool = info["splits"]["automedbench_test_500"]
    if size < 1 or size > len(pool):
        raise ValueError("cohort size must be within the frozen 500-case split")
    wanted = set(pool)
    groups = {"binary": [], "open": []}
    with (dataset / "targets.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in wanted:
                kind = row["turns"]["t1"]["question_type"]
                groups[kind].append(row["id"])
    if sum(map(len, groups.values())) != len(pool):
        raise ValueError("frozen split and evaluator targets do not match")
    n_binary = round(size * len(groups["binary"]) / len(pool))
    allocation = {"binary": n_binary, "open": size - n_binary}
    rng = random.Random(seed)
    selected = set()
    for kind in ("binary", "open"):
        selected.update(rng.sample(groups[kind], allocation[kind]))
    ids = [eid for eid in pool if eid in selected]
    cohort = {"schema": "ama-pathvqa-baseline-v1", "source_split": "automedbench_test_500",
              "size": size, "seed": seed, "question_types": allocation, "episode_ids": ids,
              "dataset_json_sha256": sha256(dataset / "dataset.json"),
              "targets_jsonl_sha256": sha256(dataset / "targets.jsonl"),
              "source_repository": report["repository"], "source_revision": report["revision"]}
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != cohort:
            raise ValueError("existing cohort differs; choose a new output path")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")
    return cohort


def run(dataset: Path, cohort_path: Path, aliases: list[str], config: Path,
        runs_root: Path, timeout: float) -> list[Path]:
    from ama.cli import _eval_run
    from ama.model import Model
    from ama.runner import execute

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    for filename, key in (("dataset.json", "dataset_json_sha256"),
                          ("targets.jsonl", "targets_jsonl_sha256")):
        if sha256(dataset / filename) != cohort[key]:
            raise ValueError(f"cohort source changed: {filename}")
    report = json.loads((dataset / "import_report.json").read_text(encoding="utf-8"))
    if any(report[key] != cohort["source_" + key] for key in ("repository", "revision")):
        raise ValueError("cohort source repository or revision changed")
    results = []
    for alias in aliases:
        model = Model.from_config(alias, config, timeout=timeout)
        run_dir = execute(dataset, model, episode_ids=cohort["episode_ids"],
                          runs_root=runs_root, max_calls=1)
        _eval_run(run_dir, "pathvqa_v1")
        results.append(run_dir)
    return results


def summarize(dataset: Path, cohort_path: Path, run_paths: list[Path], out: Path) -> dict:
    """Re-evaluate saved native runs and export only aggregate, auditable metadata."""
    from ama.cli import _eval_run

    cohort = prepare(dataset, cohort_path)
    summaries = []
    for path in run_paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = cohort["episode_ids"]
        if manifest["episode_ids"] != expected or len(rows) != len(expected) or \
                Counter(row["episode_id"] for row in rows) != Counter(expected) or \
                any(row["turn_id"] != "t1" for row in rows):
            raise ValueError(f"run does not cover the frozen cohort exactly: {path}")
        if manifest["termination"] == "running" or manifest["max_calls"] != 1 or \
                Path(manifest["dataset_dir"]).resolve() != dataset.resolve():
            raise ValueError(f"run protocol differs or is incomplete: {path}")
        events = [json.loads(line) for line in (path / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
        _eval_run(path, "pathvqa_v1")
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        config = manifest["model_config"]
        api_error_statuses = Counter()
        for event in events:
            if event.get("kind") == "error":
                match = re.search(r"HTTP (\d+)", event.get("error", ""))
                api_error_statuses[match.group(1) if match else "other"] += 1
        # Export an explicit allowlist rather than arbitrary provider configuration.
        safe_config = {key: config[key] for key in ("base_url", "model", "api_key_env", "temperature", "max_tokens",
                      "chat_template_kwargs") if key in config}
        summaries.append({"run_id": manifest["run_id"], "alias": manifest["model"],
                          "request_config": safe_config, "timeout": manifest["timeout"], "max_calls": 1,
                          "terminations": dict(Counter(row["termination"] for row in rows)),
                          "api_error_statuses": dict(api_error_statuses),
                          "aggregate": metrics["scored"]["aggregate"],
                          "files": {name: sha256(path / name) for name in
                                    ("manifest.json", "decisions.jsonl", "messages.json", "metrics.json")}})
    summary = {"schema": "ama-pathvqa-results-v1", "cohort_sha256": sha256(cohort_path),
               "source_repository": cohort["source_repository"],
               "source_revision": cohort["source_revision"],
               "source_split": "automedbench_test_500", "frozen_pool_size": 500,
               "cohort_size": cohort["size"], "question_types": cohort["question_types"],
               "scorer": "pathvqa_v1", "official_automedbench_score": False, "runs": summaries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--config", type=Path, default=Path("ama.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/pathvqa-pr100"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--summarize-run", type=Path, action="append", default=[])
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args(argv)
    cohort = prepare(args.dataset, args.cohort)
    print(f"cohort: {args.cohort} ({cohort['question_types']})")
    if args.summarize_run:
        if args.summary_out is None:
            parser.error("--summarize-run requires --summary-out")
        summarize(args.dataset, args.cohort, args.summarize_run, args.summary_out)
        return 0
    if not args.prepare_only:
        if not args.model:
            parser.error("provide at least one --model or use --prepare-only")
        for path in run(args.dataset, args.cohort, args.model, args.config,
                        args.runs_root, args.timeout):
            print(f"evaluated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
