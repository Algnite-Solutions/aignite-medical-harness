"""Compare a completed full-info batch with its paired interactive batch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ama.data import load_dataset
from ama.decisions import output_metrics
from ama.importers.mimic_cdm import score_mimic_cdm


def load(root, retry_root=None):
    batch = json.loads((root / "batch.json").read_text())
    if batch["status"] != "completed":
        raise ValueError(f"{root}: batch status is {batch['status']}")
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    dataset = load_dataset(root / "dataset", with_targets=True)
    rows = []
    for path in root.glob("shards/*/decisions.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows) != batch["count"]:
        raise ValueError(f"{root}: missing decision rows")
    original_terminations = dict(Counter(row["termination"] for row in rows))
    retried = []
    if retry_root:
        retry_batch = json.loads((retry_root / "batch.json").read_text())
        if retry_batch["status"] not in {"completed", "stopped_rate_limited"}:
            raise ValueError(f"{retry_root}: retry batch is still running")
        replacements = {}
        for path in retry_root.glob("shards/*/decisions.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                replacements[row["episode_id"]] = row
        original_by_id = {row["episode_id"]: row for row in rows}
        if not set(replacements) <= set(original_by_id):
            raise ValueError("retry contains an episode absent from the original batch")
        if any(original_by_id[eid]["termination"] == "completed" for eid in replacements):
            raise ValueError("retry contains an originally completed episode")
        retried = sorted(eid for eid, row in replacements.items() if row["termination"] == "completed")
        rows = [replacements.get(row["episode_id"], row)
                if row["episode_id"] in retried else row for row in rows]
    grouped = defaultdict(list)
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(Counter)
    for row in rows:
        grouped[row["episode_id"]].append(row)
        label = dataset.targets[row["episode_id"]].turns[row["turn_id"]]["answer"]["diagnosis"]
        answer = (row.get("decision") or {}).get("answer")
        prediction = answer.get("diagnosis") if isinstance(answer, dict) else answer
        prediction = prediction.strip().casefold() if isinstance(prediction, str) else "<missing>"
        breakdown[label]["total"] += 1
        breakdown[label]["correct"] += int(prediction == label)
        confusion[label][prediction] += 1
    metrics["scored"] = score_mimic_cdm(dataset, grouped)
    metrics["scored"]["aggregate"]["output_quality"] = output_metrics(rows, 2)
    metrics["per_diagnosis"] = dict(breakdown)
    metrics["confusion_matrix"] = {key: dict(value) for key, value in confusion.items()}
    metrics["original_terminations"] = original_terminations
    metrics["terminations"] = dict(Counter(row["termination"] for row in rows))
    metrics["successful_transport_retries"] = retried
    metrics["retry_batch"] = str(retry_root.resolve()) if retry_root else None
    metrics["scoring_revision"] = "2026-09-27: recover string diagnosis; count object format separately"
    original = root / "metrics_original.json"
    if not original.exists():
        shutil.copyfile(metrics_path, original)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return batch, metrics


def summarize(metrics, ids):
    per = metrics["scored"]["per_episode"]
    successes = sum(per[i]["correct"] for i in ids)
    return {"correct": successes, "total": len(ids), "accuracy": successes / len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path)
    parser.add_argument("interactive", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--full-retry", type=Path)
    parser.add_argument("--interactive-retry", type=Path)
    parser.add_argument("--wait", action="store_true", help="wait for both batches to complete")
    args = parser.parse_args()
    if args.wait:
        roots = [args.full, args.interactive]
        if args.full_retry:
            roots.append(args.full_retry)
        if args.interactive_retry:
            roots.append(args.interactive_retry)
        while True:
            states = []
            for root in roots:
                path = root / "batch.json"
                states.append(json.loads(path.read_text())["status"] if path.exists() else "pending")
            if all(state == "completed" for state in states):
                break
            if any(state not in {"pending", "running", "completed"} for state in states):
                raise ValueError(f"batch statuses: {states}")
            print(f"Waiting for batches: {states}", flush=True)
            time.sleep(60)
    full_batch, full = load(args.full, args.full_retry)
    interactive_batch, interactive = load(args.interactive, args.interactive_retry)
    ids = interactive_batch["episode_ids"]
    if not set(ids) <= set(full_batch["episode_ids"]):
        raise ValueError("interactive cases are not a subset of full-info cases")
    if full_batch["model_config"] != interactive_batch["model_config"]:
        raise ValueError("model configurations differ")
    fper = full["scored"]["per_episode"]
    iper = interactive["scored"]["per_episode"]
    transitions = Counter((fper[i]["correct"], iper[i]["correct"]) for i in ids)
    tool_calls = Counter()
    for shard in args.interactive.glob("shards/*"):
        transcript = shard / "messages.json"
        if transcript.exists():
            messages = [message for history in json.loads(transcript.read_text()).values()
                        for message in history]
        else:
            events = shard / "events.jsonl"
            if not events.exists():
                continue
            records = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
            messages = [record.get("message", record) for record in records]
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                tool_calls[call["function"]["name"]] += 1
    def quality(metrics):
        return metrics["scored"]["aggregate"]["output_quality"]
    report = {"model": full_batch["model"], "paired_cases": len(ids),
              "full_info_all": summarize(full, full_batch["episode_ids"]),
              "full_info_paired": summarize(full, ids),
              "interactive_paired": summarize(interactive, ids),
              "paired_outcomes": {"both_correct": transitions[(True, True)],
                                  "full_only_correct": transitions[(True, False)],
                                  "interactive_only_correct": transitions[(False, True)],
                                  "both_wrong": transitions[(False, False)]},
              "output_quality": {"full_info": quality(full), "interactive": quality(interactive)},
              "terminations": {"full_info": full["terminations"],
                               "interactive": interactive["terminations"]},
              "original_terminations": {"full_info": full["original_terminations"],
                                        "interactive": interactive["original_terminations"]},
              "successful_transport_retries": {"full_info": full["successful_transport_retries"],
                                               "interactive": interactive["successful_transport_retries"]},
              "tool_calls": dict(tool_calls),
              "mean_model_calls_per_case": {"full_info": full["model_calls"] / full_batch["count"],
                                            "interactive": interactive["model_calls"] / len(ids)},
              "median_duration_ms": {"full_info": full["median_duration_ms"],
                                     "interactive": interactive["median_duration_ms"]},
              "total_tokens": {"full_info": full["total_tokens"],
                               "interactive": interactive["total_tokens"]}}
    output = args.out or args.interactive / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"comparison: {output}")


if __name__ == "__main__":
    main()
