"""Compare matched open-answer runs without making model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runner import DECISION_PROMPT
from .mimic_cdm_benchmark import INSTRUCTIONS, VARIANTS


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired(left: dict, right: dict, ids: list[str]) -> dict:
    both = left_only = right_only = neither = 0
    for case_id in ids:
        a = bool(left[case_id]["correct_auto"])
        b = bool(right[case_id]["correct_auto"])
        both += a and b
        left_only += a and not b
        right_only += b and not a
        neither += not a and not b
    return {"both_correct": both, "left_only": left_only,
            "right_only": right_only, "neither": neither}


def compare_runs(runs: dict[str, dict[str, Path]]) -> dict:
    """Check a two-model, three-view matrix and summarize paired outcomes."""
    if len(runs) != 2 or any(set(variants) != set(VARIANTS) for variants in runs.values()):
        raise ValueError("provide exactly two models with hpi, interactive and full_info runs")
    baseline_ids = None
    benchmark_root = None
    timeout = None
    target_rows = None
    scores: dict[str, dict] = {}
    for model, variants in runs.items():
        scores[model] = {}
        model_config = None
        for variant in VARIANTS:
            folder = Path(variants[variant])
            manifest = _read(folder / "manifest.json")
            metrics = _read(folder / "metrics.json")
            messages = _read(folder / "messages.json")
            decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            ids = manifest["episode_ids"]
            if not ids or len(ids) != len(set(ids)) or set(messages) != set(ids):
                raise ValueError(f"invalid episode coverage: {folder}")
            if baseline_ids is None:
                baseline_ids = ids
            if ids != baseline_ids:
                raise ValueError(f"case IDs or order differ: {folder}")
            dataset = Path(manifest["dataset_dir"]).resolve()
            if dataset.name != f"mimic_cdm_open_{variant}":
                raise ValueError(f"wrong benchmark view: {folder}")
            if benchmark_root is None:
                benchmark_root = dataset.parent
            if dataset.parent != benchmark_root:
                raise ValueError(f"benchmark roots differ: {folder}")
            current_targets = (dataset / "targets.jsonl").read_text(encoding="utf-8").splitlines()
            if target_rows is None:
                target_rows = [json.loads(line) for line in current_targets]
            if [json.loads(line) for line in current_targets] != target_rows:
                raise ValueError(f"evaluator targets differ across views: {folder}")
            if any(not history or history[0].get("role") != "system" or
                   history[0].get("content") != INSTRUCTIONS[variant] + "\n\n" + DECISION_PROMPT
                   for history in messages.values()):
                raise ValueError(f"actual system prompt differs from benchmark task: {folder}")
            benchmark = _read(benchmark_root / "benchmark.json")
            if benchmark.get("protocol") != "mimic_cdm_open_v1" or \
                    benchmark.get("episode_ids") != ids:
                raise ValueError(f"benchmark protocol or case order differs: {folder}")
            if bool(manifest.get("tools")) != (variant == "interactive"):
                raise ValueError(f"wrong tool availability for {variant}: {folder}")
            if manifest["mode"] != "run" or manifest["model"] != model or \
                    metrics["model"] != model or metrics["scorer"] != "mimic_cdm_open" or \
                    metrics["run_id"] != manifest["run_id"]:
                raise ValueError(f"run/model/scorer mismatch: {folder}")
            if model_config is None:
                model_config = manifest["model_config"]
            if manifest["model_config"] != model_config:
                raise ValueError(f"model configuration changed across views: {model}")
            if timeout is None:
                timeout = manifest["timeout"]
            if manifest["timeout"] != timeout:
                raise ValueError("request timeouts differ across runs")
            per = metrics["scored"]["per_episode"]
            if set(per) != set(ids):
                raise ValueError(f"evaluation coverage differs from run: {folder}")
            if len(decisions) != len(ids) or [row["episode_id"] for row in decisions] != ids:
                raise ValueError(f"decision coverage differs from run: {folder}")
            aggregate = metrics["scored"]["aggregate"]
            calls = sum(len(message.get("tool_calls") or [])
                        for history in messages.values() for message in history
                        if message.get("role") == "assistant")
            results = sum(message.get("role") == "tool"
                          for history in messages.values() for message in history)
            scores[model][variant] = {
                "run": str(folder.resolve()),
                "correct_auto": aggregate["diagnosis_accuracy_auto"],
                "mapped": aggregate["diagnosis_mapped"],
                "needs_review": aggregate["needs_review"],
                "completion": aggregate["completion"],
                "output_quality": aggregate.get("output_quality"),
                "tool_calls": calls, "tool_results": results,
                "model_calls": sum(row["model_calls"] for row in decisions),
                "reported_tokens": sum((row.get("usage") or {}).get("total_tokens", 0)
                                       for row in decisions),
                "terminations": {status: sum(row["termination"] == status for row in decisions)
                                 for status in sorted({row["termination"] for row in decisions})},
                "per_episode": per,
            }
    comparisons = {}
    for model, variants in scores.items():
        comparisons[model] = {
            "interactive_vs_hpi": _paired(variants["interactive"]["per_episode"],
                                          variants["hpi"]["per_episode"], baseline_ids),
            "full_info_vs_interactive": _paired(variants["full_info"]["per_episode"],
                                                variants["interactive"]["per_episode"], baseline_ids),
        }
    left, right = list(scores)
    comparisons["models_on_full_info"] = _paired(scores[left]["full_info"]["per_episode"],
                                                  scores[right]["full_info"]["per_episode"],
                                                  baseline_ids)
    return {"protocol": "mimic_cdm_open_v1", "benchmark_root": str(benchmark_root),
            "episode_ids": baseline_ids, "timeout": timeout, "models": [left, right],
            "results": scores, "paired": comparisons,
            "interpretation": ("Full-info accuracy is the model comparison; interactive versus HPI "
                               "measures benefit from tool access; full-info versus interactive "
                               "measures the remaining evidence-access gap. Automatic diagnosis "
                               "scores are lower bounds until unmapped answers receive blinded review.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare two matched three-view MIMIC-CDM runs")
    parser.add_argument("--model", nargs=4, action="append", required=True,
                        metavar=("NAME", "HPI_RUN", "INTERACTIVE_RUN", "FULL_RUN"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {name: dict(zip(VARIANTS, map(Path, paths))) for name, *paths in args.model}
    report = compare_runs(runs)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in report["models"]:
        print(model, {variant: report["results"][model][variant]["correct_auto"]["value"]
                      for variant in VARIANTS})
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
