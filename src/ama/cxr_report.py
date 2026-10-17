"""Offline findings scoring using pinned AutoMedBench report code.

Data flow: AMA Decision.answer.findings -> temporary report.txt -> upstream
format checker and pair/corpus metrics -> AMA aggregation. No model API calls.
The paper's seven-metric backend is explicit and never silently substituted.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from .data import Dataset

SCORER = "mimic_cxr_automedbench_v1"
MLRG_SCORER = "mimic_cxr_automedbench_mlrg_v1"
REVISION = "a4e64e5502c728cee744edce2b596000fcd60fdd"
METRICS = ("BLEU", "METEOR", "ROUGE_L", "F1RadGraph", "micro_average_precision",
           "micro_average_recall", "micro_average_f1")
LIMITS = {"min_chars": 40, "max_chars": 8000, "min_alpha_chars": 20}


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def _mlrg_config(dataset: Dataset) -> dict:
    """Explicit local paths only; this function never fetches model resources."""
    rules = dataset.eval_config.rules if dataset.eval_config else {}
    config = dict(rules.get("mlrg", {}))
    path = os.environ.get("AMA_CXR_MLRG_CONFIG")
    if path:
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    for key in ("mlrg_repo_path", "chexbert_path", "bert_path", "radgraph_path"):
        if not isinstance(config.get(key), str) or not Path(config[key]).exists():
            raise ValueError(f"MLRG resource missing: {key}. Set local paths through "
                             "AMA_CXR_MLRG_CONFIG; no lightweight fallback was used.")
    return {**config, "clinical_score_backend": "mlrg"}


def _score(dataset: Dataset, decisions: dict, *, backend: str) -> dict:
    from ._vendor.automedbench_report.format_checker import check_report_file
    from ._vendor.automedbench_report.report_scorer import normalize_text, score_all, score_report_pair
    from .scorer import expected_rows

    scorer = MLRG_SCORER if backend == "mlrg" else SCORER
    config = _mlrg_config(dataset) if backend == "mlrg" else {}
    per_episode, cases = {}, []
    with tempfile.TemporaryDirectory(prefix="ama-cxr-score-") as temp:
        root = Path(temp)
        for episode in dataset.episodes:
            entries = []
            target = dataset.targets.get(episode.id)
            for row in expected_rows(episode, decisions):
                reference = ((target.turns.get(row["turn_id"], {}).get("answer") or {})
                             if target else {})
                reference = reference.get("findings") if isinstance(reference, dict) else None
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"missing reference findings: {episode.id}/{row['turn_id']}")
                decision = row.get("decision")
                answer = decision.get("answer") if isinstance(decision, dict) else None
                text = answer.get("findings") if isinstance(answer, dict) else None
                # Never recover references, raw model prose, or extra fields here.
                # The runner's saved Decision remains the inference contract.
                submitted = row["termination"] == "completed" and isinstance(text, str)
                case_id = f"case-{len(cases):06d}"
                pred_path = root / "pred" / case_id / "report.txt"
                if submitted:
                    pred_path.parent.mkdir(parents=True)
                    pred_path.write_text(text, encoding="utf-8")
                checked = check_report_file(pred_path, **LIMITS)
                valid = checked["valid"]
                pair = score_report_pair(reference, text if submitted else "")
                # Upstream's rule F1 can be 1 for an empty/normal report pair.
                # Invalid or absent AMA outputs must not earn that credit.
                entry = {"turn_id": row["turn_id"], "termination": row["termination"],
                         "metric_backend": "lightweight",
                         "report_format_valid": valid,
                         "report_format_errors": checked["errors"],
                         "report_char_count": checked["char_count"],
                         "observation_f1": pair["observation_f1"] if valid else 0.0,
                         "rouge_l_f1": pair["report_similarity"] if valid else 0.0,
                         "label_exact_match": pair["label_exact_match"] if valid else 0,
                         "case_score": round(0.7 * pair["observation_f1"] +
                                             0.3 * pair["report_similarity"], 4) if valid else 0.0}
                entries.append(entry)
                cases.append({"id": case_id, "reference": reference,
                              "prediction": text if submitted else "", "entry": entry})
            per_episode[episode.id] = {"turns": entries}

        if not cases:
            raise ValueError("no expected turns to score")
        valid = [c for c in cases if c["entry"]["report_format_valid"]]
        complete = len(valid) == len(cases)
        observation = _mean([c["entry"]["observation_f1"] for c in cases])
        rouge = _mean([c["entry"]["rouge_l_f1"] for c in cases])
        case_mean = _mean([c["entry"]["case_score"] for c in cases])
        aggregate = {
            "backend": backend, "cases_total": len(cases), "valid_reports": len(valid),
            "report_completion_rate": len(valid) / len(cases),
            "submission_format_valid": complete,
            "mean_observation_f1": observation, "mean_rouge_l_f1": rouge,
            "case_mean_score": case_mean,
            "valid_case_mean_score": _mean([c["entry"]["case_score"] for c in valid]),
            "valid_mean_observation_f1": _mean([c["entry"]["observation_f1"] for c in valid]),
            "valid_mean_rouge_l_f1": _mean([c["entry"]["rouge_l_f1"] for c in valid]),
            "complete_submission_score": (round(0.7 * observation + 0.3 * rouge, 4)
                                          if complete else 0.0),
            "mlrg_metrics": dict.fromkeys(METRICS),
        }
        if backend == "mlrg":
            # Match upstream corpus aggregation, including empty predictions for
            # missing AMA Decisions. Do not multiply corpus micro-F1 by coverage.
            for case in cases:
                gt = root / "gt" / case["id"] / "report.txt"
                gt.parent.mkdir(parents=True)
                gt.write_text(case["reference"], encoding="utf-8")
            scores = score_all(root / "pred", root / "gt", [c["id"] for c in cases], config)
            required = METRICS
            if any(normalize_text(c["prediction"]) for c in cases):
                required = (*METRICS, "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
                            "chexbert_all_micro_p", "chexbert_all_micro_r",
                            "chexbert_all_micro_f1", "F1-Radgraph-partial")
            if any(not isinstance(scores.get(k), (int, float)) or not math.isfinite(scores[k])
                   for k in required):
                raise ValueError("MLRG returned incomplete/non-finite metrics; no default zeros used")
            values = {k: scores[k] for k in METRICS}
            aggregate["lightweight_case_mean_score"] = aggregate.pop("case_mean_score")
            aggregate["valid_lightweight_case_mean_score"] = aggregate.pop("valid_case_mean_score")
            aggregate.update(mlrg_metrics=values,
                             mlrg_raw_corpus_score=round(sum(0.142857 * v for v in values.values()), 4),
                             complete_submission_score=(round(sum(0.142857 * v for v in values.values()), 4)
                                                        if complete else 0.0))
    return {"scorer": scorer, "per_episode": per_episode, "aggregate": aggregate,
            "protocol": {
                "repository": "https://github.com/AutoMedBench/AutoMedBench",
                "revision": REVISION, "official_default_backend": "mlrg",
                "active_backend": backend, "report_format_limits": LIMITS,
                "lightweight_weights": {"observation_f1": 0.7, "rouge_l_f1": 0.3},
                "primary_diagnostic_policy": "invalid/missing AMA outputs score zero per case; all expected cases remain",
                "complete_submission_policy": "score zero unless all reports pass upstream format checks",
                "observation_labels": "12 rule-derived text labels, not CheXbert or RadGraph",
                "mlrg_aggregation": "upstream corpus aggregation; no per-case micro-F1 approximation",
                "mlrg_status": "computed" if backend == "mlrg" else "not_requested; resources required separately",
                "official_leaderboard_result": False,
            }}


def score_mimic_cxr_report(dataset: Dataset, decisions: dict) -> dict:
    return _score(dataset, decisions, backend="lightweight")


def score_mimic_cxr_report_mlrg(dataset: Dataset, decisions: dict) -> dict:
    return _score(dataset, decisions, backend="mlrg")
