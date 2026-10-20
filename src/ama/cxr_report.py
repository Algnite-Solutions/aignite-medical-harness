"""Offline findings scoring with AutoMedBench's lightweight report metrics.

AMA Decisions and evaluator-only references feed the upstream format checker
and pair metrics. Invalid outputs score zero; no model API calls are made.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .data import Dataset

SCORER = "mimic_cxr_automedbench_v1"
REVISION = "a4e64e5502c728cee744edce2b596000fcd60fdd"
LIMITS = {"min_chars": 40, "max_chars": 8000, "min_alpha_chars": 20}


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def score_mimic_cxr_report(dataset: Dataset, decisions: dict, *, histories=None) -> dict:
    """Score all expected turns, keeping failed outputs in the denominator."""
    from ._vendor.automedbench_report.format_checker import check_report_file
    from ._vendor.automedbench_report.report_scorer import score_report_pair
    from .scorer import expected_rows

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
                cases.append(entry)
            per_episode[episode.id] = {"turns": entries}

        if not cases:
            raise ValueError("no expected turns to score")
        valid = [c for c in cases if c["report_format_valid"]]
        complete = len(valid) == len(cases)
        observation = _mean([c["observation_f1"] for c in cases])
        rouge = _mean([c["rouge_l_f1"] for c in cases])
        case_mean = _mean([c["case_score"] for c in cases])
        aggregate = {
            "backend": "lightweight", "cases_total": len(cases), "valid_reports": len(valid),
            "report_completion_rate": len(valid) / len(cases),
            "submission_format_valid": complete,
            "mean_observation_f1": observation, "mean_rouge_l_f1": rouge,
            "case_mean_score": case_mean,
            "valid_case_mean_score": _mean([c["case_score"] for c in valid]),
            "valid_mean_observation_f1": _mean([c["observation_f1"] for c in valid]),
            "valid_mean_rouge_l_f1": _mean([c["rouge_l_f1"] for c in valid]),
            "complete_submission_score": (round(0.7 * observation + 0.3 * rouge, 4)
                                          if complete else 0.0),
        }
    return {"scorer": SCORER, "per_episode": per_episode, "aggregate": aggregate,
            "protocol": {
                "repository": "https://github.com/AutoMedBench/AutoMedBench",
                "revision": REVISION,
                "active_backend": "lightweight", "report_format_limits": LIMITS,
                "lightweight_weights": {"observation_f1": 0.7, "rouge_l_f1": 0.3},
                "primary_diagnostic_policy": "invalid/missing AMA outputs score zero per case; all expected cases remain",
                "complete_submission_policy": "score zero unless all reports pass upstream format checks",
                "observation_labels": "12 rule-derived text labels, not CheXbert or RadGraph",
                "official_leaderboard_result": False,
            }}
