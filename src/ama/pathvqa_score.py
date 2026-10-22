"""Conservative PathVQA answer overlap metrics for AMA runs."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from .data import Dataset


def normalize_answer(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def _f1(prediction: str, reference: str) -> float:
    pred, gold = Counter(prediction.split()), Counter(reference.split())
    common = sum((pred & gold).values())
    return 2 * common / (sum(pred.values()) + sum(gold.values())) if pred and gold else 0.0


def score_pathvqa(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from .scorer import expected_rows

    per_episode = {}
    counts = {"total": 0, "submitted": 0, "valid_answer": 0, "binary_total": 0,
              "binary_valid_choice": 0, "binary_correct": 0, "open_total": 0, "open_exact": 0}
    open_f1 = 0.0
    for ep in dataset.episodes:
        target = dataset.targets.get(ep.id)
        rows = expected_rows(ep, decisions)
        turns = []
        for row in rows:
            gold = (target.turns.get(row["turn_id"]) if target else None) or {}
            if not gold:
                raise ValueError(f"missing PathVQA target: {ep.id}/{row['turn_id']}")
            kind = gold.get("question_type")
            if kind not in {"binary", "open"}:
                raise ValueError(f"invalid PathVQA question type: {kind!r}")
            reference = normalize_answer(gold.get("answer"))
            if not reference:
                raise ValueError(f"empty PathVQA reference: {ep.id}/{row['turn_id']}")
            decision = row.get("decision") or {}
            raw_answer = decision.get("answer") if isinstance(decision, dict) else None
            answer = normalize_answer(raw_answer)
            valid = bool(answer)
            exact = int(bool(answer) and answer == reference)
            counts["total"] += 1
            counts["submitted"] += int(row.get("decision") is not None)
            counts["valid_answer"] += int(valid)
            entry = {"turn_id": row["turn_id"], "question_type": kind,
                     "submitted": row.get("decision") is not None, "valid_answer": valid,
                     "exact_match": exact, "termination": row.get("termination")}
            if kind == "binary":
                counts["binary_total"] += 1
                # Binary output must be exactly one of the two permitted labels.
                entry["valid_choice"] = answer in {"yes", "no"}
                entry["correct"] = int(entry["valid_choice"] and exact)
                counts["binary_valid_choice"] += int(entry["valid_choice"])
                counts["binary_correct"] += entry["correct"]
            else:
                counts["open_total"] += 1
                counts["open_exact"] += exact
                entry["token_f1"] = _f1(answer, reference)
                open_f1 += entry["token_f1"]
            turns.append(entry)
        per_episode[ep.id] = {"turns": turns}

    def rate(num: int | float, den: int) -> float | None:
        return num / den if den else None

    return {"scorer": "pathvqa_v1", "per_episode": per_episode, "aggregate": {
        "n_expected": counts["total"],
        "completion_rate": rate(counts["submitted"], counts["total"]),
        "valid_answer_rate": rate(counts["valid_answer"], counts["total"]),
        "binary_valid_choice_rate": rate(counts["binary_valid_choice"], counts["binary_total"]),
        "binary_accuracy": rate(counts["binary_correct"], counts["binary_total"]),
        "binary_count": counts["binary_total"],
        "open_exact_match": rate(counts["open_exact"], counts["open_total"]),
        "open_token_f1": rate(open_f1, counts["open_total"]),
        "open_count": counts["open_total"],
    }}

