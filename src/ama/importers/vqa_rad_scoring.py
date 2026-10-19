"""AMA adapter of AutoMedBench's deterministic VQA heuristic (no LLM judge).

Normalization and metric semantics follow eval_vqa at commit
5a19ff72491df7f8c0279438fcfade9364f9a8b7 (non-commercial research only).
Upstream copyright/permission: docs/licenses/AutoMedBench-LICENSE.txt.
See docs/vqa-rad.md.
"""
from __future__ import annotations

import string
from collections import Counter
from typing import Any

from ..data import Dataset

SCORER = "vqa_rad_automedbench_v1"
ABBREVIATIONS = {
    "r": "right", "l": "left", "bil": "bilateral", "bilat": "bilateral",
    "ivc": "inferior vena cava", "svc": "superior vena cava",
    "lv": "left ventricle", "rv": "right ventricle", "la": "left atrium",
    "ra": "right atrium", "lad": "left anterior descending",
    "mca": "middle cerebral artery", "ica": "internal carotid artery",
    "pca": "posterior cerebral artery", "aca": "anterior cerebral artery",
    "pe": "pulmonary embolism", "dvt": "deep vein thrombosis",
    "chf": "congestive heart failure", "cad": "coronary artery disease",
    "gi": "gastrointestinal", "gu": "genitourinary", "cns": "central nervous system",
}
YES_NO = {**dict.fromkeys(("yes", "y", "yeah", "yep", "true", "present", "positive"), "yes"),
          **dict.fromkeys(("no", "n", "nope", "false", "absent", "negative"), "no")}
NUMBERS = {word: str(n) for n, word in enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve"))}
NUMBERS.update({"none": "0", "single": "1"})


def normalize_answer(text: str) -> str:
    """Lowercase, delete ASCII punctuation, collapse spaces, then map tokens.

    Yes/no synonyms apply only to the entire answer, before number/abbreviation
    expansion. No article removal, Unicode folding, substring or fuzzy match.
    """
    value = " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())
    if value in YES_NO:
        return YES_NO[value]
    tokens = [NUMBERS.get(token, token) for token in value.split()]
    return " ".join(ABBREVIATIONS.get(token, token) for token in tokens)


def answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Per-question scores in [0, 1]; F1 uses a token multiset, then macro mean."""
    pred, gold = normalize_answer(prediction), normalize_answer(reference)
    em = float(bool(pred and gold) and pred == gold)
    a, b = Counter(pred.split()), Counter(gold.split())
    overlap = sum((a & b).values())
    f1 = 2 * overlap / (sum(a.values()) + sum(b.values())) if a and b else 0.0
    return {"exact_match": em, "token_f1": f1,
            "accuracy_heuristic": em if gold in {"yes", "no"} else (em + f1) / 2}


def score_vqa_rad(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Missing, failed, empty and malformed answers remain in the denominator."""
    from ..scorer import _rate, expected_rows

    if set(decisions) - {ep.id for ep in dataset.episodes}:
        raise ValueError("unexpected decision episodes")
    totals = Counter()
    groups: dict[str, Counter] = {name: Counter() for name in ("open", "closed_yes_no")}
    per_episode = {}
    for ep in dataset.episodes:
        if len(ep.turns) != 1:
            raise ValueError(f"{ep.id}: VQA-RAD expects one turn")
        turn = ep.turns[0]
        rows = decisions.get(ep.id, [])
        if len(rows) > 1 or rows and rows[0].get("turn_id") != turn.id:
            raise ValueError(f"{ep.id}: duplicate or unexpected decision turn")
        target = dataset.targets.get(ep.id)
        payload = target.turns.get(turn.id, {}) if target else {}
        answer = payload.get("answer")
        gold = answer.get("text") if isinstance(answer, dict) else None
        kind = payload.get("question_type")
        if (not isinstance(gold, str) or not gold.strip()
                or kind not in groups
                or kind != ("closed_yes_no" if gold.strip().lower() in {"yes", "no"} else "open")):
            raise ValueError(f"{ep.id}: missing or invalid VQA-RAD target")
        row = expected_rows(ep, decisions)[0]
        decision = row["decision"]
        submitted = isinstance(decision, dict)
        answer = decision.get("answer") if submitted else None
        # AMA's task contract is one text field; no coercion of lists/numbers.
        pred = answer.get("text") if isinstance(answer, dict) and set(answer) == {"text"} else None
        valid = isinstance(pred, str) and bool(pred.strip()) and row["termination"] == "completed"
        scores = answer_metrics(pred if valid else "", gold)
        normalized_yes_no = normalize_answer(gold) in {"yes", "no"}
        counts = {**scores, "count": 1, "completion": int(submitted), "valid_answer": int(valid),
                  "yes_no_correct": scores["exact_match"] if normalized_yes_no else 0,
                  "yes_no_count": int(normalized_yes_no)}
        totals.update(counts)
        groups[kind].update(counts)
        per_episode[ep.id] = {"turn_id": turn.id, "question_type": kind,
                              "normalized_yes_no": normalized_yes_no,
                              "submitted": submitted, "valid_answer": valid,
                              "termination": row["termination"], **scores}

    def aggregate(values):
        return {**{name: _rate(values[name], values["count"]) for name in
                   ("accuracy_heuristic", "exact_match", "token_f1", "completion", "valid_answer")},
                "yes_no_accuracy": _rate(values["yes_no_correct"], values["yes_no_count"])}

    return {"scorer": SCORER, "per_episode": per_episode, "aggregate": aggregate(totals),
            "by_question_type": {name: aggregate(values) for name, values in groups.items()}}
