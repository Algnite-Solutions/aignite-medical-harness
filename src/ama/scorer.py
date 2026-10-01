"""ROCOv2 scoring: caption token overlap and concept-ID overlap."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from .data import Dataset


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"num": num, "den": den, "value": (num / den) if den else None}


def _base_turn_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": row["turn_id"],
        "submitted": row["decision"] is not None,
        "turn_termination": row["termination"],
        "model_calls": row["model_calls"],
        "usage": row["usage"],
        "duration_ms": row["duration_ms"],
    }


def expected_rows(episode, decisions):
    """Missing/failed answers remain in the expected-turn denominator."""
    indexed = {row["turn_id"]: row for row in decisions.get(episode.id, [])}
    return [{"turn_id": turn.id, "decision": None, "termination": "not_executed",
             "model_calls": 0, "usage": None, "duration_ms": 0,
             **indexed.get(turn.id, {})} for turn in episode.turns]


def score_unscored(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per = {}
    for ep in dataset.episodes:
        rows = expected_rows(ep, decisions)
        usage = [r["usage"] for r in rows if isinstance(r.get("usage"), dict)]
        per[ep.id] = {
            "completion": _rate(sum(1 for r in rows if r["decision"]), len(rows)),
            "terminations": sorted({r["termination"] for r in rows}),
            "model_calls": sum(r["model_calls"] for r in rows),
            "tokens": sum(u.get("total_tokens", 0) for u in usage) if usage else "unknown",
        }
    return {"scorer": "unscored", "per_episode": per, "aggregate": None}


# ---------------------------------------------------------------- rocov2

_WORD_RE = re.compile(r"[a-z0-9]+")


def _caption_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return _WORD_RE.findall(unicodedata.normalize("NFKC", value).lower())


def _cui_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {v.strip().upper() for v in value if isinstance(v, str) and v.strip()}


def _prf(overlap: int, predicted: int, gold: int) -> dict[str, dict[str, Any]]:
    return {
        "precision": _rate(overlap, predicted),
        "recall": _rate(overlap, gold),
        "f1": _rate(2 * overlap, predicted + gold),
    }


def score_rocov2(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Score normalized caption token overlap and unordered UMLS CUI overlap."""
    per_episode: dict[str, Any] = {}
    totals = {"caption_overlap": 0, "caption_predicted": 0, "caption_gold": 0,
              "cui_overlap": 0, "cui_predicted": 0, "cui_gold": 0,
              "refs_covered": 0, "refs_total": 0}
    for ep in dataset.episodes:
        rows = expected_rows(ep, decisions)
        target = dataset.targets.get(ep.id)
        turns_out: list[dict[str, Any]] = []
        for row in rows:
            entry = _base_turn_entry(row)
            t = target.turns.get(row["turn_id"]) if target else None
            if t is None:
                entry["scored"] = False
                turns_out.append(entry)
                continue
            decision = row.get("decision") or {}
            state = decision.get("answer") if isinstance(decision.get("answer"), dict) else {}
            want = t.get("answer") if isinstance(t.get("answer"), dict) else {}
            predicted_tokens = Counter(_caption_tokens(state.get("caption")))
            gold_tokens = Counter(_caption_tokens(want.get("caption")))
            caption_overlap = sum((predicted_tokens & gold_tokens).values())
            predicted_cuis = _cui_set(state.get("cuis"))
            gold_cuis = _cui_set(want.get("cuis"))
            cui_overlap = len(predicted_cuis & gold_cuis)
            caption = _prf(caption_overlap, sum(predicted_tokens.values()), sum(gold_tokens.values()))
            cui = _prf(cui_overlap, len(predicted_cuis), len(gold_cuis))
            entry.update({
                "caption_token_precision": caption["precision"],
                "caption_token_recall": caption["recall"],
                "caption_token_f1": caption["f1"],
                "cui_precision": cui["precision"],
                "cui_recall": cui["recall"],
                "cui_f1": cui["f1"],
            })
            required = set(t.get("required_evidence") or [])
            if required:
                entry["refs_covered"] = required <= set(decision.get("citations") or [])
                totals["refs_covered"] += int(entry["refs_covered"])
                totals["refs_total"] += 1
            totals["caption_overlap"] += caption_overlap
            totals["caption_predicted"] += sum(predicted_tokens.values())
            totals["caption_gold"] += sum(gold_tokens.values())
            totals["cui_overlap"] += cui_overlap
            totals["cui_predicted"] += len(predicted_cuis)
            totals["cui_gold"] += len(gold_cuis)
            entry["scored"] = True
            turns_out.append(entry)
        per_episode[ep.id] = {
            "turns": turns_out,
            "target_present": target is not None,
            "completion": _rate(sum(1 for r in rows if r.get("decision")), len(rows)),
        }
    caption = _prf(totals["caption_overlap"], totals["caption_predicted"], totals["caption_gold"])
    cui = _prf(totals["cui_overlap"], totals["cui_predicted"], totals["cui_gold"])
    return {"scorer": "rocov2", "per_episode": per_episode, "aggregate": {
        "caption_token_precision": caption["precision"],
        "caption_token_recall": caption["recall"],
        "caption_token_f1": caption["f1"],
        "cui_precision": cui["precision"],
        "cui_recall": cui["recall"],
        "cui_f1": cui["f1"],
        "refs_covered": _rate(totals["refs_covered"], totals["refs_total"]),
    }}



from .importers.mimic_cdm import score_mimic_cdm
from .importers.mimic_cdm_benchmark import score_mimic_cdm_open

REGISTRY = {"rocov2": score_rocov2, "unscored": score_unscored,
            "mimic_cdm": score_mimic_cdm, "mimic_cdm_open": score_mimic_cdm_open}
