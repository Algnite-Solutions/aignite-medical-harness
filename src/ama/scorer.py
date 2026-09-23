"""Evaluator-only scorers. The run path never loads targets or eval rules."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Protocol

from .data import Dataset, Decision, Episode

_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?")


class Scorer(Protocol):
    def score(self, dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]: ...


# ---------------------------------------------------------------- helpers

def _values_equal(a: Any, b: Any) -> bool:
    """Explicit scorer policy: strict match, with one narrow tolerance for
    numeric-vs-numeric-string ("1.2 mg/dL" == 1.2)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    for x, y in ((a, b), (b, a)):
        if isinstance(x, (int, float)) and isinstance(y, str):
            m = _NUM_RE.match(y)
            if m is not None and math.isclose(x, float(m.group()), rel_tol=1e-9, abs_tol=1e-12):
                return True
    return a == b


def _states_equal(got: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(k in got and _values_equal(got[k], v) for k, v in want.items())


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"num": num, "den": den, "value": (num / den) if den else None}


def _decisions_by_turn(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    return {r["turn_id"]: (r["decision"] if r["decision"] else None) for r in rows}


def _base_turn_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": row["turn_id"],
        "submitted": row["decision"] is not None,
        "turn_termination": row["termination"],
        "model_calls": row["model_calls"],
        "usage": row["usage"],
        "duration_ms": row["duration_ms"],
    }


# ---------------------------------------------------------------- exact

def score_exact(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Target keys: answer (dict subset or scalar equality), answers (alternatives),
    expect_abstain, allowed_actions (answer.next_step), required_evidence."""
    per_episode: dict[str, Any] = {}
    tot = {k: 0 for k in ("answer_correct", "action_correct", "refs_covered")}
    dens = {k: 0 for k in tot}
    for ep in dataset.episodes:
        rows = decisions.get(ep.id, [])
        target = dataset.targets.get(ep.id)
        turns_out: list[dict[str, Any]] = []
        for row in rows:
            entry = _base_turn_entry(row)
            t = target.turns.get(row["turn_id"]) if target else None
            d = row["decision"] or {}
            if t is None:
                entry["scored"] = False
                turns_out.append(entry)
                continue
            answer = d.get("answer") if isinstance(d.get("answer"), dict) else {}
            if "answer" in t:
                dens["answer_correct"] += 1
                want = t["answer"]
                ok = (row["decision"] is not None and
                      (_states_equal(answer, want) if isinstance(want, dict)
                       else _values_equal(d.get("answer"), want)))
                entry["answer_correct"] = ok
                tot["answer_correct"] += int(ok)
            elif "answers" in t:
                dens["answer_correct"] += 1
                entry["answer_correct"] = ok = (row["decision"] is not None and
                    any(_values_equal(d.get("answer"), a) for a in t["answers"]))
                tot["answer_correct"] += int(ok)
            elif t.get("expect_abstain"):
                dens["answer_correct"] += 1
                entry["answer_correct"] = ok = row["decision"] is not None and d.get("answer") is None
                tot["answer_correct"] += int(ok)
            if t.get("allowed_actions"):
                dens["action_correct"] += 1
                name = answer.get("next_step")
                entry["action_correct"] = ok = name in t["allowed_actions"]
                tot["action_correct"] += int(ok)
            if t.get("required_evidence"):
                dens["refs_covered"] += 1
                cites = set(d.get("citations") or [])
                entry["refs_covered"] = ok = set(t["required_evidence"]) <= cites
                tot["refs_covered"] += int(ok)
            entry["scored"] = True
            turns_out.append(entry)
        decided = sum(1 for r in rows if r["decision"])
        per_episode[ep.id] = {
            "turns": turns_out,
            "target_present": target is not None,
            "completion": _rate(decided, len(rows)),
        }
    return {
        "scorer": "exact",
        "per_episode": per_episode,
        "aggregate": {k: _rate(tot[k], dens[k]) for k in tot},
    }


# ---------------------------------------------------------------- workflow

def _visible_tags(ep: Episode, upto_turn: str, evidence_tags: dict[str, list[str]]) -> set[str]:
    tags: set[str] = set()
    for t in ep.turns:
        tags |= {tag for e in t.evidence for tag in evidence_tags.get(e.id, [])}
        if t.id == upto_turn:
            break
    return tags


def _validate_transition(rules: list[dict], current: str, proposed: str, action_name: str | None,
                         visible_tags: set[str]) -> dict[str, Any]:
    violations: list[str] = []
    known_states = {s for r in rules for s in (r.get("from"), r.get("to"))}
    state_valid = not known_states or proposed in known_states
    if not state_valid:
        violations.append(f"unknown_state:{proposed}")
    if proposed == current:
        outgoing = [r for r in rules if r.get("from") == proposed]
        action_valid = True
        if action_name and outgoing:
            action_valid = any(action_name in (r.get("allowed_actions") or []) for r in outgoing)
            if not action_valid:
                violations.append(f"action_not_available_at:{proposed}")
        return {"state_valid": state_valid, "action_valid": action_valid,
                "transition_valid": state_valid, "violations": violations, "matched_rule_ids": []}
    cands = [r for r in rules if r.get("from") == current and r.get("to") == proposed]
    if not cands:
        return {"state_valid": state_valid, "action_valid": False, "transition_valid": False,
                "violations": violations + [f"no_transition:{current}->{proposed}"], "matched_rule_ids": []}
    matched: list[str] = []
    transition_valid = False
    for r in cands:
        missing = [w for w in (r.get("when") or []) if w not in visible_tags]
        if missing:
            violations.append(f"missing_preconditions:{','.join(missing)}@{r.get('from')}->{r.get('to')}")
            continue
        transition_valid = True
        matched.append(f"{r.get('from')}->{r.get('to')}")
    outgoing = [r for r in rules if r.get("from") == proposed]
    action_valid = True
    if action_name and outgoing:
        action_valid = any(action_name in (r.get("allowed_actions") or []) for r in outgoing)
        if not action_valid:
            violations.append(f"action_not_available_at:{proposed}")
    return {"state_valid": state_valid, "action_valid": action_valid,
            "transition_valid": transition_valid and state_valid,
            "violations": violations, "matched_rule_ids": matched}


def score_workflow(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Target turn keys: answer (subset: workflow_state/hypotheses/...), allowed_actions,
    forbidden_actions, required_evidence, expect_abstain. eval.rules.transitions
    supplies from/to/when/allowed_actions; rules.evidence_tags maps evidence ids
    to evaluator-only transition tags."""
    hidden = dataset.eval_config.rules if dataset.eval_config else {}
    rules = hidden.get("transitions", [])
    evidence_tags = hidden.get("evidence_tags", {})
    per_episode: dict[str, Any] = {}
    agg = {"state_field": [0, 0], "transition_valid": [0, 0], "action_correct": [0, 0],
           "abstain_correct": [0, 0], "refs_covered": [0, 0]}
    gv_count = 0
    for ep in dataset.episodes:
        rows = decisions.get(ep.id, [])
        target = dataset.targets.get(ep.id)
        turns_out: list[dict[str, Any]] = []
        current = hidden.get("initial_state") or (
            rules[0]["from"] if rules else None)
        cum = 0
        for i, row in enumerate(rows):
            entry = _base_turn_entry(row)
            t = target.turns.get(row["turn_id"]) if target else None
            d = row["decision"] or {}
            if t is None:
                entry["scored"] = False
                turns_out.append(entry)
                continue
            num = den = 0
            want_state = t.get("answer") or {}
            got = d.get("answer") if isinstance(d.get("answer"), dict) else {}
            if want_state.get("workflow_state"):
                den += 1
                num += int(got.get("workflow_state") == want_state["workflow_state"])
            if want_state.get("hypotheses"):
                got_map = {h.get("label"): h.get("status") for h in (got.get("hypotheses") or [])
                           if isinstance(h, dict)}
                for label, status in want_state["hypotheses"].items():
                    den += 1
                    num += int(got_map.get(label) == status)
            if want_state.get("clinical_stage") is not None:
                den += 1
                num += int(got.get("clinical_stage") == want_state["clinical_stage"])
            entry["state_field_accuracy"] = _rate(num, den)
            agg["state_field"][0] += num
            agg["state_field"][1] += den

            action_name = got.get("next_step")
            v = _validate_transition(rules, current or "", got.get("workflow_state") or "",
                                     action_name, _visible_tags(ep, row["turn_id"], evidence_tags))
            entry["transition_valid"] = bool(v["transition_valid"])
            agg["transition_valid"][0] += int(v["transition_valid"])
            agg["transition_valid"][1] += 1
            gvs = list(v["violations"])
            if action_name and action_name in (t.get("forbidden_actions") or []):
                gvs.append(f"forbidden_action:{action_name}")
            if t.get("expect_abstain"):
                agg["abstain_correct"][1] += 1
                ok = row["decision"] is not None and d.get("answer") is None
                entry["abstain_correct"] = ok
                agg["abstain_correct"][0] += int(ok)
            elif t.get("allowed_actions"):
                agg["action_correct"][1] += 1
                ok = action_name in t["allowed_actions"]
                entry["action_correct"] = ok
                agg["action_correct"][0] += int(ok)
            if t.get("required_evidence"):
                agg["refs_covered"][1] += 1
                ok = set(t["required_evidence"]) <= set(d.get("citations") or [])
                entry["refs_covered"] = ok
                agg["refs_covered"][0] += int(ok)
            entry["guideline_violations"] = gvs
            gv_count += len(gvs)
            turn_ok = (row["decision"] is not None and den > 0 and num == den
                       and v["transition_valid"] and not gvs
                       and (not t.get("expect_abstain") or d.get("answer") is None)
                       and (not t.get("allowed_actions") or action_name in t["allowed_actions"]))
            cum += int(turn_ok)
            entry["turn_ok"] = turn_ok
            entry["cumulative_success"] = _rate(cum, i + 1)
            entry["scored"] = True
            turns_out.append(entry)
            current = got.get("workflow_state") or current
        per_episode[ep.id] = {
            "turns": turns_out, "target_present": target is not None,
            "completion": _rate(sum(1 for r in rows if r["decision"]), len(rows)),
            "final_cumulative_success": turns_out[-1].get("cumulative_success") if turns_out else None,
        }
    return {
        "scorer": "workflow",
        "per_episode": per_episode,
        "aggregate": {k: _rate(v[0], v[1]) for k, v in agg.items()},
        "guideline_violation_count": gv_count,
    }


# ---------------------------------------------------------------- unscored

def score_unscored(dataset: Dataset, decisions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per = {}
    for ep in dataset.episodes:
        rows = decisions.get(ep.id, [])
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
        rows = decisions.get(ep.id, [])
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



REGISTRY: dict[str, Any] = {
    "exact": score_exact,
    "workflow": score_workflow,
    "unscored": score_unscored,
    "rocov2": score_rocov2,
}
