"""Rule-based scoring against gold files. Gold is read ONLY here (evaluator side).

All rates carry explicit numerator/denominator; denominator 0 => value None
(reported as N/A). Failed or budget-exceeded questions stay in denominators.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .runner import load_gold, operational_metrics

_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?")


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    # narrow tolerance: numeric vs numeric-with-unit string ("1.2 mg/dL" == 1.2)
    if isinstance(a, (int, float)) and isinstance(b, str):
        m = _NUM_RE.match(b)
        return m is not None and math.isclose(a, float(m.group()), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(b, (int, float)) and isinstance(a, str):
        m = _NUM_RE.match(a)
        return m is not None and math.isclose(b, float(m.group()), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"num": num, "den": den, "value": (num / den) if den else None}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    gold_dir = Path(resolved["data_dir"]).parent / "gold"
    gold = load_gold(gold_dir) if gold_dir.exists() else {}
    rows = _load_jsonl(run_dir / "answers.jsonl")
    events = _load_jsonl(run_dir / "events.jsonl")

    violations = {"future": 0, "other_patient": 0}
    for ev in events:
        v = ev.get("violation")
        if v:
            violations[v] = violations.get(v, 0) + 1

    per_case: dict[str, dict[str, int]] = {}
    scored = {
        "n_questions": 0,
        "n_answerable": 0,
        "n_should_abstain": 0,
        "field_correct": {"num": 0, "den": 0, "value": None},
        "correct_abstain": {"num": 0, "den": 0, "value": None},
        "coverage_answered": {"num": 0, "den": 0, "value": None},
        "valid_refs": {"num": 0, "den": 0, "value": None},
        "stale_refs": {"num": 0, "den": 0, "value": None},
    }
    failures: list[dict[str, str]] = []

    for row in rows:
        case_id, qid = row["case_id"], row["question_id"]
        case_gold = gold.get(case_id)
        g = next((a for a in case_gold.answers if a.question_id == qid), None) if case_gold else None
        if g is None:
            continue  # no gold for this question: excluded from scored metrics (kept in operational)
        scored["n_questions"] += 1
        c = per_case.setdefault(case_id, {
            "n": 0, "answerable": 0, "should_abstain": 0, "correct": 0, "abstain_ok": 0,
            "answered": 0, "refs_valid": 0, "refs_den": 0, "stale": 0, "stale_den": 0,
        })
        c["n"] += 1

        completed = row["termination"] == "completed"
        ans = row.get("answer") or {}
        abstained = bool(ans.get("abstain"))
        refs = list(ans.get("evidence_refs") or [])
        for cl in ans.get("claims") or []:
            refs.extend(cl.get("evidence_refs") or [])

        if not completed:
            failures.append({"case_id": case_id, "question_id": qid, "termination": row["termination"], "detail": row.get("termination_detail", "")})

        if g.expected.abstain:
            scored["n_should_abstain"] += 1
            scored["correct_abstain"]["den"] += 1
            c["should_abstain"] += 1
            if completed and abstained:
                scored["correct_abstain"]["num"] += 1
                c["abstain_ok"] += 1
        else:
            scored["n_answerable"] += 1
            scored["field_correct"]["den"] += 1
            scored["coverage_answered"]["den"] += 1
            c["answerable"] += 1
            claim = None
            if g.expected.key is not None:
                claim = next((cl for cl in ans.get("claims") or [] if cl.get("key") == g.expected.key), None)
            elif ans.get("claims"):
                claim = ans["claims"][0]
            value_ok = (
                completed and not abstained and claim is not None
                and _values_equal(claim.get("value"), g.expected.value)
            )
            if value_ok:
                scored["field_correct"]["num"] += 1
                c["correct"] += 1
            if completed and not abstained:
                scored["coverage_answered"]["num"] += 1
                c["answered"] += 1

        if completed and refs:
            scored["valid_refs"]["den"] += 1
            c["refs_den"] += 1
            if all(r in g.allowed_refs for r in refs):
                scored["valid_refs"]["num"] += 1
                c["refs_valid"] += 1
            if g.stale_refs:
                scored["stale_refs"]["den"] += 1
                c["stale_den"] += 1
                if any(r in g.stale_refs for r in refs):
                    scored["stale_refs"]["num"] += 1
                    c["stale"] += 1

    for key in ("field_correct", "correct_abstain", "coverage_answered", "valid_refs", "stale_refs"):
        r = scored[key]
        r["value"] = (r["num"] / r["den"]) if r["den"] else None

    result = {
        "backend": resolved["model"]["type"],
        "gold_dir": str(gold_dir),
        "operational": operational_metrics(rows),
        "scored": scored,
        "violations": violations,
        "per_case": [{"case_id": k, **v} for k, v in sorted(per_case.items())],
        "failures": failures,
    }
    return result


def _fmt(r: dict[str, Any]) -> str:
    return f"{r['num']}/{r['den']}" if r["den"] else "N/A"


def write_scored_report(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if result["backend"] == "scripted":
        backend_note = "scripted 校准运行：高分只证明运行器与评分器工作，不代表真实模型效果"
    else:
        backend_note = (
            f"{result['backend']} 真实模型运行（{resolved['model'].get('model', '?')}）；"
            "数据为合成校准集且样本极小，不构成任何基准结论或临床验证"
        )
    s = result["scored"]
    lines = [
        f"# Run report — {resolved['name']}",
        "",
        f"- run_dir: `{run_dir}`",
        f"- backend: **{result['backend']}**（{backend_note}）",
        f"- gold_dir: `{result['gold_dir']}`",
        "",
        "## 运行概览",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| questions (operational) | {result['operational']['n_questions']} |",
        f"| terminations | {json.dumps(result['operational']['terminations'], ensure_ascii=False)} |",
        f"| total steps / model calls | {result['operational']['total_steps']} / {result['operational']['total_model_calls']} |",
        f"| usage | {result['operational']['usage']} |",
        "",
        "## 金标准评分（规则判分，合成校准数据）",
        "",
        "| 指标 | 分子/分母 | 值 |",
        "|---|---|---|",
        f"| 字段正确率 field_correct | {_fmt(s['field_correct'])} | {s['field_correct']['value'] if s['field_correct']['value'] is not None else 'N/A'} |",
        f"| 应弃答正确率 correct_abstain | {_fmt(s['correct_abstain'])} | {s['correct_abstain']['value'] if s['correct_abstain']['value'] is not None else 'N/A'} |",
        f"| 可回答覆盖率 coverage_answered | {_fmt(s['coverage_answered'])} | {s['coverage_answered']['value'] if s['coverage_answered']['value'] is not None else 'N/A'} |",
        f"| 有效引用率 valid_refs | {_fmt(s['valid_refs'])} | {s['valid_refs']['value'] if s['valid_refs']['value'] is not None else 'N/A'} |",
        f"| 过期引用率 stale_refs | {_fmt(s['stale_refs'])} | {s['stale_refs']['value'] if s['stale_refs']['value'] is not None else 'N/A'} |",
        f"| 患者/时间越界次数 violations | future={result['violations'].get('future', 0)}, other_patient={result['violations'].get('other_patient', 0)} | |",
        "",
        "### 按病例",
        "",
        "| case | 题数 | 答对/可答 | 弃答正确/应弃答 | 有效引用 | 过期引用 |",
        "|---|---|---|---|---|---|",
    ]
    for c in result["per_case"]:
        lines.append(
            f"| {c['case_id']} | {c['n']} | {c['correct']}/{c['answerable']} | {c['abstain_ok']}/{c['should_abstain']} | "
            f"{c['refs_valid']}/{c['refs_den']} | {c['stale']}/{c['stale_den']} |"
        )
    if result["failures"]:
        lines += ["", "### 失败/未完成任务（保留在分母中）", ""]
        for f in result["failures"]:
            lines.append(f"- {f['case_id']}/{f['question_id']}: {f['termination']} {f['detail']}")
    lines += [
        "",
        "## 口径说明",
        "",
        "- 分母为 0 的指标记为 N/A；失败、超预算、模型错误的任务保留在分母。",
        "- scripted backend 的高分仅验证运行器与评分器机制，不构成对任何真实模型的评价。",
        "- 区分 source_locator 存在与内容支持结论：valid_refs 只验证引用属于 gold 允许集合，不声称语义忠实度。",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(run_dir: Path) -> dict[str, Any]:
    result = score_run(run_dir)
    metrics = dict(result)
    (Path(run_dir) / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_scored_report(Path(run_dir), result)
    return result
