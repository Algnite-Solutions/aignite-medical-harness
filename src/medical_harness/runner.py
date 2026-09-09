"""Runner: executes a config over synthetic cases and writes a complete run directory.

Artifacts per run: resolved_config.json, events.jsonl, state_snapshots.jsonl,
answers.jsonl, metrics.json, report.md. Gold files are never read here —
scoring happens only in evaluation.py via `cli evaluate`.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .agent import QuestionResult, run_question
from .environment import TimelineEnvironment, summarize_state
from .model import make_model_factory
from .schemas import CaseInput, CaseGold, RunConfig, StateSnapshot

SRC_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- loading

def load_inputs(data_dir: Path) -> list[CaseInput]:
    cases = []
    for path in sorted(data_dir.glob("*.json")):
        cases.append(CaseInput.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    if not cases:
        raise FileNotFoundError(f"no case inputs found in {data_dir}")
    return cases


def load_gold(gold_dir: Path) -> dict[str, CaseGold]:
    gold: dict[str, CaseGold] = {}
    for path in sorted(gold_dir.glob("*.json")):
        g = CaseGold.model_validate(json.loads(path.read_text(encoding="utf-8")))
        gold[g.case_id] = g
    return gold


# ---------------------------------------------------------------- versioning

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_version() -> dict[str, str]:
    """Source hash (repo is not git-managed; see README)."""
    h = hashlib.sha256()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return {"source_sha256": h.hexdigest()[:16], "vcs": "not-a-git-repo"}


# ---------------------------------------------------------------- run dir

class EventLog:
    def __init__(self, fh: TextIO) -> None:
        self._fh = fh
        self.seq = 0

    def __call__(self, type_: str, **fields: Any) -> None:
        self.seq += 1
        row = {
            "seq": self.seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type_,
            **fields,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()


def _jsonl_writer(fh: TextIO) -> Any:
    def write(row: dict[str, Any]) -> None:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    return write


def operational_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    terms: dict[str, int] = {}
    for r in rows:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    usage_rows = [r["usage"] for r in rows if isinstance(r.get("usage"), dict)]
    if rows and len(usage_rows) == len(rows):
        usage: Any = {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_rows),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_rows),
            "total_tokens": sum(u.get("total_tokens", 0) for u in usage_rows),
        }
    elif usage_rows:
        usage = {
            "known_rows": len(usage_rows), "total_rows": len(rows),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_rows),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_rows),
            "total_tokens": sum(u.get("total_tokens", 0) for u in usage_rows),
        }
    else:
        usage = "unknown"
    return {
        "n_questions": len(rows),
        "terminations": terms,
        "total_steps": sum(r["steps"] for r in rows),
        "total_model_calls": sum(r["model_calls"] for r in rows),
        "abstain_submitted": sum(1 for r in rows if r.get("answer") and r["answer"].get("abstain")),
        "usage": usage,  # token counts when the backend reports them; costs NOT computed (no price config)
    }


def _fmt_rate(num: int, den: int) -> str:
    return f"{num}/{den}" if den else "N/A (denominator 0)"


def write_operational_report(run_dir: Path, cfg: RunConfig, ops: dict[str, Any]) -> None:
    if cfg.model.type == "scripted":
        backend_note = "scripted 校准运行：高分只证明运行器与评分器工作，不代表真实模型效果"
    else:
        backend_note = f"{cfg.model.type} 真实模型运行（{cfg.model.model or '?'}）；尚未评分，运行 evaluate 获取金标准评分"
    lines = [
        f"# Run report — {cfg.name}",
        "",
        f"- run_dir: `{run_dir}`",
        f"- backend: **{cfg.model.type}**（{backend_note}）",
        f"- strategy: {cfg.strategy.name} (state_tools={cfg.strategy.state_tools_enabled})",
        f"- started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 运行概览（未评分；运行 `evaluate --run-dir` 获得金标准评分）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| questions | {ops['n_questions']} |",
        f"| terminations | {json.dumps(ops['terminations'], ensure_ascii=False)} |",
        f"| total steps / model calls | {ops['total_steps']} / {ops['total_model_calls']} |",
        f"| abstain submitted | {ops['abstain_submitted']} |",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main entry

def run(config_path: Path, runs_root: Path, verbose: bool = False) -> Path:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = RunConfig.model_validate(raw)

    base = Path(config_path).resolve().parent
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = (base / data_dir).resolve()
    script_dir = Path(cfg.model.script_dir) if cfg.model.script_dir else None
    if script_dir is not None and not script_dir.is_absolute():
        script_dir = (base / script_dir).resolve()

    cases = load_inputs(data_dir)
    if cfg.cases is not None:
        by_id = {c.case_id: c for c in cases}
        missing = [cid for cid in cfg.cases if cid not in by_id]
        if missing:
            raise ValueError(f"cases not found in {data_dir}: {missing}")
        cases = [by_id[cid] for cid in cfg.cases]

    model_factory = make_model_factory(
        cfg.model,
        script_dir,
        state_tools_enabled=cfg.strategy.state_tools_enabled,
        request_timeout=cfg.budget.request_timeout_seconds,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root) / f"{stamp}-{cfg.name}-{secrets.token_hex(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    input_files = sorted(data_dir.glob("*.json"))
    resolved = {
        "name": cfg.name,
        "config_path": str(Path(config_path).resolve()),
        "data_dir": str(data_dir),
        "script_dir": str(script_dir) if script_dir else None,
        "model": cfg.model.model_dump(),
        "strategy": cfg.strategy.model_dump(mode="json"),
        "budget": cfg.budget.model_dump(mode="json"),
        "cases": [c.case_id for c in cases],
        "code_version": code_version(),
        "input_hashes": {p.name: _sha256(p) for p in input_files},
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []

    with open(run_dir / "events.jsonl", "w", encoding="utf-8") as ev_fh, \
            open(run_dir / "state_snapshots.jsonl", "w", encoding="utf-8") as snap_fh, \
            open(run_dir / "answers.jsonl", "w", encoding="utf-8") as ans_fh:
        log = EventLog(ev_fh)
        write_snapshot = _jsonl_writer(snap_fh)
        write_answer = _jsonl_writer(ans_fh)

        log("run_start", run_dir=str(run_dir), name=cfg.name, cases=resolved["cases"])

        for case in cases:
            log("case_start", case_id=case.case_id, patient_id=case.task.patient_id)
            last_snapshot: StateSnapshot | None = None
            for tp in case.task.timepoints:
                log("timepoint_start", case_id=case.case_id, as_of=tp.as_of.isoformat())
                env = TimelineEnvironment(
                    case,
                    as_of=tp.as_of,
                    allow_state_tools=cfg.strategy.state_tools_enabled,
                    initial_state=last_snapshot,
                )
                n0 = len(env.snapshots)
                for q in tp.questions:
                    prefix = f"[{case.case_id}/{q.question_id} @ {tp.as_of.isoformat()}]"
                    announce = (lambda msg, p=prefix: print(f"{p} {msg}", flush=True)) if verbose else None

                    def qlog(type_: str, _c=case.case_id, _q=q.question_id, _a=tp.as_of.isoformat(), **fields: Any) -> None:
                        log(type_, case_id=_c, question_id=_q, as_of=_a, **fields)

                    prior = None
                    if cfg.strategy.inject_prior_state and env.snapshots[-1].claims:
                        prior = summarize_state(env.snapshots[-1])

                    model = model_factory(case.case_id, q.question_id)
                    res: QuestionResult = run_question(
                        env, model, q, cfg.budget, qlog, cfg.strategy,
                        announce=announce, prior_state_summary=prior,
                    )
                    row = {
                        "case_id": case.case_id,
                        "question_id": q.question_id,
                        "as_of": tp.as_of.isoformat(),
                        "termination": res.termination,
                        "termination_detail": res.termination_detail,
                        "steps": res.steps,
                        "model_calls": res.model_calls,
                        "duration_ms": res.duration_ms,
                        "usage": res.usage,
                        "answer": res.answer.model_dump(mode="json") if res.answer else None,
                    }
                    write_answer(row)
                    rows.append(row)
                    if announce:
                        announce(f"TERMINATION {res.termination}" + (f" ({res.termination_detail})" if res.termination_detail else ""))

                for snap in env.snapshots[n0:]:
                    write_snapshot(snap.model_dump(mode="json"))
                last_snapshot = env.snapshots[-1] if cfg.strategy.state_tools_enabled else None
            log("case_end", case_id=case.case_id, violations=dict(env.violations))

        ops = operational_metrics(rows)
        metrics = {"backend": cfg.model.type, "operational": ops}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log("run_end", operational=ops)
        write_operational_report(run_dir, cfg, ops)

    print(f"run complete: {run_dir} ({ops['n_questions']} questions, terminations={ops['terminations']})")
    return run_dir
