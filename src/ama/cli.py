"""ama — one CLI, one loop. validate / inspect / run / eval / import."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------- interactive shell

class ShellInteraction:
    """Operator-supervised run over the SAME run_episode loop. Prompt: (ama) ep t1/3 >"""

    def __init__(self, episode_id: str, note_fn=None) -> None:
        self.episode_id = episode_id
        self.note_fn = note_fn
        self.operator_wait_ms = 0
        self.trace_lines: list[str] = []
        self._label = ""
        self._decision = None
        self.messages: list[Any] = []

    def _input(self, prompt: str) -> str:
        t0 = time.monotonic()
        try:
            return input(prompt).strip()
        except EOFError:
            return "quit"
        finally:
            self.operator_wait_ms += int((time.monotonic() - t0) * 1000)

    def on_turn_start(self, index, total, turn, visible_meta) -> None:
        self._label = f"(ama) {self.episode_id} t{index + 1}/{total} "
        print(f"\n[{self.episode_id} 第{index + 1}/{total}轮 {turn.turn_id}] {turn.message}")
        for m in visible_meta:
            print(f"  新增证据: {m['evidence_id']} ({m['kind']})")
        self.trace_lines = []
        self._decision = None

    def on_message(self, message) -> None:
        self.messages.append(message)

    @staticmethod
    def _show_message(message) -> None:
        role = "human" if message.role == "user" else message.role
        print(f"\n  [{role}]")
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                fn = call.get("function") or {}
                raw = fn.get("arguments") or "{}"
                try:
                    raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
                except (TypeError, json.JSONDecodeError):
                    pass
                print(f"  tool_call {fn.get('name', '?')}: {raw}")
            return
        if isinstance(message.content, list):
            for part in message.content:
                if part.get("type") == "text":
                    print(f"  {part.get('text', '')}")
                elif part.get("type") == "image_url":
                    print("  <inline image payload omitted>")
            return
        shown = message.content
        if message.role == "tool":
            try:
                shown = json.dumps(json.loads(shown), ensure_ascii=False, indent=2)
            except (TypeError, json.JSONDecodeError):
                pass
        print(f"  {shown}")

    def _show_conversation(self) -> None:
        if not self.messages:
            print("  （尚无消息）")
            return
        for message in self.messages:
            self._show_message(message)

    def _menu(self, allowed: str) -> bool:
        hint = "run" if allowed == "run" else "next"
        other = "next" if allowed == "run" else "run"
        while True:
            cmd = self._input(f"{self._label}{hint} > ")
            if cmd in ("", allowed):
                return True
            if cmd == other:
                if allowed == "run":
                    print("  本轮尚未执行：先输入 run 让模型完成当前轮。")
                else:
                    print("  本轮已完成：输入 next 进入下一轮（或 quit 结束）。")
                continue
            if cmd == "quit":
                return False
            if cmd == "show":
                self._show_conversation()
            elif cmd == "trace":
                for line in self.trace_lines:
                    print(f"  {line}")
            elif cmd.startswith("note "):
                text = cmd[5:].strip()
                if self.note_fn:
                    self.note_fn(note=text)
                print(f"  operator note 已记录: {text}")
            else:
                print(f"  未知命令: {cmd}（可用: {allowed} show trace note <text> quit）")

    def wait_run(self) -> bool:
        return self._menu("run")

    def on_step(self, line: str) -> None:
        print(f"  {line}")
        self.trace_lines.append(line)

    def on_turn_end(self, decision, note: str) -> None:
        self._decision = decision.model_dump(mode="json") if decision else None
        if decision is None:
            print(f"  本轮未获得决策（{note or '见 trace'}），按真实轨迹进入下一轮。")
        else:
            print(f"  decision: workflow_state={decision.state.get('workflow_state')} "
                  f"abstain={decision.abstain} action="
                  f"{decision.action.name if decision.action else None} "
                  f"citations={decision.citations}")

    def wait_next(self) -> bool:
        return self._menu("next")


# ---------------------------------------------------------------- run / eval orchestration

def _run_dataset(dataset_dir: Path, model_name: str, split: str | None, episode_id: str | None,
                 runs_root: Path, interactive: bool, verbose: bool, budget_args: dict[str, Any]) -> Path:
    from .agent import Interaction, run_episode
    from .data import load_dataset, resolve_dataset_dir
    from .model import make_model_factory
    from .recorder import Budget, Recorder, TraceConfig

    dataset_dir = resolve_dataset_dir(dataset_dir)
    dataset = load_dataset(dataset_dir, with_targets=False)  # run NEVER opens targets.jsonl
    episodes = dataset.select(split=split, episode_id=episode_id)
    factory = make_model_factory(model_name, request_timeout=budget_args.get("request_timeout", 60.0))

    budget = Budget(max_model_calls=budget_args.get("max_model_calls", 60),
                    per_turn_model_calls=budget_args.get("per_turn_model_calls", 15),
                    deadline_seconds=budget_args.get("deadline_seconds", 900.0),
                    max_retries=budget_args.get("max_retries", 1))
    recorder = Recorder(runs_root, name=dataset.info.name, model_name=model_name,
                        dataset_dir=dataset_dir, episode_ids=[e.episode_id for e in episodes],
                        trace=TraceConfig(), budget=budget, interactive=interactive)

    summaries: list[dict[str, Any]] = []
    for ep in episodes:
        model = factory(ep.episode_id)
        interaction: Interaction = Interaction()
        if interactive:
            interaction = ShellInteraction(ep.episode_id,
                                            note_fn=lambda **kw: recorder.log("operator_note",
                                                                              episode_id=ep.episode_id, **kw))
        announce = (lambda m: print(m, flush=True)) if verbose else None
        s = run_episode(ep, model, recorder, public_policy=dataset.policy.public,
                        interaction=interaction, announce=announce)
        s.pop("rows")
        summaries.append(s)

    operational = {
        "n_episodes": len(summaries),
        "terminations": {t: sum(1 for s in summaries if s["termination"] == t)
                         for t in {s["termination"] for s in summaries}},
        "total_model_calls": sum(s["model_calls"] for s in summaries),
    }
    recorder.finalize(summaries, operational)
    print(f"run complete: {recorder.run_dir}  next: ama eval {recorder.run_dir}")
    return recorder.run_dir


def _eval_run(run_dir: Path, scorer_override: str | None = None) -> int:
    from .data import load_dataset
    from .recorder import sha256_file
    from . import scorer as scorer_mod

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_dataset(Path(manifest["dataset_dir"]), with_targets=True)  # eval reads targets
    decisions: dict[str, list[dict[str, Any]]] = {}
    for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            decisions.setdefault(row["episode_id"], []).append(row)

    scorer_name = scorer_override or dataset.info.scorer or "unscored"
    fn = scorer_mod.REGISTRY.get(scorer_name)
    if fn is None:
        print(f"unknown scorer '{scorer_name}' (registry: {list(scorer_mod.REGISTRY)})")
        return 1
    result = fn(dataset, decisions)

    targets_path = Path(manifest["dataset_dir"]) / "targets.jsonl"
    metrics = {
        "run_id": manifest["run_id"],
        "model": manifest["model"],
        "scorer": scorer_name,
        "targets_sha256": sha256_file(targets_path) if targets_path.exists() else None,
        "policy_sha256": sha256_file(Path(manifest["dataset_dir"]) / "policy.json")
        if (Path(manifest["dataset_dir"]) / "policy.json").exists() else None,
        "scored": result,
        "operational": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")).get("operational"),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_scored_report(run_dir, manifest, result)
    _print_eval_summary(result)
    return 0


def _fmt(r: Any) -> str:
    if not isinstance(r, dict) or not r.get("den"):
        return "N/A"
    return f"{r.get('num', 0)}/{r['den']}"


def _print_eval_summary(result: dict[str, Any]) -> None:
    agg = result.get("aggregate")
    if agg:
        print("aggregate: " + "  ".join(f"{k}={_fmt(v)}" for k, v in agg.items()))
    for eid, ep in result.get("per_episode", {}).items():
        if "turns" in ep:
            decided = sum(1 for t in ep["turns"] if t.get("scored"))
            print(f"  {eid}: scored_turns={decided}/{len(ep['turns'])}")


def _write_scored_report(run_dir: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    backend_note = ("scripted 满分只证明运行器与评分器工作，不代表真实模型效果"
                    if manifest["model"] == "scripted" else
                    f"{manifest['model']} 真实模型；teacher-forced replay 评估，小样本不构成基准结论")
    lines = [f"# AMA run — {manifest['run_id']}", "",
             f"- model: **{manifest['model']}**（{backend_note}）",
             f"- scorer: {result.get('scorer')}", "",
             "## Per-episode", ""]
    for eid, ep in result.get("per_episode", {}).items():
        lines.append(f"### {eid}")
        lines.append("")
        if "turns" not in ep:
            lines.append(f"- {ep}")
            lines.append("")
            continue
        scored = [t for t in ep["turns"] if t.get("scored")]
        keys = [k for k in ("state_correct", "answer_correct", "action_correct", "refs_covered",
                            "state_field_accuracy", "transition_valid", "abstain_correct")
                if any(k in t for t in scored)]
        if keys:
            lines.append("| turn | " + " | ".join(keys) + " | violations |")
            lines.append("|---|" + "---|" * (len(keys) + 1))
            for t in scored:
                cells = []
                for k in keys:
                    v = t.get(k)
                    if isinstance(v, dict):
                        cells.append(_fmt(v))
                    elif isinstance(v, bool):
                        cells.append("✓" if v else "✗")
                    else:
                        cells.append(str(v))
                lines.append(f"| {t['turn_id']} | " + " | ".join(cells) +
                             f" | {len(t.get('guideline_violations', []))} |")
        else:
            for t in ep["turns"]:
                lines.append(f"- {t['turn_id']}: submitted={t['submitted']} term={t['turn_termination']}")
        if ep.get("final_cumulative_success"):
            lines.append(f"\ncumulative success: {_fmt(ep['final_cumulative_success'])}")
        lines.append("")
    if result.get("aggregate"):
        lines += ["## Aggregate", "",
                  "| metric | value |", "|---|---|"]
        for k, v in result["aggregate"].items():
            lines.append(f"| {k} | {_fmt(v)} |")
    lines += ["", "失败/未提交轮保留在分母；分母为 0 记 N/A。", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _inspect(dataset_dir: Path, episode_id: str | None) -> int:
    from .data import load_dataset, resolve_dataset_dir

    dataset_dir = resolve_dataset_dir(dataset_dir)
    ds = load_dataset(dataset_dir, with_targets=False)
    print(json.dumps({k: v for k, v in ds.info.model_dump().items()}, ensure_ascii=False, indent=2))
    print(f"episodes: {len(ds.episodes)}  splits: {list(ds.info.splits)}  "
          f"targets: {'present (evaluator-only, not shown)' if (dataset_dir / 'targets.jsonl').exists() else 'absent'}")
    if ds.policy.public:
        print("policy.public keys:", list(ds.policy.public))
    for ep in ds.episodes:
        if episode_id and ep.episode_id != episode_id:
            continue
        print(f"\n== {ep.episode_id} (subject {ep.subject_id}, {len(ep.turns)} turns)")
        for t in ep.turns:
            print(f"  {t.turn_id} @ {t.time} | evidence: {[e.evidence_id for e in t.evidence]}")
            print(f"    {t.message[:100]}")
            for e in t.evidence:
                preview = e.text[:80].replace("\n", " ") if e.text else f"<artifact {e.artifact}>"
                print(f"      {e.evidence_id} [{e.kind}]: {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="ama", description="aignite-medical-agent: minimal research runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="offline structural validation")
    v.add_argument("dataset_dir")

    ins = sub.add_parser("inspect", help="show dataset/episodes (never targets)")
    ins.add_argument("dataset_dir")
    ins.add_argument("--episode")

    r = sub.add_parser("run", help="run episodes (batch or --interactive)")
    r.add_argument("dataset_dir")
    r.add_argument("--model", required=True)
    r.add_argument("--split")
    r.add_argument("--episode")
    r.add_argument("--interactive", action="store_true")
    r.add_argument("--runs-root", default="runs")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--max-model-calls", type=int, default=60)
    r.add_argument("--per-turn-model-calls", type=int, default=15)
    r.add_argument("--deadline-seconds", type=float, default=900.0)
    r.add_argument("--max-retries", type=int, default=1)

    e = sub.add_parser("eval", help="score a run (reads targets here, not during run)")
    e.add_argument("run_dir")
    e.add_argument("--scorer")

    imp = sub.add_parser("import", help="offline dataset conversion")
    imp.add_argument("kind", choices=["medagentbench", "episode-folder"])
    imp.add_argument("--source", required=True)
    imp.add_argument("--out", required=True)
    imp.add_argument("--fhir-base", default=None, help="medagentbench: also build FHIR patient episodes")
    imp.add_argument("--fhir-patients", type=int, default=5)
    imp.add_argument("--name", default=None, help="episode-folder: dataset name")

    args = p.parse_args(argv)

    if args.cmd == "validate":
        from .data import resolve_dataset_dir, validate_dataset
        try:
            dataset_dir = resolve_dataset_dir(args.dataset_dir)
        except FileNotFoundError as exc:
            print(f"INVALID: {exc}")
            return 1
        errors = validate_dataset(dataset_dir)
        if errors:
            print(f"INVALID: {args.dataset_dir}")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"OK: {dataset_dir}")
        return 0

    if args.cmd == "inspect":
        try:
            return _inspect(Path(args.dataset_dir), args.episode)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1

    if args.cmd == "run":
        try:
            _run_dataset(Path(args.dataset_dir), args.model, args.split, args.episode,
                         Path(args.runs_root), args.interactive, args.verbose,
                         {"max_model_calls": args.max_model_calls,
                          "per_turn_model_calls": args.per_turn_model_calls,
                          "deadline_seconds": args.deadline_seconds,
                          "max_retries": args.max_retries})
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        return 0

    if args.cmd == "eval":
        return _eval_run(Path(args.run_dir), args.scorer)

    if args.cmd == "import":
        if args.kind == "medagentbench":
            from .importers.medagentbench import import_medagentbench
            report = import_medagentbench(Path(args.source), Path(args.out),
                                          fhir_base=args.fhir_base, fhir_patients=args.fhir_patients)
        else:
            from .importers.episode_folder import import_episode_folders
            report = import_episode_folders(Path(args.source), Path(args.out), dataset_name=args.name)
        print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
        print(f"\nimported -> {args.out}  next: ama validate {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
