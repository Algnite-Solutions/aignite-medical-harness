"""ama — one CLI, one loop. validate / inspect / run / eval / import."""
from __future__ import annotations

import argparse
import hashlib
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
        print(f"\n[{self.episode_id} 第{index + 1}/{total}轮 {turn.id}] {turn.observation or ''}")
        for m in visible_meta:
            print(f"  新增证据: {m['id']} ({m['type']})")
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
            print(f"  decision: answer={decision.answer!r} citations={decision.citations}")

    def wait_next(self) -> bool:
        return self._menu("next")


# ---------------------------------------------------------------- run / eval orchestration

def _run_dataset(dataset_dir: Path, model_name: str, split: str | None, episode_ids: list[str] | None,
                 runs_root: Path, interactive: bool, verbose: bool, budget_args: dict[str, Any],
                 instruction: str = "") -> Path:
    from .agent import Interaction, run_episode
    from .data import load_dataset, resolve_dataset_dir, validate_visible_files
    from .model import make_model_factory, resolve_model_config
    from .recorder import Budget, Recorder, TraceConfig

    dataset_dir = resolve_dataset_dir(dataset_dir)
    dataset = load_dataset(dataset_dir, with_targets=False)  # run NEVER opens targets.jsonl
    validate_visible_files(dataset)
    if episode_ids:
        wanted = list(dict.fromkeys(episode_ids))
        known = {e.id: e for e in dataset.episodes}
        missing = [eid for eid in wanted if eid not in known]
        if missing:
            raise KeyError(f"episodes not found: {missing}")
        episodes = [known[eid] for eid in wanted]
    else:
        episodes = dataset.select(split=split, episode_id=None)
    factory = make_model_factory(model_name, request_timeout=budget_args.get("request_timeout", 60.0))

    budget = Budget(max_model_calls=budget_args.get("max_model_calls", 60),
                    per_turn_model_calls=budget_args.get("per_turn_model_calls", 15),
                    deadline_seconds=budget_args.get("deadline_seconds", 900.0),
                    max_retries=budget_args.get("max_retries", 1))
    resolved = resolve_model_config(model_name)
    recorder = Recorder(runs_root, name=dataset.info.name, model_name=model_name,
                        dataset_dir=dataset_dir, episode_ids=[e.id for e in episodes],
                        trace=TraceConfig(), budget=budget, interactive=interactive,
                        experiment={"protocol": "direct_decision",
                                    "instruction": instruction,
                                    "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest()},
                        model_info={"alias": model_name, "type": resolved.type,
                                    "provider_model": resolved.model,
                                    "base_url": resolved.base_url,
                                    "wire": resolved.wire,
                                    "temperature": resolved.temperature,
                                    "max_tokens": resolved.max_tokens,
                                    "chat_template_kwargs": resolved.chat_template_kwargs,
                                    "context_window": resolved.context_window})

    summaries: list[dict[str, Any]] = []
    for ep in episodes:
        model = factory(ep.id)
        interaction: Interaction = Interaction()
        if interactive:
            interaction = ShellInteraction(ep.id,
                                            note_fn=lambda **kw: recorder.log("operator_note",
                                                                              episode_id=ep.id, **kw))
        announce = (lambda m: print(m, flush=True)) if verbose else None
        s = run_episode(ep, model, recorder, instruction=instruction,
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
    selected_ids = set(manifest.get("episode_ids") or [])
    if selected_ids:
        dataset.episodes = [ep for ep in dataset.episodes if ep.id in selected_ids]
    decisions: dict[str, list[dict[str, Any]]] = {}
    for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            decisions.setdefault(row["episode_id"], []).append(row)

    scorer_name = scorer_override or (dataset.eval_config.scorer if dataset.eval_config else "unscored")
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
        "eval_sha256": sha256_file(Path(manifest["dataset_dir"]) / "eval.json")
        if (Path(manifest["dataset_dir"]) / "eval.json").exists() else None,
        "scored": result,
        "operational": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")).get("operational"),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_scored_report(run_dir, manifest, result, dataset, decisions)
    _print_eval_summary(result)
    return 0


def _fmt(r: Any) -> str:
    if not isinstance(r, dict):
        return "N/A" if r is None else str(r)
    if "den" not in r:
        return json.dumps(r, ensure_ascii=False, separators=(",", ":"))
    if not r.get("den"):
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


def _write_scored_report(run_dir: Path, manifest: dict[str, Any], result: dict[str, Any],
                         dataset, decisions: dict[str, list[dict[str, Any]]]) -> None:
    """中文阅读层：展示英文原文与评分，不改变模型输入或评分内容。"""
    labels = {
        "caption_token_precision": "描述词精确率（预测词中匹配参考的比例）",
        "caption_token_recall": "描述词召回率（参考词中被预测覆盖的比例）",
        "caption_token_f1": "描述词 F1（精确率与召回率的综合）",
        "cui_precision": "概念编号精确率",
        "cui_recall": "概念编号召回率",
        "cui_f1": "概念编号 F1",
        "refs_covered": "引用覆盖率",
    }
    lines = [f"# AMA 实验报告 — {manifest['run_id']}", "",
             f"模型：{manifest['model']}；评分器：{result.get('scorer')}", "",
             "caption = 英文影像描述；cuis = UMLS 医学概念编号列表。编号本身不是诊断。",
             "本报告比较描述词和概念编号的重合程度；分数不代表临床判断正确率。"]
    if manifest["model"] == "scripted":
        lines += ["scripted 使用预写答案，只用于检查运行流程。"]
    lines += ["", "## 汇总", "", "| 指标 | 分子/分母 |", "|---|---|"]
    for key, value in (result.get("aggregate") or {}).items():
        lines.append(f"| {labels.get(key, key)} | {_fmt(value)} |")
    lines += ["", "## 逐例对照", ""]
    for ep in dataset.episodes:
        lines += [f"### {ep.id}", ""]
        rows = {row["turn_id"]: row for row in decisions.get(ep.id, [])}
        target = dataset.targets.get(ep.id)
        for turn in ep.turns:
            lines += [f"轮次：{turn.id}", ""]
            for evidence in turn.evidence:
                if evidence.file and Path(evidence.file).suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                    image = os.path.relpath(dataset.folder / evidence.file, run_dir)
                    lines += [f"![影像 {evidence.id}](<{image}>)", ""]
            row = rows.get(turn.id, {})
            decision = row.get("decision")
            if decision is None:
                lines += [f"未获得答案：{row.get('termination', '未执行')}", ""]
            else:
                lines += ["模型原文：", "", "```json",
                          json.dumps(decision["answer"], ensure_ascii=False, indent=2), "```", ""]
            reference = target.turns.get(turn.id, {}) if target else {}
            if "answer" in reference:
                lines += ["参考原文（仅评测阶段读取）：", "", "```json",
                          json.dumps(reference["answer"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["分母为 0 时显示 N/A。医学术语与三个示例的中文对照见仓库 docs/rocov2-lab-guide.zh-CN.md。", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _inspect(dataset_dir: Path, episode_id: str | None) -> int:
    from .data import load_dataset, resolve_dataset_dir

    dataset_dir = resolve_dataset_dir(dataset_dir)
    ds = load_dataset(dataset_dir, with_targets=False)
    print(json.dumps(ds.info.model_dump(by_alias=True), ensure_ascii=False, indent=2))
    print(f"episodes: {len(ds.episodes)}  splits: {list(ds.info.splits)}  "
          f"targets: {'present (evaluator-only, not shown)' if (dataset_dir / 'targets.jsonl').exists() else 'absent'}")
    for ep in ds.episodes:
        if episode_id and ep.id != episode_id:
            continue
        print(f"\n== {ep.id} ({len(ep.turns)} turns)")
        for t in ep.turns:
            print(f"  {t.id} @ {t.available_at} | evidence: {[e.id for e in t.evidence]}")
            print(f"    {(t.observation or '')[:100]}")
            for e in t.evidence:
                preview = e.text[:80].replace("\n", " ") if e.text else f"<file {e.file}>"
                print(f"      {e.id} [{e.type}]: {preview}")
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
    r.add_argument("--episode", action="append", dest="episodes",
                   help="run only this episode (repeatable)")
    r.add_argument("--interactive", action="store_true")
    r.add_argument("--runs-root", default="runs")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--instruction-file", type=Path,
                   help="task-specific model instruction, copied into the run manifest")
    r.add_argument("--max-model-calls", type=int, default=60)
    r.add_argument("--per-turn-model-calls", type=int, default=15)
    r.add_argument("--request-timeout", type=float, default=60.0,
                   help="per-request timeout in seconds (thinking/vision models may need more)")
    r.add_argument("--deadline-seconds", type=float, default=900.0)
    r.add_argument("--max-retries", type=int, default=1)

    e = sub.add_parser("eval", help="score a run (reads targets here, not during run)")
    e.add_argument("run_dir")
    e.add_argument("--scorer")

    imp = sub.add_parser("import", help="offline dataset conversion")
    imp.add_argument("kind", choices=["rocov2"])
    imp.add_argument("--source", required=True)
    imp.add_argument("--out", required=True)
    imp.add_argument("--split", default="test", help="rocov2: source split (default: test)")
    imp.add_argument("--limit", type=int, default=None, help="rocov2: maximum records after sorting")
    imp.add_argument("--id", action="append", dest="ids", help="rocov2: exact image ID (repeatable)")

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
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    if args.cmd == "run":
        try:
            instruction = args.instruction_file.read_text(encoding="utf-8") if args.instruction_file else ""
            _run_dataset(Path(args.dataset_dir), args.model, args.split, args.episodes,
                         Path(args.runs_root), args.interactive, args.verbose,
                         {"max_model_calls": args.max_model_calls,
                          "per_turn_model_calls": args.per_turn_model_calls,
                          "deadline_seconds": args.deadline_seconds,
                          "max_retries": args.max_retries,
                          "request_timeout": args.request_timeout},
                         instruction=instruction)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        return 0

    if args.cmd == "eval":
        return _eval_run(Path(args.run_dir), args.scorer)

    if args.cmd == "import":
        from .importers.rocov2 import import_rocov2
        report = import_rocov2(Path(args.source), Path(args.out), split=args.split,
                              limit=args.limit, ids=args.ids)
        print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
        print(f"\nimported -> {args.out}  next: ama validate {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
