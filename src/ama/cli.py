"""CLI wiring: run / chat / probe / eval / import rocov2."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .model import Model, ModelError
from .runner import execute


def _load_dotenv(path=Path(".env")):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _eval_run(run_dir: Path, scorer_override=None):
    from .data import load_dataset, validate_dataset
    from .recorder import write_json
    from .scorer import REGISTRY, expected_rows
    from .decisions import output_metrics

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") == "probe":
        if scorer_override is not None:
            raise ValueError("ama probe has its own offline evaluator; omit --scorer")
        from .probe import evaluate
        return evaluate(run_dir)
    if manifest.get("mode") == "chat":
        raise ValueError("chat 是探索记录，不生成基准分数；请使用 ama run 后再 eval。")
    if manifest.get("mode") != "run" or "expected_turns" not in manifest:
        raise ValueError("unsupported historical run; regenerate it with ama run")
    folder = Path(manifest["dataset_dir"])
    errors = validate_dataset(folder)
    if errors:
        raise ValueError("invalid evaluation dataset: " + "; ".join(errors))
    dataset = load_dataset(folder, with_targets=True)
    dataset.episodes = [dataset.select(episode_id=e)[0] for e in manifest["episode_ids"]]
    if {ep.id: [t.id for t in ep.turns] for ep in dataset.episodes} != manifest["expected_turns"]:
        raise ValueError("expected turns do not match inference manifest")
    decisions = {}
    for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            decisions.setdefault(row["episode_id"], []).append(row)
    if manifest.get("probing_plan"):
        from .probing import evaluate_probing
        return evaluate_probing(run_dir, dataset, manifest, decisions, scorer_override)
    scorer = scorer_override or (dataset.eval_config.scorer if dataset.eval_config else "unscored")
    if scorer not in REGISTRY:
        raise ValueError(f"unknown scorer: {scorer}")
    result = REGISTRY[scorer](dataset, decisions)
    if manifest.get("output_format") == "binary-label":
        from .probing import binary_run_metrics
        result["binary_probe"] = binary_run_metrics(run_dir)
    quality = output_metrics([row for episode in dataset.episodes
                              for row in expected_rows(episode, decisions)],
                             manifest.get("output_contract_version"))
    if result.get("aggregate") is None:
        result["aggregate"] = {}
    result["aggregate"]["output_quality"] = quality
    write_json(run_dir / "metrics.json", {
        "run_id": manifest["run_id"], "model": manifest["model"], "scorer": scorer,
        "scored": result,
    })
    print(json.dumps(result.get("aggregate"), ensure_ascii=False, indent=2))
    print(f"metrics: {run_dir / 'metrics.json'}")
    return 0


def main(argv=None):
    _load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run" and any(a == "--probing-plan" or a.startswith("--probing-plan=") for a in argv):
        print("ERROR: ama run --probing-plan was removed. Use ama probe SOURCE_RUN --mode cumulative|endpoint; resume archived tasks with ama probe --resume PROBE_RUN.")
        return 1
    parser = argparse.ArgumentParser(prog="ama", description="Clinical runs, exploratory chat, frozen probes and offline evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("run", "chat"):
        command = sub.add_parser(mode)
        command.add_argument("dataset_dir")
        command.add_argument("--model", required=True, help="alias registered in ama.json")
        command.add_argument("--config", type=Path, default=Path("ama.json"))
        command.add_argument("--episode", action="append" if mode == "run" else "store",
                             required=mode == "chat")
        if mode == "run":
            command.add_argument("--split")
            command.add_argument("--output-format", choices=["decision", "text", "binary-label"], default="decision")
        command.add_argument("--runs-root", type=Path, default=Path("runs"))
        command.add_argument("--instruction-file", type=Path)
        command.add_argument("--tools", type=Path, help="trusted Python file exporting TOOLS")
        command.add_argument("--max-calls", type=int, default=8, help="model calls per input")
        command.add_argument("--timeout", type=float, default=60, help="seconds per API request")
    probe = sub.add_parser("probe", help="independent measurements of a completed clinical run")
    probe.add_argument("source_run", nargs="?", type=Path)
    probe.add_argument("--mode", choices=["cumulative", "endpoint"])
    probe.add_argument("--model")
    probe.add_argument("--config", type=Path)
    probe.add_argument("--episode", action="append")
    probe.add_argument("--kind", action="append", choices=["sufficient", "true"])
    probe.add_argument("--candidates", type=Path, help="JSON object mapping episode IDs to explicit diagnosis overrides")
    probe.add_argument("--prepare-only", action="store_true", help="freeze requests without model calls or API credentials")
    probe.add_argument("--resume", type=Path)
    probe.add_argument("--runs-root", type=Path)
    probe.add_argument("--timeout", type=float)
    probe.add_argument("--max-probe-tokens", type=int)
    probe.add_argument("--top-logprobs", type=int)
    evaluate = sub.add_parser("eval", help="separate scoring; reads references only here")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--scorer")
    importer = sub.add_parser("import")
    importer.add_argument("kind", choices=["rocov2"])
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--out", type=Path, required=True)
    importer.add_argument("--split", default="test")
    importer.add_argument("--limit", type=int)
    importer.add_argument("--id", action="append", dest="ids")
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "chat"):
            model = Model.from_config(args.model, args.config, timeout=args.timeout)
            path = execute(args.dataset_dir, model, mode=args.command,
                       episode_ids=[args.episode] if args.command == "chat" else args.episode,
                       split=getattr(args, "split", None), runs_root=args.runs_root,
                       instruction_file=args.instruction_file, tools_path=args.tools, max_calls=args.max_calls,
                       output_format=getattr(args, "output_format", "decision"))
            manifest = json.loads((path / "manifest.json").read_text())
            print(f"{args.command}: {manifest['termination']} — {path}")
            if args.command == "run":
                print(f"next: ama eval {path}")
            return 0 if manifest["termination"] in {"completed", "quit", "eof"} else 1
        if args.command == "probe":
            from .probe import prepare, execute as execute_probe, resume
            if args.resume:
                if any(getattr(args, k) is not None for k in ("source_run", "mode", "model", "config", "episode", "kind", "candidates", "runs_root", "timeout", "max_probe_tokens", "top_logprobs")) or args.prepare_only:
                    raise ValueError("--resume uses frozen settings; do not supply preparation options")
                path = resume(args.resume)
            else:
                if not args.source_run or not args.mode or not args.model:
                    raise ValueError("ama probe requires SOURCE_RUN, --mode and --model, or --resume PROBE_RUN")
                configs = json.loads((args.config or Path("ama.json")).read_text())["models"]
                if args.model not in configs:
                    raise ValueError("unknown model alias: "+args.model)
                path = prepare(args.source_run, mode=args.mode, model_name=args.model, config=configs[args.model],
                               episodes=args.episode, candidates_file=args.candidates, kinds=args.kind,
                               runs_root=args.runs_root or Path("runs"), timeout=args.timeout if args.timeout is not None else 60,
                               max_probe_tokens=args.max_probe_tokens if args.max_probe_tokens is not None else 32,
                               top_logprobs=args.top_logprobs if args.top_logprobs is not None else 20)
                print(f"prepared: {path}")
                if not args.prepare_only:
                    execute_probe(path)
            print(f"probe: {path}")
            print(f"next: ama probe --resume {path}" if args.prepare_only else f"next: ama eval {path}")
            return 0
        if args.command == "eval":
            return _eval_run(args.run_dir, args.scorer)
        from .importers.rocov2 import import_rocov2
        report = import_rocov2(args.source, args.out, split=args.split, limit=args.limit, ids=args.ids)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"imported -> {args.out}")
        return 0
    except (OSError, ValueError, KeyError, ImportError, ModelError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
