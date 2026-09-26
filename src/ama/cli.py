"""CLI wiring: run / chat / eval / import rocov2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model import Model
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
    from .recorder import sha256_file, write_json
    from .scorer import REGISTRY

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") == "chat":
        raise ValueError("chat 是探索记录，不生成基准分数；请使用 ama run 后再 eval。")
    if manifest.get("mode") != "run" or "expected_turns" not in manifest:
        raise ValueError("unsupported historical run; regenerate it with ama run")
    folder = Path(manifest["dataset_dir"])
    for name, digest in manifest["dataset_sha256"].items():
        if sha256_file(folder / name) != digest:
            raise ValueError(f"dataset changed since inference: {name}")
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
    scorer = scorer_override or (dataset.eval_config.scorer if dataset.eval_config else "unscored")
    if scorer not in REGISTRY:
        raise ValueError(f"unknown scorer: {scorer}")
    result = REGISTRY[scorer](dataset, decisions)
    write_json(run_dir / "metrics.json", {
        "run_id": manifest["run_id"], "model": manifest["model"], "scorer": scorer,
        "targets_sha256": sha256_file(folder / "targets.jsonl") if (folder / "targets.jsonl").exists() else None,
        "eval_sha256": sha256_file(folder / "eval.json") if (folder / "eval.json").exists() else None,
        "scored": result,
    })
    print(json.dumps(result.get("aggregate"), ensure_ascii=False, indent=2))
    print(f"metrics: {run_dir / 'metrics.json'}")
    return 0


def main(argv=None):
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="ama", description="One Agent, dataset runs and exploratory chat")
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
        command.add_argument("--runs-root", type=Path, default=Path("runs"))
        command.add_argument("--instruction-file", type=Path)
        command.add_argument("--tools", type=Path, help="trusted Python file exporting TOOLS")
        command.add_argument("--max-calls", type=int, default=8, help="model calls per input")
        command.add_argument("--timeout", type=float, default=60, help="seconds per API request")
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
                           instruction_file=args.instruction_file, tools_path=args.tools, max_calls=args.max_calls)
            manifest = json.loads((path / "manifest.json").read_text())
            print(f"{args.command}: {manifest['termination']} — {path}")
            if args.command == "run":
                print(f"next: ama eval {path}")
            return 0 if manifest["termination"] in {"completed", "quit", "eof"} else 1
        if args.command == "eval":
            return _eval_run(args.run_dir, args.scorer)
        from .importers.rocov2 import import_rocov2
        report = import_rocov2(args.source, args.out, split=args.split, limit=args.limit, ids=args.ids)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"imported -> {args.out}")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
