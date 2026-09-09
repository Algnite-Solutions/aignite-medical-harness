"""CLI entry: run / evaluate / profile. Offline by default; real API runs
require an explicit openai_compatible config plus a key in the environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader (no dependency): KEY=VALUE lines, never overrides real env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="medical-harness",
        description="Minimal medical agent harness (research instrument; v0 offline scripted runs)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute a config over synthetic cases")
    p_run.add_argument("--config", required=True, help="path to config json (e.g. configs/mock.json)")
    p_run.add_argument("--runs-root", default="runs", help="directory for run outputs (default ./runs)")
    p_run.add_argument("--verbose", action="store_true", help="print per-step trace")

    p_eval = sub.add_parser("evaluate", help="score a run directory against gold")
    p_eval.add_argument("--run-dir", required=True)

    p_prof = sub.add_parser("profile", help="structural profile of evidence data (observation instrument)")
    p_prof.add_argument("--data-dir", default="data/sim/inputs")
    p_prof.add_argument("--out", help="optional path to write profile json")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .runner import run

        run_dir = run(Path(args.config), Path(args.runs_root), verbose=args.verbose)
        print(f"next: python -m medical_harness.cli evaluate --run-dir {run_dir}")
        return 0

    if args.cmd == "evaluate":
        from .evaluation import evaluate

        result = evaluate(Path(args.run_dir))
        s = result["scored"]
        print(f"backend={result['backend']}  questions={s['n_questions']}  "
              f"field_correct={s['field_correct']['num']}/{s['field_correct']['den']}  "
              f"correct_abstain={s['correct_abstain']['num']}/{s['correct_abstain']['den']}  "
              f"violations={result['violations']}")
        print(f"report: {Path(args.run_dir) / 'report.md'}")
        return 0

    if args.cmd == "profile":
        from .profile import profile_dir

        prof = profile_dir(Path(args.data_dir))
        text = json.dumps(prof, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"profile written: {args.out}")
        else:
            print(text)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
