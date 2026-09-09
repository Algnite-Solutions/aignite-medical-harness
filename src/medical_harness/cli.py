"""CLI entry. Offline by default; real API runs require an explicit
openai_compatible config plus a key in the environment (.env supported)."""
from __future__ import annotations

import argparse
import json
import os
import time
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


class CliInteraction:
    """Operator-supervised online run: pause/show/note around the SAME run_episode loop."""

    def __init__(self, episode_id: str, note_fn=None) -> None:
        self.episode_id = episode_id
        self.note_fn = note_fn
        self.operator_wait_ms = 0
        self.trace_lines: list[str] = []
        self._turn_label = ""

    def _input(self, prompt: str) -> str:
        t0 = time.monotonic()
        try:
            return input(prompt).strip()
        except EOFError:
            return "quit"
        finally:
            self.operator_wait_ms += int((time.monotonic() - t0) * 1000)

    def on_turn_start(self, turn_index, total, turn, released_meta) -> None:
        self._turn_label = f"[{self.episode_id} 第{turn_index + 1}/{total}轮 {turn.turn_id} @ {turn.as_of.isoformat()}]"
        print(f"\n{self._turn_label} {turn.world_message or turn.event}")
        for m in released_meta:
            print(f"  新增证据: {m['evidence_id']} ({m['modality']}, recorded {m['recorded_time']})")
        self.trace_lines = []

    def _menu(self, allowed_next: str) -> bool:
        while True:
            cmd = self._input(f"({allowed_next} | state | trace | note <text> | quit) > ")
            if cmd in ("", allowed_next, "run", "next"):
                return True
            if cmd == "quit":
                return False
            if cmd == "state":
                print("  （公开状态见上方本轮决策；本 v0.2 不在此重复渲染 claim 状态）")
            elif cmd == "trace":
                for line in self.trace_lines:
                    print(f"  {line}")
            elif cmd.startswith("note "):
                text = cmd[5:].strip()
                if self.note_fn:
                    self.note_fn(episode_id=self.episode_id, note=text)
                print(f"  operator note 已记录: {text}")
            else:
                print(f"  未知命令: {cmd}")

    def wait_run(self) -> bool:
        return self._menu("run")

    def on_step(self, line: str) -> None:
        print(f"  {line}")
        self.trace_lines.append(line)

    def on_turn_end(self, decision, public_result, note: str) -> None:
        if decision is None:
            print(f"  本轮未获得决策（{note or '见 trace'}），按真实轨迹进入下一轮。")
            return
        print(f"  决策: workflow_state={decision.workflow_state} "
              f"abstain={decision.abstain} "
              f"next_action={decision.next_action.action_type if decision.next_action else None} "
              f"hypotheses={[(h.label, h.status) for h in decision.state.hypotheses]}")
        if public_result:
            print(f"  validator(公开): state={public_result['state_valid']} "
                  f"action={public_result['action_valid']} transition={public_result['transition_valid']} "
                  f"violations={public_result['violations']}")

    def wait_next(self) -> bool:
        return self._menu("next")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="medical-harness",
        description="Minimal medical agent harness: question runs + teacher-forced trajectory episodes",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="batch question run over synthetic cases")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--runs-root", default="runs")
    p_run.add_argument("--verbose", action="store_true")

    p_eval = sub.add_parser("evaluate", help="score a question run against gold")
    p_eval.add_argument("--run-dir", required=True)

    p_prof = sub.add_parser("profile", help="structural profile of evidence data")
    p_prof.add_argument("--data-dir", default="data/sim/inputs")
    p_prof.add_argument("--out")

    p_ve = sub.add_parser("validate-episode", help="offline structural validation of an episode folder")
    p_ve.add_argument("--episode-dir", required=True)

    p_re = sub.add_parser("run-episode", help="run ONE episode (batch, or --interactive operator-supervised)")
    p_re.add_argument("--episode-dir", required=True)
    p_re.add_argument("--config", required=True, help="model/budget/strategy config (episodes field ignored)")
    p_re.add_argument("--runs-root", default="runs")
    p_re.add_argument("--interactive", action="store_true")

    p_rt = sub.add_parser("run-trajectory", help="batch run over all episodes listed in config")
    p_rt.add_argument("--config", required=True)
    p_rt.add_argument("--runs-root", default="runs")
    p_rt.add_argument("--verbose", action="store_true")

    p_et = sub.add_parser("evaluate-trajectory", help="score a trajectory run against episode gold")
    p_et.add_argument("--run-dir", required=True)

    p_mb = sub.add_parser("profile-medagentbench", help="audit MedAgentBench tasks by patient (offline tier)")
    p_mb.add_argument("--data-file", default=None, help="path to test_data_v2.json")
    p_mb.add_argument("--patient-limit", type=int, default=5)
    p_mb.add_argument("--fhir-base", default="http://localhost:8080/fhir", help="FHIR base for the online tier")
    p_mb.add_argument("--out", help="optional json output path")

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

    if args.cmd == "validate-episode":
        from .trajectory import validate_episode
        errors = validate_episode(Path(args.episode_dir))
        if errors:
            print(f"INVALID: {args.episode_dir}")
            for e in errors:
                print(f"  - {e}")
            return 1
        print(f"OK: {args.episode_dir}")
        return 0

    if args.cmd in ("run-episode", "run-trajectory"):
        from .trajectory import run_trajectory
        kwargs = dict(runs_root=Path(args.runs_root))
        if args.cmd == "run-episode":
            kwargs["episode_dirs"] = [Path(args.episode_dir).resolve()]
            kwargs["interactive"] = args.interactive
        else:
            kwargs["verbose"] = args.verbose
        run_dir = run_trajectory(Path(args.config), **kwargs)
        print(f"next: python -m medical_harness.cli evaluate-trajectory --run-dir {run_dir}")
        return 0

    if args.cmd == "evaluate-trajectory":
        from .trajectory import evaluate_trajectory
        metrics = evaluate_trajectory(Path(args.run_dir))
        scored = metrics.get("scored", {})
        for eid, sc in scored.items():
            if "aggregate" in sc:
                agg = sc["aggregate"]
                print(f"{eid}: next_action={agg['next_action_accuracy']['num']}/{agg['next_action_accuracy']['den']} "
                      f"state={agg['state_field_accuracy']['num']}/{agg['state_field_accuracy']['den']} "
                      f"violations={agg['guideline_violation_count']} "
                      f"cumulative={_agg_cum(agg)}")
            else:
                print(f"{eid}: 未评分（{sc.get('reason', '?')}）")
        print(f"report: {Path(args.run_dir) / 'report.md'}")
        return 0

    if args.cmd == "profile-medagentbench":
        from .adapters.medagentbench import profile_patients
        prof = profile_patients(
            data_file=Path(args.data_file) if args.data_file else None,
            patient_limit=args.patient_limit,
            fhir_base=args.fhir_base,
        )
        text = json.dumps(prof, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"profile written: {args.out}")
        else:
            print(text)
        return 0

    return 1


def _agg_cum(agg: dict) -> str:
    fc = agg.get("final_cumulative_success")
    if not isinstance(fc, dict) or not fc.get("den"):
        return "N/A"
    return f"{fc['num']}/{fc['den']}"


if __name__ == "__main__":
    raise SystemExit(main())
