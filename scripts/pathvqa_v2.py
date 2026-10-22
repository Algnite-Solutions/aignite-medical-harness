"""Frozen PathVQA cohort verification and resumable native AMA baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from ama.data import load_dataset, validate_dataset
from ama.model import content
from ama.runner import DECISION_PROMPT, render
from ama.recorder import write_json


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identity(dataset: Path, ids: list[str]) -> dict:
    ds = load_dataset(dataset, with_targets=True)
    by_id = {ep.id: ep for ep in ds.episodes}
    if len(ids) != len(set(ids)) or any(eid not in by_id or eid not in ds.targets for eid in ids):
        raise ValueError("duplicate or missing frozen episode/target")
    selected = [by_id[eid] for eid in ids]
    images = {}
    for ep in selected:
        if len(ep.turns) != 1 or len(ep.turns[0].evidence) != 1:
            raise ValueError(f"unexpected PathVQA observation shape: {ep.id}")
        relative = ep.turns[0].evidence[0].file
        path = (dataset / relative).resolve()
        if dataset.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"unsafe or missing image: {ep.id}")
        images[relative] = digest(path)
    report = json.loads((dataset / "import_report.json").read_text(encoding="utf-8"))
    instruction = (dataset / "instructions.txt").read_text(encoding="utf-8")
    return {"source_repository": report["repository"], "source_revision": report["revision"],
            "dataset_json_sha256": digest(dataset / "dataset.json"),
            "episodes_jsonl_sha256": digest(dataset / "episodes.jsonl"),
            "targets_jsonl_sha256": digest(dataset / "targets.jsonl"),
            "eval_json_sha256": digest(dataset / "eval.json"),
            "instruction_sha256": canonical(instruction),
            "system_sha256": canonical(instruction + "\n\n" + DECISION_PROMPT),
            "selected_episodes_sha256": canonical([ep.model_dump(mode="json") for ep in selected]),
            "selected_targets_sha256": canonical([ds.targets[eid].model_dump(mode="json") for eid in ids]),
            "images": images}


def prepare(dataset: Path, old_lock: Path, new_lock: Path) -> dict:
    frozen = json.loads(old_lock.read_text(encoding="utf-8"))
    ids = frozen["episode_ids"]
    if frozen["size"] != 100 or frozen["question_types"] != {"binary": 49, "open": 51}:
        raise ValueError("unexpected historical cohort")
    ds = load_dataset(dataset, with_targets=True)
    if ds.info.splits["automedbench_test_500"] != [eid for eid in ds.info.splits["test"]
                                                    if eid in set(ds.info.splits["automedbench_test_500"])]:
        raise ValueError("frozen 500-case order changed")
    if len(ds.info.splits["automedbench_test_500"]) != 500 or not set(ids) <= set(ds.info.splits["automedbench_test_500"]):
        raise ValueError("cohort differs from frozen 500-case pool")
    lock = {"schema": "ama-pathvqa-cohort-v2", "episode_ids": ids,
            "question_types": frozen["question_types"], "identity": identity(dataset, ids)}
    if new_lock.exists():
        if json.loads(new_lock.read_text(encoding="utf-8")) != lock:
            raise ValueError("dataset differs from frozen v2 lock")
    else:
        write_json(new_lock, lock)
    return lock


def verify(dataset: Path, lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["identity"] != identity(dataset, lock["episode_ids"]):
        raise ValueError("PathVQA questions, images, targets, instructions, or dataset changed")
    return lock


def mini_dataset(dataset: Path, lock: dict, out: Path) -> Path:
    if out.exists():
        if (out / "source.lock.json").read_text(encoding="utf-8") != json.dumps(lock, indent=2) + "\n":
            raise ValueError("existing 100-case dataset has a different source lock")
        mini = load_dataset(out, with_targets=True)
        ids = lock["episode_ids"]
        if mini.info.splits != {"test": ids} or [ep.id for ep in mini.episodes] != ids or \
                canonical([ep.model_dump(mode="json") for ep in mini.episodes]) != lock["identity"]["selected_episodes_sha256"] or \
                canonical([mini.targets[eid].model_dump(mode="json") for eid in ids]) != lock["identity"]["selected_targets_sha256"] or \
                canonical((out / "instructions.txt").read_text(encoding="utf-8")) != lock["identity"]["instruction_sha256"] or \
                digest(out / "eval.json") != lock["identity"]["eval_json_sha256"]:
            raise ValueError("existing 100-case dataset changed")
        for relative, expected in lock["identity"]["images"].items():
            if digest(out / relative) != expected:
                raise ValueError(f"frozen image changed: {relative}")
        return out
    out.mkdir(parents=True)
    ids = lock["episode_ids"]
    wanted = set(ids)
    source = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    source["name"] = "pathvqa-pr100-v2"
    source["splits"] = {"test": ids}
    write_json(out / "dataset.json", source)
    for name in ("episodes.jsonl", "targets.jsonl"):
        rows = {row["id"]: row for row in lines(dataset / name) if row["id"] in wanted}
        (out / name).write_text("".join(json.dumps(rows[eid], ensure_ascii=False) + "\n" for eid in ids), encoding="utf-8")
    for name in ("eval.json", "instructions.txt"):
        shutil.copyfile(dataset / name, out / name)
    for relative in lock["identity"]["images"]:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset / relative, path)
    write_json(out / "source.lock.json", lock)
    if validate_dataset(out):
        raise ValueError("invalid frozen 100-case dataset")
    return out


def _attempts(case: Path) -> list[Path]:
    if not case.exists():
        return []
    found = sorted(case.iterdir())
    for index, path in enumerate(found, start=1):
        if not path.is_dir() or path.name != f"attempt-{index:02d}":
            raise ValueError(f"orphan attempt directory: {path}")
    return found


def _read_attempt(path: Path, eid: str, alias: str, config: dict) -> tuple[dict, Path]:
    runs = [p for p in path.iterdir() if p.is_dir()]
    if len(runs) != 1:
        raise ValueError(f"orphan or duplicate native run: {path}")
    run = runs[0]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["termination"] == "running":
        raise ValueError(f"uncertain interrupted request; inspect before resuming: {run}")
    if manifest["episode_ids"] != [eid] or manifest["model"] != alias or manifest["model_config"] != config or manifest["max_calls"] != 1:
        raise ValueError(f"attempt protocol differs: {run}")
    rows = lines(run / "decisions.jsonl")
    if len(rows) != 1 or rows[0]["episode_id"] != eid or rows[0]["turn_id"] != "t1":
        raise ValueError(f"incomplete attempt: {run}")
    return rows[0], run


def _temporary(row: dict) -> bool:
    if row["termination"] != "api_error":
        return False
    error = row.get("error") or ""
    match = re.search(r"HTTP (\d+)", error)
    return (match and (int(match.group(1)) in (408, 429) or int(match.group(1)) >= 500)) or "connection failed" in error


def _permanent_protocol_error(row: dict) -> bool:
    if row["termination"] != "api_error":
        return False
    match = re.search(r"HTTP (\d+)", row.get("error") or "")
    # A successful image smoke request establishes support for this configuration.
    # Later 400/422 responses can be case-specific; freeze and score those as failures.
    return bool(match and int(match.group(1)) in (401, 403))


def run(dataset: Path, lock_path: Path, root: Path, config_path: Path, alias: str,
        *, timeout: float = 300, interval: float = 1, max_attempts: int = 3) -> None:
    from ama.model import Model
    from ama.runner import execute

    lock = verify(dataset, lock_path)
    model = Model.from_config(alias, config_path, timeout=timeout)
    if model.config.get("temperature") != 0 or model.config.get("max_tokens") != 8192:
        raise ValueError("baseline requires temperature=0 and max_tokens=8192")
    mini = mini_dataset(dataset, lock, root / "dataset")
    batch = root / alias
    batch.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "ama-pathvqa-protocol-v2", "alias": alias,
                "model_config": model.config, "timeout": timeout, "max_calls": 1,
                "interval_seconds": interval, "max_attempts": max_attempts,
                "lock_sha256": digest(lock_path),
                "code_sha256": canonical({name: digest(Path(__file__).resolve().parents[1] / name)
                                           for name in ("scripts/pathvqa_v2.py", "src/ama/runner.py",
                                                        "src/ama/pathvqa_score.py", "src/ama/model.py")})}
    ppath = batch / "protocol.json"
    if ppath.exists():
        existing = json.loads(ppath.read_text(encoding="utf-8"))
        if {k: v for k, v in existing.items() if k not in {"code_sha256", "code_revisions"}} != \
                {k: v for k, v in protocol.items() if k != "code_sha256"}:
            raise ValueError("existing run uses a different request protocol")
        if existing["code_sha256"] != protocol["code_sha256"] and \
                protocol["code_sha256"] not in existing.get("code_revisions", []):
            existing.setdefault("code_revisions", []).append(protocol["code_sha256"])
            write_json(ppath, existing)
    else:
        write_json(ppath, protocol)
    for index, eid in enumerate(lock["episode_ids"], 1):
        case = batch / f"{index:03d}-{eid}"
        attempts = _attempts(case)
        if len(attempts) > max_attempts:
            raise ValueError(f"too many attempts: {eid}")
        for attempt in attempts:
            row, _ = _read_attempt(attempt, eid, alias, model.config)
            if not _temporary(row) or attempt != attempts[-1]:
                if attempt != attempts[-1] and not _temporary(row):
                    raise ValueError(f"answer was regenerated: {eid}")
        if attempts:
            row, _ = _read_attempt(attempts[-1], eid, alias, model.config)
            if not _temporary(row) or len(attempts) >= max_attempts:
                if _permanent_protocol_error(row):
                    raise ValueError(f"provider rejected request; stop batch: {eid}: {row['error']}")
                continue
        for number in range(len(attempts) + 1, max_attempts + 1):
            attempt_root = case / f"attempt-{number:02d}"
            attempt_root.mkdir(parents=True)
            execute(mini, model, episode_ids=[eid], runs_root=attempt_root, max_calls=1)
            row, _ = _read_attempt(attempt_root, eid, alias, model.config)
            print(f"{alias}: {index}/100 attempt {number}: {row['termination']}", flush=True)
            if _permanent_protocol_error(row):
                raise ValueError(f"provider rejected request; stop batch: {eid}: {row['error']}")
            if not _temporary(row):
                break
            if number < max_attempts:
                time.sleep(min(2 ** number, 30))
        if index < len(lock["episode_ids"]):
            time.sleep(interval)


def smoke(dataset: Path, lock_path: Path, root: Path, config_path: Path, alias: str) -> Path:
    from ama.model import Model
    from ama.runner import execute

    lock = verify(dataset, lock_path)
    model = Model.from_config(alias, config_path, timeout=300)
    if model.config.get("temperature") != 0 or model.config.get("max_tokens") != 8192:
        raise ValueError("smoke requires the complete uniform request configuration")
    mini = mini_dataset(dataset, lock, root / "dataset")
    smoke_root = root / "smoke" / alias
    if smoke_root.exists():
        raise FileExistsError(f"smoke is already recorded: {smoke_root}")
    run_dir = execute(mini, model, episode_ids=[lock["episode_ids"][0]],
                      runs_root=smoke_root, max_calls=1)
    row = lines(run_dir / "decisions.jsonl")[0]
    if row["termination"] == "api_error":
        raise ValueError(f"smoke request rejected: {row['error']}")
    return run_dir


def compare(dataset: Path, lock_path: Path, root: Path, aliases: list[str], out: Path) -> dict:
    from ama.cli import _eval_run

    lock = verify(dataset, lock_path)
    mini = mini_dataset(dataset, lock, root / "dataset")
    summaries = []
    for alias in aliases:
        batch = root / alias
        protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
        if protocol["lock_sha256"] != digest(lock_path) or protocol["alias"] != alias:
            raise ValueError(f"inconsistent cohort or alias: {alias}")
        config = protocol["model_config"]
        if protocol["timeout"] != 300 or protocol["max_calls"] != 1 or \
                protocol["interval_seconds"] != 1 or protocol["max_attempts"] != 3 or \
                config.get("temperature") != 0 or config.get("max_tokens") != 8192:
            raise ValueError(f"nonuniform request protocol: {alias}")
        runs, rows, events, histories, attempt_ledger = [], [], [], {}, []
        api_statuses = Counter()
        for index, eid in enumerate(lock["episode_ids"], 1):
            case = batch / f"{index:03d}-{eid}"
            attempts = _attempts(case)
            if not attempts:
                raise ValueError(f"missing case: {eid}")
            for number, attempt in enumerate(attempts, start=1):
                row, native = _read_attempt(attempt, eid, alias, protocol["model_config"])
                native_manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
                if native_manifest["timeout"] != protocol["timeout"] or native_manifest["tools"] is not None:
                    raise ValueError(f"native request protocol changed: {native}")
                runs.append(native)
                match = re.search(r"HTTP (\d+)", row.get("error") or "")
                if row["termination"] == "api_error":
                    api_statuses[match.group(1) if match else "transport_or_other"] += 1
                attempt_ledger.append({"episode_id": eid, "attempt": number,
                                       "termination": row["termination"],
                                       "http_status": int(match.group(1)) if match else None,
                                       "retryable": bool(_temporary(row)),
                                       "configured_wait_after_seconds": 2 ** number if _temporary(row) and number < len(attempts) else 0,
                                       "run_id": native.name,
                                       "run_files_sha256": {name: digest(native / name) for name in
                                                           ("manifest.json", "messages.json", "decisions.jsonl", "diagnostics.jsonl")}})
            if len(attempts) > protocol["max_attempts"]:
                raise ValueError(f"too many attempts: {eid}")
            if any(not _temporary(_read_attempt(a, eid, alias, protocol["model_config"])[0]) for a in attempts[:-1]):
                raise ValueError(f"regenerated a received answer: {eid}")
            row, native = _read_attempt(attempts[-1], eid, alias, protocol["model_config"])
            if _permanent_protocol_error(row):
                raise ValueError(f"provider configuration failure: {eid}")
            rows.append(row)
            history = json.loads((native / "messages.json").read_text(encoding="utf-8"))[eid]
            system = (mini / "instructions.txt").read_text(encoding="utf-8") + "\n\n" + DECISION_PROMPT
            if not history or history[0] != {"role": "system", "content": system}:
                raise ValueError(f"system prompt changed: {eid}")
            ep = load_dataset(mini).episode(eid)
            expected = render(ep.turns[0], mini)
            expected.append('Final Decision turn_id: t1. Currently citeable evidence IDs: ["image"]')
            if len(history) < 2 or history[1].get("content") != content(expected):
                raise ValueError(f"question or image request changed: {eid}")
            histories[eid] = history
            events.extend(lines(native / "diagnostics.jsonl"))
        aggregate = batch / "aggregate"
        aggregate.mkdir(exist_ok=True)
        first = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
        if first["tools"] is not None or first["timeout"] != protocol["timeout"]:
            raise ValueError(f"native run used tools or a different timeout: {alias}")
        first.update(run_id=f"pathvqa-v2-{alias}", dataset_dir=str(mini.resolve()),
                     episode_ids=lock["episode_ids"],
                     expected_turns={eid: ["t1"] for eid in lock["episode_ids"]},
                     termination="completed_with_errors" if any(r["termination"] != "completed" for r in rows) else "completed")
        write_json(aggregate / "manifest.json", first)
        write_json(aggregate / "messages.json", histories)
        for name, records in (("decisions.jsonl", rows), ("diagnostics.jsonl", events)):
            (aggregate / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in records), encoding="utf-8")
        (aggregate / "attempts.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in attempt_ledger), encoding="utf-8")
        _eval_run(aggregate, "pathvqa_v1")
        metrics = json.loads((aggregate / "metrics.json").read_text(encoding="utf-8"))
        usage = Counter()
        for row in rows:
            if isinstance(row.get("usage"), dict):
                usage.update(row["usage"])
        summaries.append({"alias": alias, "model_config": protocol["model_config"],
                          "code_sha256": protocol["code_sha256"],
                          "code_revisions": protocol.get("code_revisions", []),
                          "timeout": protocol["timeout"], "n_attempts": len(runs),
                          "n_retries": len(runs) - len(rows),
                          "termination_counts": dict(Counter(r["termination"] for r in rows)),
                          "api_error_statuses": dict(api_statuses),
                          "elapsed_wall_seconds": round(max((p / "decisions.jsonl").stat().st_mtime for p in runs) -
                                                        min((p / "manifest.json").stat().st_mtime for p in runs), 3),
                          "duration_ms": sum(r.get("duration_ms", 0) for r in rows),
                          "reported_usage": dict(usage),
                          "aggregate": metrics["scored"]["aggregate"],
                          "files": {name: digest(aggregate / name) for name in
                                    ("manifest.json", "messages.json", "decisions.jsonl", "diagnostics.jsonl",
                                     "attempts.jsonl", "metrics.json")}})
    result = {"schema": "ama-pathvqa-results-v2", "cohort_lock_sha256": digest(lock_path),
              "official_automedbench_score": False, "runs": summaries}
    write_json(out, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "verify", "smoke", "run", "eval", "compare"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--old-lock", type=Path, default=Path("datasets/pathvqa/pr100.lock.json"))
    p.add_argument("--lock", type=Path, default=Path("datasets/pathvqa/pr100.v2.lock.json"))
    p.add_argument("--root", type=Path, default=Path("runs/pathvqa-pr100-v2"))
    p.add_argument("--config", type=Path, default=Path("datasets/pathvqa/models.v2.json"))
    p.add_argument("--alias", action="append", default=[])
    p.add_argument("--out", type=Path, default=Path("datasets/pathvqa/baseline.v2.json"))
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.old_lock, args.lock)
    elif args.command == "verify":
        verify(args.dataset, args.lock)
    elif args.command == "run":
        if len(args.alias) != 1:
            p.error("run requires exactly one --alias")
        run(args.dataset, args.lock, args.root, args.config, args.alias[0])
    elif args.command == "smoke":
        if len(args.alias) != 1:
            p.error("smoke requires exactly one --alias")
        print(smoke(args.dataset, args.lock, args.root, args.config, args.alias[0]))
    else:
        if not args.alias:
            p.error("eval and compare require --alias for every included run")
        compare(args.dataset, args.lock, args.root, args.alias, args.out)


if __name__ == "__main__":
    main()
