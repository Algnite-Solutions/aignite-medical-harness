"""Build paired, open-answer MIMIC-CDM benchmark views from processed data.

The source cohort still contains only four pathologies. The model never sees
their names in the task prompt; the evaluator retains the source labels.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from ..data import load_dataset, validate_dataset
from ..dataset_card import write_dataset_cards
from .mimic_cdm import PATHOLOGIES, SOURCE_URL, _write_json, _write_jsonl

VARIANTS = ("hpi", "interactive", "full_info")
TASK = ("Review this de-identified abdominal case for research evaluation. "
        "Give the single most likely diagnosis in free text in answer.diagnosis. "
        "Do not include a list of alternatives in that field. This is not a live clinical decision.\n")
INSTRUCTIONS = {
    "hpi": TASK + "Only the presenting history is available. Answer from that evidence.\n",
    "interactive": TASK + ("Start with the presenting history. Request examination, laboratory "
                           "results, and imaging reports as needed before answering.\n"),
    "full_info": TASK + "The available case evidence is provided together.\n",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_benchmark_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value.get("episode_ids")
    if value.get("protocol") != "mimic_cdm_open_v1" or not isinstance(ids, list) or \
            not all(isinstance(case_id, str) and case_id for case_id in ids):
        raise ValueError("invalid benchmark exclusion file")
    return ids


def _target_label(target: dict) -> str:
    return target["turns"]["t1"]["answer"]["diagnosis"]


def _select_ids(targets: list[dict], ids: list[str] | None, per_class: int,
                seed: int, excluded: set[str]) -> list[str]:
    by_id = {row["id"]: row for row in targets}
    if len(by_id) != len(targets):
        raise ValueError("duplicate source target IDs")
    if ids is not None:
        selected = list(dict.fromkeys(ids))
        if not selected or set(selected) - by_id.keys():
            raise ValueError("unknown or empty requested admission IDs")
        if set(selected) & excluded:
            raise ValueError("requested admissions overlap excluded benchmark")
        return selected
    if per_class < 1:
        raise ValueError("per_class must be positive")
    groups = {label: [] for label in PATHOLOGIES}
    for row in targets:
        if row["id"] not in excluded:
            groups[_target_label(row)].append(row["id"])
    rng = random.Random(seed)
    selected = []
    for label, candidates in groups.items():
        if len(candidates) < per_class:
            raise ValueError(f"{label}: only {len(candidates)} cases; need {per_class}")
        selected.extend(rng.sample(sorted(candidates), per_class))
    rng.shuffle(selected)
    return selected


def build_benchmark(full_dir: Path, interactive_dir: Path, out: Path, *,
                    ids: list[str] | None = None, per_class: int = 25,
                    seed: int = 42, exclude: set[str] | None = None) -> dict:
    """Write matched HPI, tool, and full-evidence views with identical targets."""
    full_dir, interactive_dir, out = Path(full_dir), Path(interactive_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"benchmark output already exists: {out}")
    for folder in (full_dir, interactive_dir):
        problems = validate_dataset(folder)
        if problems:
            raise ValueError(f"invalid source dataset {folder}: {problems}")
    full = load_dataset(full_dir, with_targets=True)
    interactive = load_dataset(interactive_dir, with_targets=True)
    full_episodes = {ep.id: ep.model_dump(mode="json", exclude_none=True) for ep in full.episodes}
    tool_episodes = {ep.id: ep.model_dump(mode="json", exclude_none=True) for ep in interactive.episodes}
    if set(full_episodes) != set(tool_episodes) or set(full.targets) != set(interactive.targets):
        raise ValueError("source variants do not contain the same cases and targets")
    full_targets = _rows(full_dir / "targets.jsonl")
    tool_targets = {row["id"]: row for row in _rows(interactive_dir / "targets.jsonl")}
    if any(row != tool_targets[row["id"]] for row in full_targets):
        raise ValueError("source variants have different evaluator targets")
    excluded = exclude or set()
    selected = _select_ids(full_targets, ids, per_class, seed, excluded)
    # The HPI must match, and the tool case must belong to the same admission.
    cases = {row["hadm_id"]: row for row in _rows(interactive_dir / "cases.jsonl")}
    if set(selected) - cases.keys():
        raise ValueError("tool case data missing selected admissions")
    for case_id in selected:
        initial = tool_episodes[case_id]["turns"][0]["evidence"]
        if len(initial) != 1 or initial[0]["id"] != "hpi" or \
                initial[0]["text"] != cases[case_id]["hpi"] or \
                initial[0] != full_episodes[case_id]["turns"][0]["evidence"][0]:
            raise ValueError(f"HPI mismatch or unsupported turn structure: {case_id}")
    out.mkdir(parents=True)
    try:
        target_by_id = {row["id"]: row for row in full_targets}
        for variant in VARIANTS:
            folder = out / f"mimic_cdm_open_{variant}"
            folder.mkdir()
            _write_json(folder / "dataset.json", {"schema": "ama-dataset", "name": folder.name,
                                                   "splits": {"all": selected}})
            source = full_episodes if variant == "full_info" else tool_episodes
            episodes = [source[case_id] for case_id in selected]
            _write_jsonl(folder / "episodes.jsonl", episodes)
            _write_jsonl(folder / "targets.jsonl", [target_by_id[case_id] for case_id in selected])
            _write_json(folder / "eval.json", {"scorer": "mimic_cdm_open"})
            (folder / "instructions.txt").write_text(INSTRUCTIONS[variant], encoding="utf-8")
            if variant == "interactive":
                _write_jsonl(folder / "cases.jsonl", [cases[case_id] for case_id in selected])
                shutil.copyfile(interactive_dir / "lab_mapping.json", folder / "lab_mapping.json")
                _write_json(folder / "tool_data_files.json", ["cases.jsonl", "lab_mapping.json"])
            shutil.copyfile(full_dir / "LICENSE.txt", folder / "LICENSE.txt")
            write_dataset_cards(
                folder, name=folder.name, source=SOURCE_URL,
                license_name="PhysioNet Credentialed Health Data License 1.5.0",
                purpose_en="Paired open-answer abdominal diagnosis research benchmark.",
                purpose_zh="配对的开放作答腹部疾病诊断研究基准。",
                construction_en="One matched admission in each evidence view; evaluator targets retain source labels.",
                construction_zh="每个证据视图包含相同住院病例，评估目标保留源数据标签。",
                episodes=len(selected), turns=len(selected),
                evidence=sum(len(ep["turns"][0]["evidence"]) for ep in episodes),
                scorer="mimic_cdm_open", targets=len(selected),
                limitations_en=("The source cohort contains only four pathologies. Unmapped or ambiguous "
                                "free-text diagnoses need blinded clinical review. Examination text may "
                                "reflect later care; no treatment quality is scored."),
                limitations_zh=("源数据仅有四种疾病。未映射或含糊的自由文本诊断需要盲态临床复核。"
                                "查体文本可能反映后续诊疗；不评分治疗质量。"),
            )
            problems = validate_dataset(folder)
            if problems:
                raise ValueError(f"invalid generated dataset: {problems}")
        _write_json(out / "benchmark.json", {
            "protocol": "mimic_cdm_open_v1", "seed": seed if ids is None else None,
            "sampling": "balanced_random" if ids is None else "explicit_ids",
            "per_class": per_class if ids is None else None, "episode_ids": selected,
            "excluded_admissions": len(excluded),
            "variants": list(VARIANTS),
        })
    except BaseException:
        shutil.rmtree(out)
        raise
    return {"cases": len(selected), "output": str(out), "variants": list(VARIANTS)}


# These are conservative lexical equivalents, not an open-world clinical judge.
_PATTERNS = {
    "appendicitis": r"\bappendicitis\b|\bappendiceal inflammation\b",
    "cholecystitis": r"\bcholecystitis\b|\binflam(?:mation|ed) of the gallbladder\b",
    "diverticulitis": r"\bdiverticulitis\b|\bdiverticular inflammation\b",
    "pancreatitis": r"\bpancreatitis\b|\bpancreatic inflammation\b",
}


def _diagnosis_label(value: object) -> tuple[str | None, str]:
    if not isinstance(value, str) or not value.strip():
        return None, "missing"
    matches = []
    for label, pattern in _PATTERNS.items():
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            nearby = value[max(0, match.start() - 45):match.start()].lower()
            after = value[match.end():match.end() + 20].lower()
            if re.search(r"\b(?:no|not|without|unlikely|exclude|excluded|history of|"
                         r"rule out|ruled out)\b(?:\W+\w+){0,3}\W*$", nearby) or \
                    re.search(r"^\W*(?:is\W+)?(?:ruled out|unlikely|excluded)\b", after):
                return None, "ambiguous"
            matches.append(label)
    if len(matches) != 1:
        return None, "ambiguous" if matches else "unmapped"
    return matches[0], "mapped"


def score_mimic_cdm_open(dataset, decisions: dict[str, list[dict]]) -> dict:
    """Conservative automatic top-1 score; expose answers needing blinded review."""
    from ..scorer import expected_rows

    per_episode = {}
    correct = mapped = submitted = answer_objects = 0
    review = []
    per_pathology = {label: {"correct": 0, "total": 0} for label in PATHOLOGIES}
    for episode in dataset.episodes:
        target = dataset.targets.get(episode.id)
        wanted = target.turns["t1"]["answer"]["diagnosis"] if target else None
        if wanted not in PATHOLOGIES:
            raise ValueError(f"missing or invalid pathology target: {episode.id}")
        row = expected_rows(episode, decisions)[0]
        decision = row.get("decision") or {}
        answer = decision.get("answer")
        value = answer.get("diagnosis") if isinstance(answer, dict) else answer
        predicted, mapping = _diagnosis_label(value)
        is_correct = predicted == wanted
        correct += is_correct
        mapped += mapping == "mapped"
        submitted += row.get("decision") is not None
        answer_objects += isinstance(answer, dict)
        per_pathology[wanted]["total"] += 1
        per_pathology[wanted]["correct"] += is_correct
        if mapping in {"ambiguous", "unmapped"}:
            review.append(episode.id)
        per_episode[episode.id] = {"submitted": row.get("decision") is not None,
                                  "answer_is_object": isinstance(answer, dict),
                                  "diagnosis_text": value, "predicted_pathology": predicted,
                                  "mapping": mapping, "reference_pathology": wanted,
                                  "correct_auto": is_correct, "termination": row["termination"]}
    total = len(dataset.episodes)
    rate = lambda n: {"num": n, "den": total, "value": n / total if total else None}
    return {"scorer": "mimic_cdm_open", "per_episode": per_episode,
            "aggregate": {"diagnosis_accuracy_auto": rate(correct),
                          "diagnosis_mapped": rate(mapped), "completion": rate(submitted),
                          "answer_object_format": rate(answer_objects),
                          "needs_review": len(review), "review_episode_ids": review,
                          "per_pathology": per_pathology}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create matched open-answer MIMIC-CDM views")
    parser.add_argument("--full-info", type=Path, required=True)
    parser.add_argument("--interactive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--exclude-benchmark", type=Path,
                        help="existing benchmark.json whose admissions must be excluded")
    args = parser.parse_args(argv)
    excluded = set(_read_benchmark_ids(args.exclude_benchmark)) if args.exclude_benchmark else set()
    print(json.dumps(build_benchmark(args.full_info, args.interactive, args.out,
                                     ids=args.ids, per_class=args.per_class,
                                     seed=args.seed, exclude=excluded), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
