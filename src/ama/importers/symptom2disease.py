"""Convert a local Symptom2Disease CSV into single-turn text Episodes."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from ..data import Dataset, validate_dataset
from ..dataset_card import write_dataset_cards

SCORER = "symptom2disease_v1"
LABELS = (
    "Acne", "Arthritis", "Bronchial Asthma", "Cervical spondylosis", "Chicken pox",
    "Common Cold", "Dengue", "Dimorphic Hemorrhoids", "Fungal infection",
    "Hypertension", "Impetigo", "Jaundice", "Malaria", "Migraine", "Pneumonia",
    "Psoriasis", "Typhoid", "Varicose Veins", "allergy", "diabetes", "drug reaction",
    "gastroesophageal reflux disease", "peptic ulcer disease", "urinary tract infection",
)


def _label_key(value: str) -> str:
    return " ".join(value.split()).casefold()


LABEL_BY_KEY = {_label_key(label): label for label in LABELS}
INSTRUCTIONS = (
    "Classify the patient's symptom description into exactly one of these 24 dataset labels:\n"
    + ", ".join(LABELS) + ".\n"
    "The answer must be an object with exactly one disease label in the disease field, "
    "for example {\"disease\": \"Acne\"}.\n"
)


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--row", action="append", type=int, dest="rows",
                        help="CSV record number, counting the header as row 1; repeat to select rows")


def import_from_cli(args: Namespace) -> dict[str, Any]:
    return import_symptom2disease(args.source, args.out, limit=args.limit, rows=args.rows)


def _source_csv(source: Path) -> Path:
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError(f"expected a CSV source: {source}")
        return source
    if source.is_dir():
        candidates = sorted(source.glob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"expected exactly one CSV in {source}; found {len(candidates)}")
    raise FileNotFoundError(f"Symptom2Disease source not found: {source}")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def import_symptom2disease(source: Path, out: Path, *, limit: int | None = None,
                           rows: list[int] | None = None) -> dict[str, Any]:
    """Import label/text rows without exposing labels to the run path.

    Source order and one-based CSV row numbers define stable Episode IDs. An
    existing destination is rejected so an older target cannot survive a rerun.
    """
    source_csv, out = _source_csv(Path(source)), Path(out)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if rows is not None and (not rows or any(row < 2 for row in rows)):
        raise ValueError("--row values must be CSV record numbers of at least 2")
    if rows is not None and limit is not None:
        raise ValueError("choose --row or --limit, not both")
    selected_rows = set(rows) if rows is not None else None
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")

    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    excluded = Counter()
    labels = Counter()
    seen_text: set[str] = set()
    duplicate_text = 0
    n_source_rows = 0
    n_eligible = 0
    n_selected_candidates = 0
    found_rows: set[int] = set()
    with source_csv.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"label", "text"} <= set(reader.fieldnames):
            raise ValueError(f"{source_csv}: expected label and text columns")
        for row_number, row in enumerate(reader, start=2):
            n_source_rows += 1
            if selected_rows is not None and row_number in selected_rows:
                found_rows.add(row_number)
            if None in row:
                raise ValueError(f"{source_csv}:{row_number}: malformed CSV row")
            symptom = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not symptom:
                if row_number in found_rows:
                    raise ValueError(f"{source_csv}:{row_number}: selected row has empty text")
                excluded["empty_text"] += 1
                continue
            if not label:
                if row_number in found_rows:
                    raise ValueError(f"{source_csv}:{row_number}: selected row has empty label")
                excluded["empty_label"] += 1
                continue
            if _label_key(label) not in LABEL_BY_KEY:
                raise ValueError(f"{source_csv}:{row_number}: unknown disease label {label!r}")
            label = LABEL_BY_KEY[_label_key(label)]
            n_eligible += 1
            if selected_rows is not None and row_number not in selected_rows:
                continue
            n_selected_candidates += 1
            if limit is not None and len(episodes) >= limit:
                continue
            episode_id = f"s2d-{row_number - 1:06d}"
            if symptom in seen_text:
                duplicate_text += 1
            seen_text.add(symptom)
            labels[label] += 1
            episodes.append({
                "id": episode_id,
                "turns": [{"id": "t1", "evidence": [
                    {"id": "symptoms", "type": "symptom_text", "text": symptom},
                ]}],
            })
            targets.append({"id": episode_id, "turns": {"t1": {
                "answer": {"disease": label}, "required_evidence": ["symptoms"],
            }}})
            provenance.append({"id": episode_id, "source_row": row_number})
    if selected_rows is not None and selected_rows - found_rows:
        raise ValueError(f"CSV rows not found: {sorted(selected_rows - found_rows)}")
    if not episodes:
        raise ValueError("no complete Symptom2Disease rows selected")

    report: dict[str, Any] = {
        "source_file": source_csv.name,
        "source_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "n_source_rows": n_source_rows,
        "n_eligible": n_eligible,
        "n_imported": len(episodes),
        "n_not_selected_due_row_filter": n_eligible - n_selected_candidates,
        "n_not_selected_due_limit": n_selected_candidates - len(episodes),
        "excluded": dict(sorted(excluded.items())),
        "duplicate_text_in_selected": duplicate_text,
        "label_counts_in_selected": dict(sorted(labels.items())),
        "selection": {"order": "source CSV order", "limit": limit,
                      "rows": sorted(selected_rows) if selected_rows is not None else None},
        "selected_source_rows": provenance,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ama-s2d-", dir=out.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        (stage / "dataset.json").write_text(json.dumps({
            "schema": "ama-dataset", "name": out.name,
            "splits": {"all": [episode["id"] for episode in episodes]},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _jsonl(stage / "episodes.jsonl", episodes)
        _jsonl(stage / "targets.jsonl", targets)
        _jsonl(stage / "provenance.jsonl", provenance)
        (stage / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
        (stage / "eval.json").write_text(json.dumps({"scorer": SCORER}) + "\n", encoding="utf-8")
        (stage / "import_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_dataset_cards(
            stage, name=out.name,
            source="Symptom2Disease CSV (https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)",
            license_name="Verify the terms of the specific source copy before redistribution.",
            purpose_en="Single-turn research classification of symptom text into source disease labels.",
            purpose_zh="将症状文本映射到源数据疾病标签的单轮研究分类任务。",
            construction_en="Each non-empty label/text row becomes one Episode and one Turn. Text is visible Evidence; the label is evaluator-only.",
            construction_zh="每条标签和症状均非空的记录生成一个 Episode 和一个 Turn；症状是可见 Evidence，标签仅供评测。",
            episodes=len(episodes), turns=len(episodes), evidence=len(episodes),
            scorer=SCORER, targets=len(targets),
            limitations_en="This importer defines no official split. The AMA 24-choice top-1 score uses exact source labels after case and whitespace normalization, counting missing and invalid answers as wrong. It is not an official dataset benchmark or clinical diagnosis accuracy.",
            limitations_zh="本次导入未定义官方数据划分。AMA 的 24 选 1 top-1 评分忽略大小写和多余空格后精确匹配源标签，缺失及无效回答计为错误；它并非官方数据集基准或临床诊断准确率。",
            artifacts_en="No external artifacts; symptom text is embedded in Evidence. Source row numbers are in import_report.json.",
            artifacts_zh="没有外部 artifact；症状文本直接存于 Evidence，原始行号存于 import_report.json。",
        )
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("generated dataset is invalid: " + "; ".join(errors))
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output already exists: {out}")
        stage.rename(out)
    return report


def score_symptom2disease(dataset: Dataset,
                          decisions: dict[str, list[dict[str, Any]]], *, histories=None) -> dict[str, Any]:
    """Exact top-1 match among source labels; missing or invalid choices score zero."""
    from ..scorer import _rate, expected_rows

    episode_ids = {episode.id for episode in dataset.episodes}
    unexpected = set(decisions) - episode_ids
    if unexpected:
        raise ValueError(f"unexpected decision episodes: {sorted(unexpected)}")
    per_episode: dict[str, Any] = {}
    correct = submitted = valid = answer_objects = 0
    for episode in dataset.episodes:
        if len(episode.turns) != 1:
            raise ValueError(f"{episode.id}: Symptom2Disease expects one turn")
        if len(decisions.get(episode.id, [])) > 1:
            raise ValueError(f"{episode.id}: duplicate recorded decisions")
        turn = episode.turns[0]
        target = dataset.targets.get(episode.id)
        if target is None or turn.id not in target.turns:
            raise ValueError(f"{episode.id}/{turn.id}: missing disease target")
        target_answer = target.turns[turn.id].get("answer")
        wanted = target_answer.get("disease") if isinstance(target_answer, dict) else None
        if not isinstance(wanted, str) or _label_key(wanted) not in LABEL_BY_KEY:
            raise ValueError(f"{episode.id}/{turn.id}: invalid disease target")
        if decisions.get(episode.id) and decisions[episode.id][0].get("turn_id") != turn.id:
            raise ValueError(f"{episode.id}: unexpected decision turn")
        row = expected_rows(episode, decisions)[0]
        decision = row["decision"]
        answer = decision.get("answer") if isinstance(decision, dict) else None
        is_object = isinstance(answer, dict)
        choice = answer.get("disease") if is_object and set(answer) == {"disease"} else answer if isinstance(answer, str) else None
        predicted = LABEL_BY_KEY.get(_label_key(choice)) if isinstance(choice, str) else None
        is_correct = predicted == LABEL_BY_KEY[_label_key(wanted)]
        submitted += decision is not None
        valid += predicted is not None
        correct += is_correct
        answer_objects += is_object
        per_episode[episode.id] = {
            "turn_id": turn.id, "submitted": decision is not None,
            "predicted_disease": predicted, "correct": is_correct,
            "valid_choice": predicted is not None, "answer_is_object": is_object,
            "termination": row["termination"],
        }
    total = len(dataset.episodes)
    return {"scorer": SCORER, "per_episode": per_episode, "aggregate": {
        "disease_accuracy": _rate(correct, total),
        "valid_choice": _rate(valid, total),
        "completion": _rate(submitted, total),
        "answer_object_format": _rate(answer_objects, total),
    }}
