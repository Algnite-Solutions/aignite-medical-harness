"""Frozen ReXrank subset; official references bypass the raw report parser."""
from __future__ import annotations

import json
import math
import csv
import gzip
import random
import re
import shutil
from pathlib import Path

from ..data import validate_dataset
from .mimic_cxr import decode_image, digest, json_file, jsonl, safe_path, staging

REVISION = "0a18a41e580992ac7cd32a8a9463009654f0d390"
SOURCE_SHA256 = "39fe05a446cebd7af4507681cc423b04409eaf8d205ad40b6a043168a14059df"
PROTOCOL = "mimic_cxr_rexrank_radgraph_100_v1"
SCORER = "mimic_cxr_radgraph_v1"


def section_text(value) -> str:
    # The pinned official JSON contains NaN for 723 missing impressions.
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        raise ValueError("invalid official section type")
    return value


def image_relative(value: str, subject: str, study: str) -> str:
    """ReXrank paths begin with /pXX; they are NOT filesystem absolute paths."""
    pattern = rf"/p{subject[:2]}/p{subject}/s{study}/([A-Za-z0-9-]+)\.jpg"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError("image path disagrees with subject/study or is unsafe")
    return "files" + value


def load_official(path: Path) -> dict:
    if digest(path) != SOURCE_SHA256:
        raise ValueError("ReXrank source hash mismatch; use the pinned official JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def select_records(records: dict, count: int = 100, seed: int = 42) -> tuple[list[str], dict]:
    """Validate the entire official population before deterministic sampling."""
    if not isinstance(records, dict) or not 1 <= count <= len(records):
        raise ValueError("invalid population or sample size")
    all_images = set()
    for study, row in records.items():
        if not isinstance(study, str) or not study.isdigit() or row.get("split") != "test":
            raise ValueError("invalid study ID or non-test record")
        subject = str(row["subject_id"])
        if not subject.isdigit() or len(subject) < 2:
            raise ValueError("invalid subject ID")
        images, views = row["image_path"], row["frontal_lateral"]
        if not isinstance(images, list) or not images or len(images) != len(views):
            raise ValueError("invalid image/view pairing")
        for value in images:
            image_relative(value, subject, study)
            if value in all_images:
                raise ValueError("duplicate image")
            all_images.add(value)
        if row["key_image_path"] not in images:
            raise ValueError("key image is not in study")
        for key in ("report", "section_findings", "section_impression"):
            text = section_text(row.get(key))
            if key == "report" and not text.strip():
                raise ValueError(f"missing official reference field: {key}")
    selected = sorted(random.Random(seed).sample(sorted(records, key=int), count), key=int)
    return selected, {"population_studies": len(records), "population_images": len(all_images),
                      "selected_studies": count,
                      "selected_images": sum(len(records[s]["image_path"]) for s in selected),
                      "seed": seed, "excluded_studies": 0}


def freeze_cohort(source: Path, out: Path) -> dict:
    records = load_official(source)
    ids, summary = select_records(records)
    manifest = {"protocol": PROTOCOL, "source_revision": REVISION,
                "source_sha256": digest(source), "study_ids": ids, **summary,
                "input": "current-study-all-images-and-views; no context/history",
                "reference": "official report field, unchanged",
                "materialized": False}
    with staging(out) as stage:
        json_file(stage / "cohort.json", manifest)
    return manifest


def materialize(source: Path, cohort: Path, image_root: Path, out: Path) -> dict:
    """image_root is the MIMIC directory containing files/pXX/... JPGs."""
    records = load_official(source)
    ids, summary = select_records(records)
    frozen = json.loads(cohort.read_text(encoding="utf-8"))
    if (frozen.get("source_sha256") != SOURCE_SHA256 or frozen.get("source_revision") != REVISION
            or frozen.get("protocol") != PROTOCOL or frozen.get("study_ids") != ids
            or frozen.get("seed") != 42):
        raise ValueError("cohort differs from frozen v1 protocol")
    expected = {Path(p).stem: (str(records[s]["subject_id"]), s)
                for s in ids for p in records[s]["image_path"]}
    matched = set()
    split_path = safe_path(image_root, "mimic-cxr-2.0.0-split.csv.gz")
    with gzip.open(split_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            iid = row["dicom_id"]
            if iid in expected:
                if (iid in matched or (row["subject_id"], row["study_id"]) != expected[iid]
                        or row["split"] != "test"):
                    raise ValueError("official image/study/test split mismatch")
                matched.add(iid)
    if matched != set(expected):
        raise ValueError("selected images missing from official split CSV")
    episodes, targets, origins = [], [], []
    with staging(out) as stage:
        (stage / "artifacts").mkdir()
        for sid in ids:
            row = records[sid]
            eid, evidence, copied = f"mimic-cxr-s{sid}", [], []
            for value, view in sorted(zip(row["image_path"], row["frontal_lateral"])):
                rel = image_relative(value, str(row["subject_id"]), sid)
                origin = safe_path(image_root, rel)
                decode_image(origin)
                image_id = Path(rel).stem
                dest = f"artifacts/{image_id}.jpg"
                checksum = digest(origin)
                shutil.copy2(origin, stage / dest)
                if digest(stage / dest) != checksum:
                    raise ValueError("copied image hash mismatch")
                item = {"id": f"image-{image_id}", "type": "image", "file": dest}
                if isinstance(view, str) and view.strip():
                    item["text"] = f"View: {view}"
                evidence.append(item)
                copied.append({"source": rel, "file": dest, "sha256": checksum})
            episodes.append({"id": eid, "turns": [{"id": "t1", "evidence": evidence}]})
            targets.append({"id": eid, "turns": {"t1": {"answer": {
                "findings": section_text(row["section_findings"]),
                "impression": section_text(row["section_impression"])},
                "report": row["report"]}}})
            origins.append({"id": eid, "source_revision": REVISION, "images": copied})
        json_file(stage / "dataset.json", {"schema": "ama-dataset", "name": PROTOCOL,
                  "splits": {"test": [e["id"] for e in episodes]}})
        jsonl(stage / "episodes.jsonl", episodes)
        jsonl(stage / "targets.jsonl", targets)
        jsonl(stage / "provenance.jsonl", origins)
        json_file(stage / "eval.json", {"scorer": SCORER})
        json_file(stage / "cohort.json", {**frozen, "materialized": True,
                                         "cohort_sha256": digest(cohort),
                                         "split_csv_sha256": digest(split_path)})
        (stage / "instructions.txt").write_text(
            'Write an English chest X-ray report using all supplied images and views. '
            'Return answer {"findings": "...", "impression": "..."} in the required Decision. '
            'Cite supplied image Evidence IDs. Do not invent prior-study comparisons.\n', encoding="utf-8")
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("invalid dataset: " + "; ".join(errors))
    return summary
