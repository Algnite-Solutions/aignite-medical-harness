#!/usr/bin/env python3
"""Read-only MIMIC-CXR inventory; write aggregate counts, never report text.

CSV joins use source identifiers. Image existence checks are a bounded sample,
not a full integrity check. ZIP inspection reads member names, not report bodies.
Only Python's standard library is required; no AMA schema dependency.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def audit(source: Path, sample_per_split: int) -> dict:
    split_name = "mimic-cxr-2.0.0-split.csv.gz"
    metadata_name = "mimic-cxr-2.0.0-metadata.csv.gz"
    label_names = ["mimic-cxr-2.0.0-chexpert.csv.gz",
                   "mimic-cxr-2.0.0-negbio.csv.gz",
                   "mimic-cxr-2.1.0-test-set-labeled.csv"]
    required = [split_name, metadata_name, *label_names, "mimic-cxr-reports.zip"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"required files missing: {missing}")

    images = {}
    study_splits = defaultdict(set)
    subjects = defaultdict(set)
    studies = defaultdict(set)
    image_counts = Counter()
    samples = defaultdict(list)
    duplicate_images = 0
    for row in rows(source / split_name):
        image, subject, study, split = (row[key] for key in
                                       ("dicom_id", "subject_id", "study_id", "split"))
        if split not in {"train", "validate", "test"}:
            raise ValueError("unexpected split value")
        if image in images:
            duplicate_images += 1
            if images[image] != (subject, study, split):
                raise ValueError("conflicting duplicate image identifier in split CSV")
        images[image] = (subject, study, split)
        study_splits[study].add(split)
        subjects[split].add(subject)
        studies[split].add(study)
        image_counts[split] += 1
        if len(samples[split]) < sample_per_split:
            samples[split].append((image, subject, study))

    metadata_ids = set()
    metadata_counts = Counter()
    view_counts = Counter()
    for row in rows(source / metadata_name):
        image = row["dicom_id"]
        metadata_counts["rows"] += 1
        metadata_counts["duplicate_image_ids"] += int(image in metadata_ids)
        metadata_ids.add(image)
        source_image = images.get(image)
        if source_image is None:
            metadata_counts["images_not_in_split"] += 1
        elif source_image[:2] != (row["subject_id"], row["study_id"]):
            metadata_counts["identifier_mismatches"] += 1
        view_counts[row.get("ViewPosition") or "missing"] += 1

    labels = {}
    for name in label_names:
        seen = set()
        counts = Counter()
        matched_splits = Counter()
        label_values = Counter()
        columns = []
        for row in rows(source / name):
            columns = list(row)
            study = row["study_id"]
            counts["rows"] += 1
            counts["duplicate_study_ids"] += int(study in seen)
            seen.add(study)
            matching = study_splits.get(study, set())
            group = next(iter(matching)) if len(matching) == 1 else (
                "unmatched" if not matching else "conflicting_split")
            matched_splits[group] += 1
            label_values.update(value for key, value in row.items()
                                if key not in {"subject_id", "study_id"})
        labels[name] = {**counts, "unique_studies": len(seen),
                        "columns": columns, "split_rows": dict(matched_splits),
                        "label_values": dict(label_values)}

    report_studies = set()
    report_counts = Counter()
    pattern = re.compile(r"(?:^|/)s(\d+)\.txt$")
    with zipfile.ZipFile(source / "mimic-cxr-reports.zip") as archive:
        report_counts["zip_members"] = len(archive.infolist())
        for entry in archive.infolist():
            match = pattern.search(entry.filename)
            if match:
                study = match.group(1)
                report_counts["report_entries"] += 1
                report_counts["duplicate_study_ids"] += int(study in report_studies)
                report_studies.add(study)

    sampled_images = {}
    for split, sample in samples.items():
        present = 0
        for image, subject, study in sample:
            path = source / "files" / f"p{subject[:2]}" / f"p{subject}" / f"s{study}" / f"{image}.jpg"
            present += int(path.is_file())
        sampled_images[split] = {"checked": len(sample), "present": present,
                                 "missing": len(sample) - present}

    return {
        "audit_version": 1,
        "source_directory": str(source.resolve()),
        "scope": {"csvs": "all rows", "reports": "ZIP member names only",
                  "images": "first N split-CSV rows per split; existence only",
                  "sample_per_split": sample_per_split,
                  "not_verified": ["full image inventory", "image decoding",
                                   "report contents and ZIP CRC", "published source checksums"]},
        "source_sha256": {name: digest(source / name) for name in
                          [split_name, metadata_name, *label_names]},
        "splits": {split: {"image_rows": image_counts[split],
                            "studies": len(studies[split]), "subjects": len(subjects[split]),
                            "studies_missing_report": len(studies[split] - report_studies)}
                   for split in sorted(image_counts)},
        "subject_overlap": {f"{a}/{b}": len(subjects[a] & subjects[b]) for a, b in
                            [("train", "validate"), ("train", "test"), ("validate", "test")]},
        "split_checks": {"duplicate_image_ids": duplicate_images,
                         "studies_in_multiple_splits": sum(len(s) > 1 for s in study_splits.values())},
        "metadata": {**metadata_counts, "unique_images": len(metadata_ids),
                     "split_images_missing_metadata": len(images.keys() - metadata_ids),
                     "views": dict(view_counts)},
        "labels": labels,
        "reports": {**report_counts, "unique_studies": len(report_studies),
                    "studies_not_in_split": len(report_studies - study_splits.keys())},
        "sampled_images": sampled_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-per-split", type=int, default=20)
    args = parser.parse_args()
    if args.sample_per_split < 1:
        parser.error("--sample-per-split must be positive")
    if args.out.resolve().is_relative_to(args.source.resolve()):
        parser.error("write the audit outside the source dataset")
    if args.out.exists():
        parser.error("output exists; choose a new path to preserve the earlier audit")
    result = audit(args.source, args.sample_per_split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Audit saved: {args.out.resolve()}")
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
