"""Bounded image copying from MIMIC source into a standalone local raw package."""
from __future__ import annotations

import csv
import gzip
import hashlib
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from .mimic_cxr import (PARSER, decode_image, digest, identifier, json_file, jsonl,
                        parse_sections, staging)


def csv_rows(path: Path, required: set[str]):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"missing columns in {path.name}")
        for row in reader:
            if None in row:
                raise ValueError(f"malformed CSV: {path.name}")
            yield row


def prepare_raw(source: Path, out: Path, split: str = "validate", limit: int = 3,
                study_ids: list[str] | None = None) -> dict:
    """Join all CSV IDs strictly; read only requested-split reports and selected images.

    Automatic selection reserves the earliest single- and multi-image eligible
    studies, then fills by numeric study ID. Image failure aborts rather than
    replacing a selected case. Report exclusions retain local provenance.
    """
    source, out = Path(source), Path(out)
    if split not in {"train", "validate", "test"} or limit < 1:
        raise ValueError("invalid split or non-positive limit")
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the source")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    wanted = None
    if study_ids is not None:
        wanted = {identifier(s, numeric=True) for s in study_ids}
        if len(wanted) != len(study_ids) or not wanted:
            raise ValueError("duplicate or empty requested study IDs")
        if len(wanted) > limit:
            raise ValueError("limit is smaller than the explicit study ID count")
    split_path = source / "mimic-cxr-2.0.0-split.csv.gz"
    meta_path = source / "mimic-cxr-2.0.0-metadata.csv.gz"
    images, groups, study_owners = {}, {}, {}
    for row in csv_rows(split_path, {"subject_id", "study_id", "dicom_id", "split"}):
        iid = identifier(row['dicom_id'])
        sid = identifier(row['study_id'], numeric=True)
        subject = identifier(row['subject_id'], numeric=True)
        partition = row['split']
        if partition not in {"train", "validate", "test"} or iid in images:
            raise ValueError("invalid split or duplicate dicom_id in split CSV")
        if sid in study_owners and study_owners[sid] != (subject, partition):
            raise ValueError("study subject/split conflict")
        study_owners[sid] = (subject, partition)
        images[iid] = (subject, sid, partition)
        if partition == split:
            groups.setdefault(sid, {"study_id": sid, "subject_id": subject, "split": split, "images": []})['images'].append({"dicom_id": iid})
    if wanted is not None and wanted - groups.keys():
        raise ValueError("unknown study ID in requested split")
    metadata_seen = set()
    metadata = {}
    for row in csv_rows(meta_path, {"subject_id", "study_id", "dicom_id", "ViewPosition"}):
        iid = row['dicom_id']
        if iid in metadata_seen or iid not in images or images[iid][:2] != (row['subject_id'], row['study_id']):
            raise ValueError("duplicate metadata ID or incorrect metadata association")
        metadata_seen.add(iid)
        if images[iid][2] == split:
            item = {}
            view = (row.get('ViewPosition') or '').strip()
            if view:
                item['view_position'] = view
            for key in ('StudyDate', 'StudyTime'):
                value = (row.get(key) or '').strip()
                if value:
                    item[key] = value
            metadata[iid] = item
    if images.keys() - metadata_seen:
        raise ValueError("split images missing metadata")
    excluded, excluded_origins, eligible = Counter(), [], []
    filtered_groups = [g for sid, g in groups.items() if wanted is None or sid in wanted]
    with zipfile.ZipFile(source / 'mimic-cxr-reports.zip') as archive:
        index = {}
        for entry in archive.infolist():
            # Exact expected path suffix associates report with subject and study.
            match = re.search(r'(?:^|/)(files/p\d+/p(\d+)/s(\d+)\.txt)$', entry.filename)
            if match:
                subject, sid = match[2], match[3]
                if sid in index:
                    raise ValueError("duplicate study report in ZIP")
                index[sid] = (subject, entry.filename)
        for group in sorted(filtered_groups, key=lambda g: int(g['study_id'])):
            sid = group['study_id']
            reason = None
            location = index.get(sid)
            if location is None:
                reason = 'missing_report'
            elif location[0] != group['subject_id']:
                raise ValueError("report subject/study association mismatch")
            else:
                body = archive.read(location[1])  # ZIP read verifies selected/report CRC
                answer = parse_sections(body.decode('utf-8'))
                missing = [key for key, value in answer.items() if not value]
                if missing:
                    reason = 'missing_' + '_and_'.join(missing)
            if reason:
                if wanted is not None:
                    raise ValueError(f"explicitly selected study lacks material: {reason}")
                excluded[reason] += 1
                excluded_origins.append({"study_id": sid, "subject_id": group['subject_id'],
                                         "report_member": location[1] if location else None, "reason": reason})
            else:
                eligible.append({**group, 'report_member': location[1]})
        if not eligible:
            raise ValueError("no eligible studies")
        chosen = []
        if wanted is None and limit >= 2:
            for single in (True, False):
                first = next((g for g in eligible if (len(g['images']) == 1) == single), None)
                if first:
                    chosen.append(first)
        for group in eligible:
            if len(chosen) >= limit:
                break
            if group not in chosen:
                chosen.append(group)
        chosen = sorted(chosen[:limit], key=lambda g: int(g['study_id']))
        manifest = []
        with staging(out) as stage:
            (stage / 'images').mkdir()
            (stage / 'reports').mkdir()
            for group in chosen:
                sid, subject = group['study_id'], group['subject_id']
                body = archive.read(group['report_member'])
                rel_report = f'reports/s{sid}.txt'
                (stage / rel_report).write_bytes(body)
                checksum = hashlib.sha256(body).hexdigest()
                if digest(stage / rel_report) != checksum:
                    raise ValueError("report copy hash mismatch")
                copied = []
                for image in sorted(group['images'], key=lambda i: i['dicom_id']):
                    iid = image['dicom_id']
                    rel_source = f'files/p{subject[:2]}/p{subject}/s{sid}/{iid}.jpg'
                    origin = source / rel_source
                    decode_image(origin)
                    checksum_image = digest(origin)
                    relative = f'images/{iid}.jpg'
                    shutil.copy2(origin, stage / relative)
                    decode_image(stage / relative)
                    if digest(stage / relative) != checksum_image:
                        raise ValueError("image copy hash mismatch")
                    copied.append({'dicom_id': iid, 'file': relative, 'source_file': rel_source,
                                   'sha256': checksum_image, **metadata[iid]})
                manifest.append({'study_id': sid, 'subject_id': subject, 'split': split,
                                 'images': copied, 'report': rel_report, 'report_member': group['report_member'],
                                 'report_sha256': checksum})
            report = {'protocol': PARSER, 'split': split, 'n_source_images': len(images),
                      'n_split_studies': len(groups), 'n_after_id_filter': len(filtered_groups),
                      'n_not_selected_due_id_filter': len(groups)-len(filtered_groups),
                      'n_eligible': len(eligible), 'excluded': dict(excluded),
                      'n_not_selected_due_limit': len(eligible)-len(chosen),
                      'n_selected': len(chosen), 'n_images': sum(len(g['images']) for g in chosen),
                      'n_single_image_studies': sum(len(g['images']) == 1 for g in chosen),
                      'n_multi_image_studies': sum(len(g['images']) > 1 for g in chosen),
                      'selection_rule': 'earliest eligible single + multi then numeric study order; explicit IDs use numeric order',
                      'selected_study_ids': [g['study_id'] for g in chosen],
                      'source_sha256': {p.name: digest(p) for p in (split_path, meta_path)},
                      'excluded_provenance': excluded_origins,
                      'verification': 'every selected JPG decoded with Pillow load; source/copy SHA256 equal; report ZIP CRC and copy SHA256 checked'}
            jsonl(stage / 'manifest.jsonl', manifest)
            json_file(stage / 'selection_report.json', report)
    return report
