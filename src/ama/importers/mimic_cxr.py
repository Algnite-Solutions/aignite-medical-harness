"""Current-study-only MIMIC-CXR onboarding, explicit-sections-v1 (not a benchmark)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..data import validate_dataset
from ..dataset_card import write_dataset_cards

PARSER = "explicit-sections-v1"
OFFICIAL_REFERENCE = "31901464793d0bb23835092f05bd1aba6af7baff"


def parse_sections(text: str) -> dict[str, str]:
    """Explicit line headings only; concatenate repeats; never infer missing sections."""
    heading = re.compile(r"^[ \t]*([A-Za-z][A-Za-z ()/,-]*?):[ \t]*(.*)$")
    sections: dict[str, list[str]] = {"findings": [], "impression": []}
    current = None
    for line in text.splitlines():
        match = heading.match(line)
        bare = line.strip().lower()
        if match or bare in sections:
            name = match[1].strip().lower() if match else bare
            current = name if name in sections else None
            if current and match:
                sections[current].append(match[2])
        elif current:
            sections[current].append(line)
    return {key: " ".join(" ".join(value).split()) for key, value in sections.items()}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_image(path: Path) -> None:
    from PIL import Image
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected JPEG: {path.name}")
        image.load()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("expected non-empty relative path")
    candidate = root / relative
    if ".." in Path(relative).parts or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe raw path: {relative}")
    # Reject links even if their destination is inside the package.
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ValueError(f"symbolic link in raw path: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing raw file: {relative}")
    return candidate


def identifier(value: str, *, numeric: bool = False) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+" if numeric else r"[A-Za-z0-9-]+", value):
        raise ValueError("invalid source identifier")
    return value


def json_file(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


@contextmanager
def staging(out: Path):
    """Build on the destination filesystem; publish only on successful exit."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mimic-cxr-", dir=out.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        yield stage
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output already exists: {out}")
        stage.rename(out)


def import_mimic_cxr(source: Path, out: Path, split: str = "validate",
                     limit: int | None = None, ids: list[str] | None = None) -> dict:
    """Read only a standalone raw package, keeping answers evaluator-only."""
    source, out = Path(source), Path(out)
    if split not in {"train", "validate", "test"} or (limit is not None and limit < 1):
        raise ValueError("invalid split or non-positive limit")
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the raw source")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    records = [json.loads(line) for line in safe_path(source, "manifest.jsonl").read_text().splitlines() if line.strip()]
    seen, image_ids = set(), set()
    for row in records:
        sid = identifier(row["study_id"], numeric=True)
        identifier(row["subject_id"], numeric=True)
        if sid in seen or row["split"] not in {"train", "validate", "test"} or not row["images"]:
            raise ValueError("duplicate study, invalid split or empty images")
        seen.add(sid)
        for image in row["images"]:
            iid = identifier(image["dicom_id"])
            if iid in image_ids:
                raise ValueError("duplicate dicom_id")
            image_ids.add(iid)
    candidates = sorted([r for r in records if r["split"] == split], key=lambda r: int(r["study_id"]))
    if ids is not None:
        wanted = {identifier(i, numeric=True) for i in ids}
        if len(wanted) != len(ids):
            raise ValueError("duplicate requested study IDs")
        if not wanted or wanted - {r["study_id"] for r in candidates}:
            raise ValueError("unknown study ID in requested split")
        candidates = [r for r in candidates if r["study_id"] in wanted]
    selected = candidates[:limit] if limit is not None else candidates
    if not selected:
        raise ValueError("no studies selected")
    episodes, targets, provenance = [], [], []
    with staging(out) as stage:
        (stage / "artifacts").mkdir()
        for row in selected:
            answer = parse_sections(safe_path(source, row["report"]).read_text(encoding="utf-8"))
            report_hash = digest(safe_path(source, row["report"]))
            if report_hash != row["report_sha256"] or not all(answer.values()):
                raise ValueError("report hash mismatch or missing findings/impression")
            eid = f"mimic-cxr-s{row['study_id']}"
            evidence, origins = [], []
            for image in sorted(row["images"], key=lambda i: i["dicom_id"]):
                path = safe_path(source, image["file"])
                decode_image(path)
                checksum = digest(path)
                if checksum != image["sha256"]:
                    raise ValueError("image hash mismatch")
                rel = f"artifacts/{image['dicom_id']}.jpg"
                shutil.copy2(path, stage / rel)
                if digest(stage / rel) != checksum:
                    raise ValueError("copied image hash mismatch")
                item = {"id": f"image-{image['dicom_id']}", "type": "image", "file": rel}
                view = image.get("view_position")
                if view:
                    item["text"] = f"ViewPosition: {view}"
                evidence.append(item)
                origins.append(dict(image))
            episodes.append({"id": eid, "turns": [{"id": "t1", "evidence": evidence}]})
            targets.append({"id": eid, "turns": {"t1": {"answer": answer}}})
            provenance.append({"id": eid, **row, "images": origins})
        report = {"protocol": PARSER, "official_parser_reference": OFFICIAL_REFERENCE,
                  "split": split, "n_source_studies": len(records), "n_split_studies": sum(r['split'] == split for r in records),
                  "n_after_id_filter": len(candidates), "n_not_selected_due_id_filter": sum(r["split"] == split for r in records)-len(candidates), "n_not_selected_due_limit": len(candidates)-len(selected),
                  "n_imported": len(selected), "n_images": sum(len(r['images']) for r in selected),
                  "selected_study_ids": [r['study_id'] for r in selected],
                  "manifest_sha256": digest(source / 'manifest.jsonl'), "scorer": "unscored"}
        json_file(stage / "dataset.json", {"schema": "ama-dataset", "name": out.name,
                  "splits": {split: [e['id'] for e in episodes]}})
        jsonl(stage / "episodes.jsonl", episodes)
        jsonl(stage / "targets.jsonl", targets)
        jsonl(stage / "provenance.jsonl", provenance)
        json_file(stage / "import_report.json", report)
        json_file(stage / "eval.json", {"scorer": "unscored"})
        (stage / "instructions.txt").write_text(
            'Write findings and impression in English from all supplied images of the current chest study.\n'
            'Use answer {"findings": "...", "impression": "..."}. Cite only supplied image Evidence IDs; '
            'citing every image is not required and does not measure clinical correctness.\n', encoding="utf-8")
        write_dataset_cards(stage, name=out.name, source="MIMIC-CXR-JPG 2.1.0; standalone raw package",
            license_name="PhysioNet credentialed data terms apply; no redistribution of real cases.",
            purpose_en="Onboarding derived current-study-only report generation task.", purpose_zh="Onboarding 派生任务：仅当前 study 的报告生成。",
            construction_en="One study, one turn, all images sorted by dicom_id; explicit findings/impression are evaluator-only.",
            construction_zh="一个 study、一轮；全部图片按 dicom_id 排序；明确 findings/impression 仅供评测。",
            episodes=len(episodes), turns=len(episodes), evidence=report['n_images'], scorer="unscored", targets=len(targets),
            limitations_en="Structural conversion only: no report quality score, no numerical onboarding acceptance, no history=1 reproduction. Parser: explicit-sections-v1, not official rules.",
            limitations_zh="仅证明转换结构；无报告质量分数，未满足最终 onboarding 数值验收，非 history=1 复现。parser 为 explicit-sections-v1，不等同官方规则。",
            artifacts_en="JPEG copies in artifacts; source locations, hashes and acquisition dates only in provenance.",
            artifacts_zh="JPEG 实际复制到 artifacts；来源、hash、采集时间仅在 provenance。")
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("generated dataset is invalid: " + "; ".join(errors))
    return report
