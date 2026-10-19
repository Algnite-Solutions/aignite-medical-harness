"""MIMIC-CXR tasks in native AMA format; references stay evaluator-only."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..data import load_dataset, validate_dataset
from ..dataset_card import write_dataset_cards
from ..cxr_report import SCORER as REPORT_SCORER

PARSER = "explicit-sections-v1"
OFFICIAL_REFERENCE = "31901464793d0bb23835092f05bd1aba6af7baff"
TASK_INSTRUCTIONS = (
    'Write an English chest X-ray report using all supplied images and views of the current study.\n'
    'Put the report in answer as {"findings": "...", "impression": "..."}; both values must be strings.\n'
    'Do not invent prior-study comparisons.\n'
)
FINDINGS_INSTRUCTIONS = (
    'Write the English findings section of a chest X-ray report using all supplied images '
    'and views of the current study.\n'
    'Put the report in answer as {"findings": "..."}, with a non-empty string value.\n'
    'Describe visible findings only. Do not add an impression section, clinical history, '
    'or unsupported prior-study comparisons.\n'
)
AUTOMEDBENCH_TASK = {
    "benchmark": "AutoMedBench", "paper": "https://arxiv.org/abs/2606.01961",
    "task_id": "mimic-cxr-report-task",
    "task_definition_revision": "a4e64e5502c728cee744edce2b596000fcd60fdd",
    "reference_text_mode": "findings_only",
    "adaptation": "AMA Decision.answer.findings; native AMA execution and evaluation",
    "official_cohort_reproduced": False,
}


def report_fields(report_mode: str) -> tuple[str, ...]:
    if report_mode == "findings-only":
        return ("findings",)
    if report_mode == "findings-impression":
        return ("findings", "impression")
    raise ValueError("unknown report mode")


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
                     limit: int | None = None, ids: list[str] | None = None, *,
                     report_mode: str = "findings-impression") -> dict:
    """Read only a standalone raw package, keeping answers evaluator-only."""
    source, out = Path(source), Path(out)
    required = report_fields(report_mode)
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
            sections = parse_sections(safe_path(source, row["report"]).read_text(encoding="utf-8"))
            answer = {key: sections[key] for key in required}
            report_hash = digest(safe_path(source, row["report"]))
            if report_hash != row["report_sha256"] or not all(answer.values()):
                raise ValueError("report hash mismatch or missing required report sections")
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
                origins.append({**image, "file": rel, "raw_file": image["file"]})
            episodes.append({"id": eid, "turns": [{"id": "t1", "evidence": evidence}]})
            targets.append({"id": eid, "turns": {"t1": {"answer": answer}}})
            provenance.append({"id": eid, **row, "images": origins})
        report = {"protocol": PARSER, "report_mode": report_mode, "source_format": "raw",
                  "official_parser_reference": OFFICIAL_REFERENCE,
                  "split": split, "n_source_studies": len(records), "n_split_studies": sum(r['split'] == split for r in records),
                  "n_after_id_filter": len(candidates), "n_not_selected_due_id_filter": sum(r["split"] == split for r in records)-len(candidates), "n_not_selected_due_limit": len(candidates)-len(selected),
                  "n_imported": len(selected), "n_images": sum(len(r['images']) for r in selected),
                  "selected_study_ids": [r['study_id'] for r in selected],
                  "manifest_sha256": digest(source / 'manifest.jsonl'), "scorer": "unscored"}
        _write_dataset(stage, out.name, split, episodes, targets, provenance, report)
    return report


def _write_dataset(stage, name, split, episodes, targets, provenance, report):
    """Shared publication boundary for raw imports and reuse of frozen AMA cases."""
    findings_only = report["report_mode"] == "findings-only"
    if findings_only:
        report["task"] = dict(AUTOMEDBENCH_TASK)
    json_file(stage / "dataset.json", {"schema": "ama-dataset", "name": name,
              "splits": {split: [e["id"] for e in episodes]}})
    jsonl(stage / "episodes.jsonl", episodes)
    jsonl(stage / "targets.jsonl", targets)
    jsonl(stage / "provenance.jsonl", provenance)
    scorer = REPORT_SCORER if findings_only else "unscored"
    report["scorer"] = scorer
    json_file(stage / "import_report.json", report)
    json_file(stage / "eval.json", {"scorer": scorer})
    (stage / "instructions.txt").write_text(
        FINDINGS_INSTRUCTIONS if findings_only else TASK_INSTRUCTIONS, encoding="utf-8")
    write_dataset_cards(stage, name=name, source="MIMIC-CXR; " + report["source_format"],
        license_name="PhysioNet credentialed data terms apply; no redistribution of real cases.",
        purpose_en=("AutoMedBench findings-generation task adapted to AMA." if findings_only else
                    "Current-study findings and impression generation."),
        purpose_zh=("AutoMedBench findings 生成任务接入 AMA。" if findings_only else "当前 study 的 findings/impression 生成。"),
        construction_en="One study, one turn, all supplied JPEG views. Only requested report sections enter evaluator-only targets.",
        construction_zh="一个 study、一轮，包含全部 JPEG 视角；要求的报告字段只写入评测 targets。",
        episodes=len(episodes), turns=len(episodes), evidence=report["n_images"],
        scorer=scorer, targets=len(targets),
        limitations_en="Findings-only uses AutoMedBench's lightweight report metrics (rule observation F1 and ROUGE-L), not the paper's default seven-metric backend. Invalid cases score zero in AMA case means; complete-submission scores also apply upstream's all-cases-valid gate. These are not official leaderboard results or verified clinical accuracy. Source selection and hashes are in import_report.json.",
        limitations_zh="Findings-only 使用 AutoMedBench 轻量评分（规则观察项 F1 与 ROUGE-L），非论文默认七指标。AMA 逐例均值将无效病例计零；整批提交分数另遵循全部病例有效的门槛。结果不等同官方榜单或经验证的临床正确率。抽样来源与 hash 见 import_report.json。",
        artifacts_en="Verified JPEG copies in artifacts; source metadata stays in local provenance.",
        artifacts_zh="校验后的 JPEG 复制到 artifacts；来源元数据保留在本地 provenance。")
    errors = validate_dataset(stage)
    if errors:
        raise ValueError("generated dataset is invalid: " + "; ".join(errors))


def reuse_mimic_cxr(source: Path, out: Path, split: str = "test",
                   limit: int | None = None, *, report_mode: str = "findings-only",
                   ids: list[str] | None = None) -> dict:
    """Reuse a frozen CXR cohort, preserving IDs, views and structured sections.

    This deliberately does not reparse a flattened full report, resample cases,
    carry forward the old scorer, or copy any old instructions/observations.
    Every image must have a matching source provenance hash.
    """
    source, out = Path(source), Path(out)
    required = report_fields(report_mode)
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the source dataset")
    if limit is not None and limit < 1:
        raise ValueError("non-positive limit")
    dataset = load_dataset(source, with_targets=True)
    errors = validate_dataset(source)
    if errors:
        raise ValueError("invalid source dataset: " + "; ".join(errors))
    by_id = {e.id: e for e in dataset.select(split=split)}
    # Dataset.select filters membership in episode-file order; reuse promises split order.
    candidates = [by_id[eid] for eid in dataset.info.splits[split]]
    if ids is not None:
        wanted = {f"mimic-cxr-s{identifier(s, numeric=True)}" for s in ids}
        if not wanted or len(wanted) != len(ids) or wanted - {e.id for e in candidates}:
            raise ValueError("duplicate or unknown requested study IDs")
        candidates = [e for e in candidates if e.id in wanted]
    selected = candidates[:limit] if limit is not None else candidates
    if not selected:
        raise ValueError("no studies selected")
    origin_rows = [json.loads(s) for s in safe_path(source, "provenance.jsonl").read_text().splitlines() if s.strip()]
    origins = {r["id"]: r for r in origin_rows}
    if len(origins) != len(origin_rows):
        raise ValueError("duplicate source provenance IDs")
    episodes, targets, provenance = [], [], []
    with staging(out) as stage:
        (stage / "artifacts").mkdir()
        for episode in selected:
            if len(episode.turns) != 1 or not re.fullmatch(r"mimic-cxr-s[0-9]+", episode.id):
                raise ValueError("expected one-turn MIMIC-CXR studies")
            turn = episode.turns[0]
            target = dataset.targets[episode.id].turns[turn.id]["answer"]
            if not isinstance(target, dict) or any(
                    not isinstance(target.get(k), str) or not target[k].strip() for k in required):
                raise ValueError("missing structured reference section; no full-report fallback")
            if not turn.evidence:
                raise ValueError("study has no images")
            origin = origins[episode.id]
            image_origins = origin["images"]
            if [i["file"] for i in image_origins] != [e.file for e in turn.evidence]:
                raise ValueError("image/provenance association mismatch")
            evidence, copied = [], []
            for item, image_origin in zip(turn.evidence, image_origins):
                if item.type != "image" or not item.file or item.text and not re.fullmatch(
                        r"View(?:Position)?: [A-Za-z -]+", item.text):
                    raise ValueError("expected only JPEG views and view-position text")
                path = safe_path(source, item.file)
                decode_image(path)
                checksum = digest(path)
                if checksum != image_origin["sha256"]:
                    raise ValueError("source image hash mismatch")
                relative = f"artifacts/{checksum}.jpg"
                shutil.copy2(path, stage / relative)
                if digest(stage / relative) != checksum:
                    raise ValueError("copied image hash mismatch")
                evidence.append({**item.model_dump(exclude_none=True), "file": relative})
                copied.append({**image_origin, "file": relative, "source_file": item.file})
            episodes.append({"id": episode.id, "turns": [{"id": turn.id, "evidence": evidence}]})
            targets.append({"id": episode.id, "turns": {turn.id: {
                "answer": {k: target[k] for k in required}}}})
            provenance.append({**origin, "images": copied})
        source_files = ["dataset.json", "episodes.jsonl", "targets.jsonl", "provenance.jsonl"]
        source_files += [n for n in ("cohort.json", "import_report.json") if (source / n).exists()]
        report = {"protocol": "mimic-cxr-ama-reuse-v1", "source_format": "ama",
                  "report_mode": report_mode, "split": split, "scorer": "unscored",
                  "n_imported": len(episodes), "n_images": sum(len(e["turns"][0]["evidence"]) for e in episodes),
                  "selected_episode_ids": [e["id"] for e in episodes],
                  "selection_rule": "preserve source split order; optional ID filter and prefix limit; no resampling",
                  "reference_extraction": "structured targets.answer sections copied unchanged",
                  "source_sha256": {n: digest(safe_path(source, n)) for n in source_files}}
        if "cohort.json" in source_files:
            json_file(stage / "source_cohort.json", json.loads((source / "cohort.json").read_text()))
        _write_dataset(stage, out.name, split, episodes, targets, provenance, report)
    return report
