"""Import the pinned Hugging Face PathVQA Parquet snapshot into AMA."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import string
import tempfile
from argparse import ArgumentParser, Namespace
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from ..data import validate_dataset
from ..dataset_card import write_dataset_cards

REPOSITORY = "flaviagiammarino/path-vqa"
REVISION = "7568940e604f3e911441b1f6c7aedb5f1c2866bd"
SOURCE_LOCK = Path(__file__).with_name("pathvqa-source.lock.json")
SPLITS = ("train", "validation", "test")
INSTRUCTIONS = (
    "Answer each pathology visual question in concise English using the supplied image.\n"
    "The answer field must be a short string, not an object. For a binary question, "
    "return exactly yes or no. Cite the image evidence used for the answer.\n"
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _snapshot(source: Path, source_lock: Path) -> tuple[str, dict[str, list[tuple[Path, str]]]]:
    manifest_path = source / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest.get("resolved_revision")
    if manifest.get("repository") != REPOSITORY or revision != REVISION:
        raise ValueError("PathVQA source repository or revision differs from the pinned release")
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    if lock.get("repository") != REPOSITORY or lock.get("revision") != REVISION:
        raise ValueError("invalid committed PathVQA source lock")
    locked = {item["path"]: item for item in lock["files"] if item["path"].endswith(".parquet")}
    if len(locked) != sum(item["path"].endswith(".parquet") for item in lock["files"]):
        raise ValueError("duplicate source lock entries")
    entries: dict[str, list[tuple[Path, str]]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    for item in manifest.get("files", []):
        relative = item.get("path", "")
        if not isinstance(relative, str):
            raise ValueError("invalid source manifest path")
        if not relative.endswith(".parquet"):
            continue
        if not relative.startswith("data/"):
            raise ValueError(f"unexpected Parquet path: {relative}")
        if relative in seen:
            raise ValueError(f"duplicate source manifest entry: {relative}")
        seen.add(relative)
        if relative not in locked or item.get("sha256") != locked[relative].get("sha256"):
            raise ValueError(f"source manifest differs from committed lock: {relative}")
        path = (source / relative).resolve()
        if source.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"missing or unsafe source file: {relative}")
        split = next((s for s in SPLITS if Path(relative).name.startswith(s + "-")), None)
        if split is None:
            raise ValueError(f"unknown source split: {relative}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"missing SHA-256 for {relative}")
        if path.stat().st_size != locked[relative]["size"]:
            raise ValueError(f"source size mismatch: {relative}")
        if _sha_file(path) != expected:
            raise ValueError(f"source checksum mismatch: {relative}")
        entries[split].append((path, relative))
    if seen != set(locked) or any(not entries[split] for split in SPLITS):
        raise ValueError("source snapshot does not contain exactly the locked Parquet shards")
    for split in SPLITS:
        entries[split].sort(key=lambda item: item[1])
    return revision, entries


def import_pathvqa(source: Path, out: Path, *, limit_per_split: int | None = None,
                   benchmark_count: int = 500,
                   source_lock: Path = SOURCE_LOCK) -> dict[str, Any]:
    """Build a self-contained dataset; never overwrite an existing destination."""
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('install the importer dependencies with pip install -e ".[pathvqa]"') from exc

    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be positive")
    if benchmark_count < 0:
        raise ValueError("benchmark_count must be nonnegative")
    revision, files = _snapshot(source, source_lock)
    test_count = sum(pq.ParquetFile(path).metadata.num_rows for path, _ in files["test"])
    if limit_per_split is not None:
        test_count = min(test_count, limit_per_split)
    if benchmark_count > test_count and limit_per_split is None:
        raise ValueError("benchmark_count exceeds test split")
    indexes = (sorted(random.Random(42).sample(range(test_count), benchmark_count))
               if 0 < benchmark_count <= test_count else [])
    benchmark_qids = {index: f"pathvqa_{position:05d}" for position, index in enumerate(indexes)}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    split_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    source_files: dict[str, list[str]] = {s: [] for s in SPLITS}
    questions = Counter()
    images: set[str] = set()
    try:
        image_dir = stage / "artifacts" / "images"
        image_dir.mkdir(parents=True)
        with (stage / "episodes.jsonl").open("w", encoding="utf-8") as episodes, \
                (stage / "targets.jsonl").open("w", encoding="utf-8") as targets, \
                (stage / "provenance.jsonl").open("w", encoding="utf-8") as provenance:
            for split in SPLITS:
                source_index = 0
                for parquet_path, relative in files[split]:
                    if limit_per_split is not None and source_index >= limit_per_split:
                        break
                    source_files[split].append(relative)
                    parquet = pq.ParquetFile(parquet_path)
                    if set(parquet.schema_arrow.names) != {"image", "question", "answer"}:
                        raise ValueError(f"unexpected PathVQA columns: {relative}")
                    file_row = 0
                    for batch in parquet.iter_batches(batch_size=64):
                        for row in batch.to_pylist():
                            if limit_per_split is not None and source_index >= limit_per_split:
                                break
                            question, answer = row["question"], row["answer"]
                            image_value = row["image"]
                            if not isinstance(question, str) or not question.strip() or \
                                    not isinstance(answer, str) or not answer.strip() or \
                                    not isinstance(image_value, dict) or not isinstance(image_value.get("bytes"), bytes):
                                raise ValueError(f"invalid source row: {relative}:{file_row}")
                            image_bytes = image_value["bytes"]
                            with Image.open(BytesIO(image_bytes)) as image:
                                image.verify()
                                image_format = image.format
                            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
                            if extension is None:
                                raise ValueError(f"unsupported image format: {relative}:{file_row}")
                            image_hash = hashlib.sha256(image_bytes).hexdigest()
                            artifact = f"artifacts/images/{image_hash}{extension}"
                            if image_hash not in images:
                                (stage / artifact).write_bytes(image_bytes)
                                images.add(image_hash)
                            eid = f"pathvqa-{split}-{source_index:06d}"
                            # Preserve the existing source adapter's target representation.
                            # Scoring applies its own documented lexical normalization.
                            normalized = answer.strip().lower().strip(string.punctuation).strip()
                            if not normalized:
                                raise ValueError(f"empty normalized answer: {relative}:{file_row}")
                            kind = "binary" if normalized in {"yes", "no"} else "open"
                            questions[kind] += 1
                            episode = {"id": eid, "turns": [{"id": "t1", "observation": question,
                                       "evidence": [{"id": "image", "type": "image", "file": artifact}]}]}
                            target = {"id": eid, "turns": {"t1": {"answer": answer,
                                      "normalized_answer": normalized, "question_type": kind,
                                      "required_evidence": ["image"]}}}
                            origin = {"id": eid, "repository": REPOSITORY, "revision": revision,
                                      "split": split, "source_index": source_index,
                                      "source_file": relative, "source_file_row": file_row,
                                      "image_sha256": image_hash, "artifact": artifact}
                            if split == "test" and source_index in benchmark_qids:
                                origin["automedbench_qid"] = benchmark_qids[source_index]
                            for stream, value in ((episodes, episode), (targets, target),
                                                  (provenance, origin)):
                                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                            split_ids[split].append(eid)
                            source_index += 1
                            file_row += 1
                        if limit_per_split is not None and source_index >= limit_per_split:
                            break
                if not split_ids[split]:
                    raise ValueError(f"no records imported from {split}")

        if indexes:
            split_ids[f"automedbench_test_{benchmark_count}"] = [split_ids["test"][i] for i in indexes]
        _write_json(stage / "dataset.json", {"schema": "ama-dataset", "name": out.name,
                                               "splits": split_ids})
        _write_json(stage / "eval.json", {"scorer": "pathvqa_v1"})
        (stage / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
        report = {"repository": REPOSITORY, "revision": revision, "source_dir": str(source.resolve()),
                  "parquet_files": source_files,
                  "split_counts": {s: len(split_ids[s]) for s in SPLITS},
                  "n_imported": sum(len(split_ids[s]) for s in SPLITS),
                  "n_targets": sum(len(split_ids[s]) for s in SPLITS),
                  "n_unique_images": len(images), "question_types": dict(questions),
                  "automedbench": {"source_split": "test", "requested_count": benchmark_count,
                                   "selected_count": len(indexes), "seed": 42,
                                   "source_indexes": indexes,
                                   "split_name": f"automedbench_test_{benchmark_count}" if indexes else None},
                  "limit_per_split": limit_per_split, "scorer": "pathvqa_v1"}
        _write_json(stage / "import_report.json", report)
        write_dataset_cards(stage, name=out.name, source=f"{REPOSITORY}@{revision}",
            license_name="Hugging Face entry marked MIT; underlying image rights require source review.",
            purpose_en="Single-turn pathology image question answering.",
            purpose_zh="单轮病理图像问答。",
            construction_en="One source row is one Episode; image bytes are shared by hash. Answers are evaluator-only.",
            construction_zh="每条原始记录对应一个 Episode；图像按哈希共享，答案仅供评测读取。",
            episodes=report["n_imported"], turns=report["n_imported"],
            evidence=report["n_imported"], scorer="pathvqa_v1", targets=report["n_targets"],
            limitations_en="Lexical metrics do not establish clinical correctness or reproduce the official AutoMedBench score.",
            limitations_zh="词面指标不能证明临床正确性，也不是 AutoMedBench 官方得分。",
            artifacts_en="Image bytes are preserved under artifacts/images; provenance records source rows.",
            artifacts_zh="图像原始字节保存在 artifacts/images，provenance 记录来源行。")
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("invalid imported dataset: " + "; ".join(errors[:5]))
        stage.rename(out)
        return report
    except Exception:
        shutil.rmtree(stage)
        raise


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument("--limit-per-split", type=int, help="smoke import only")
    parser.add_argument("--benchmark-count", type=int, default=500)


def import_from_cli(args: Namespace) -> dict[str, Any]:
    return import_pathvqa(args.source, args.out, limit_per_split=args.limit_per_split,
                          benchmark_count=args.benchmark_count)
