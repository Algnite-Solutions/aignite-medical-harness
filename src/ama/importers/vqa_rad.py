"""Offline HF VQA-RAD snapshot -> AMA episodes, evaluator targets and audit files."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import tempfile
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from ..data import validate_dataset
from ..dataset_card import write_dataset_cards
from .vqa_rad_scoring import SCORER

REPO = "flaviagiammarino/vqa-rad"
REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
AUTOMEDBENCH_REVISION = "5a19ff72491df7f8c0279438fcfade9364f9a8b7"
INSTRUCTIONS = (
    'Answer the radiology question using the image. Give a concise English answer.\n'
    'The task answer must be an object with exactly one string field: {"text": "your answer"}.\n'
    'For yes/no questions, answer only "yes" or "no" in that field. '
    'Do not put an explanation in the text field.\n'
)


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument("--subset", choices=("full", "smoke"), default="full")
    parser.add_argument("--smoke-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)


def import_from_cli(args: Namespace) -> dict[str, Any]:
    return import_vqa_rad(args.source, args.out, subset=args.subset,
                          smoke_count=args.smoke_count, seed=args.seed)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _snapshot(source: Path) -> tuple[Path, dict[str, Any], dict[str, list[Path]]]:
    """Verify the local download manifest before consuming any Parquet bytes.

    This checks consistency with the supplied manifest, not its authenticity.
    The download script separately compares every file to the upstream revision.
    """
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("repo_id") != REPO or manifest.get("revision") != REVISION:
        raise ValueError("expected the pinned VQA-RAD repo and revision")
    snapshot = source / "snapshot"
    files = {}
    for entry in manifest["files"]:
        name = entry["path"]
        path = snapshot / name
        if (name in files or Path(name).is_absolute() or ".." in Path(name).parts
                or snapshot.resolve() not in path.resolve().parents):
            raise ValueError("duplicate or unsafe source path")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}")
        files[name] = path
    shards = {split: [files[name] for name in sorted(files)
                      if name.startswith(f"data/{split}-") and name.endswith(".parquet")]
              for split in ("train", "test")}
    if any(not paths for paths in shards.values()):
        raise ValueError("both train and test Parquet shards are required")
    discovered = set(snapshot.glob("data/*.parquet"))
    if discovered != {path for paths in shards.values() for path in paths}:
        raise ValueError("unlisted or unexpected Parquet shards")
    return snapshot, manifest, shards


def import_vqa_rad(source: Path, out: Path, *, subset: str = "full",
                   smoke_count: int = 8, seed: int = 42) -> dict[str, Any]:
    """Preserve split + source row identity; publish output only after validation."""
    try:
        import pyarrow
        import pyarrow.parquet as pq
        import PIL
        from PIL import Image
    except ImportError as exc:
        raise ImportError("VQA-RAD import needs: pip install '.[vqa-rad]'") from exc

    source, out = Path(source), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    if subset not in {"full", "smoke"} or smoke_count < 1:
        raise ValueError("subset must be full/smoke and smoke-count must be positive")
    snapshot, manifest, shards = _snapshot(source)
    episodes, targets, provenance = [], [], []
    splits = {"train": [], "test": []}
    source_counts, selected_rows, source_files = {}, {}, {}
    types = {split: Counter() for split in splits}
    image_hashes = {split: set() for split in splits}
    selected_images = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ama-vqarad-", dir=out.parent) as temp:
        stage = Path(temp) / "dataset"
        (stage / "artifacts").mkdir(parents=True)
        for split, paths in shards.items():
            count = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
            if not count:
                raise ValueError(f"empty source split: {split}")
            source_counts[split] = count
            indexes = list(range(count)) if subset == "full" else (
                sorted(random.Random(seed).sample(range(count), min(smoke_count, count)))
                if split == "test" else [])
            selected_rows[split] = indexes
            selected = set(indexes)
            row_index = 0
            for path in paths:
                source_files[str(path.relative_to(snapshot))] = _sha(path)
                parquet = pq.ParquetFile(path)
                if not {"image", "question", "answer"} <= set(parquet.schema_arrow.names):
                    raise ValueError(f"{path.name}: expected image, question, answer columns")
                for batch in parquet.iter_batches(batch_size=128, columns=["image", "question", "answer"]):
                    for row in batch.to_pylist():
                        idx = row_index
                        row_index += 1
                        if not isinstance(row["question"], str) or not row["question"].strip():
                            raise ValueError(f"{split}/{idx}: empty or invalid question")
                        if not isinstance(row["answer"], str) or not row["answer"].strip():
                            raise ValueError(f"{split}/{idx}: empty or invalid answer")
                        image = row["image"]
                        raw = image.get("bytes") if isinstance(image, dict) else None
                        if not isinstance(raw, bytes) or not raw:
                            raise ValueError(f"{split}/{idx}: embedded image bytes required")
                        image_sha = hashlib.sha256(raw).hexdigest()
                        image_hashes[split].add(image_sha)
                        if idx not in selected:
                            continue
                        artifact = f"artifacts/{image_sha}.png"
                        if image_sha not in selected_images:
                            with Image.open(io.BytesIO(raw)) as im:
                                im.load()
                                if im.mode not in ("RGB", "L"):
                                    im = im.convert("RGB")
                                # A fresh image removes source metadata without JPEG recompression.
                                clean = Image.frombytes(im.mode, im.size, im.tobytes())
                                clean.save(stage / artifact, format="PNG")
                            selected_images.add(image_sha)
                        episode_id = f"vqarad-{split}-{idx:06d}"
                        kind = "closed_yes_no" if row["answer"].strip().lower() in {"yes", "no"} else "open"
                        types[split][kind] += 1
                        splits[split].append(episode_id)
                        episodes.append({"id": episode_id, "turns": [{"id": "t1", "evidence": [
                            {"id": "image", "type": "image", "file": artifact},
                            {"id": "question", "type": "text", "text": row["question"]},
                        ]}]})
                        targets.append({"id": episode_id, "turns": {"t1": {
                            "answer": {"text": row["answer"].strip()}, "question_type": kind,
                            "required_evidence": ["image", "question"],
                        }}})
                        provenance.append({"id": episode_id, "split": split, "row_index": idx,
                                           "shard": str(path.relative_to(snapshot)),
                                           "image_sha256": image_sha})

        report = {
            "source": {"repo_id": REPO, "revision": REVISION,
                       "files_sha256": source_files,
                       "snapshot_sha256": hashlib.sha256(json.dumps(
                           {entry["path"]: entry["sha256"] for entry in manifest["files"]},
                           sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
            "automedbench_revision": AUTOMEDBENCH_REVISION,
            "selection": {"subset": subset, "seed": seed if subset == "smoke" else None,
                          "count": smoke_count if subset == "smoke" else None,
                          "algorithm": "all rows" if subset == "full" else
                          "test: sorted(random.Random(seed).sample(range(N), min(count, N))); train: empty",
                          "order": "train then test; lexicographic shard name; physical row order",
                          "selected_row_indices": selected_rows},
            "source_counts": source_counts, "n_imported": len(episodes),
            "split_counts": {split: len(ids) for split, ids in splits.items()},
            "question_type_counts": {split: dict(counts) for split, counts in types.items()},
            "unique_source_images": {split: len(hashes) for split, hashes in image_hashes.items()},
            "source_image_overlap_train_test": len(image_hashes["train"] & image_hashes["test"]),
            "n_artifacts": len(selected_images),
            "runtime": {"python": platform.python_version(), "pyarrow": pyarrow.__version__,
                        "pillow": PIL.__version__},
        }
        _json(stage / "dataset.json", {"schema": "ama-dataset", "name": out.name, "splits": splits})
        _jsonl(stage / "episodes.jsonl", episodes)
        _jsonl(stage / "targets.jsonl", targets)
        _jsonl(stage / "provenance.jsonl", provenance)
        _json(stage / "import_report.json", report)
        _json(stage / "eval.json", {"scorer": SCORER})
        (stage / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
        write_dataset_cards(
            stage, name=out.name, source=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
            license_name="Upstream HF data card: CC0-1.0. Adapted AutoMedBench task/scoring code: non-commercial research only; see docs/licenses/AutoMedBench-LICENSE.txt.",
            purpose_en="Single-turn radiology visual question answering research; no clinical use.",
            purpose_zh="单轮放射影像视觉问答研究，不用于临床决策。",
            construction_en=f"Subset: {subset}. Original train/test membership and source-row IDs are retained. Each QA row is one Episode/Turn with image and question Evidence. Answers and answer-derived question types are evaluator-only. Images are metadata-free PNGs with no JPEG recompression. Duplicate images share an artifact.",
            construction_zh=f"子集：{subset}。保留源 train/test 归属和行号 ID；每条 QA 对应一个 Episode/Turn，输入为图像和问题 Evidence。答案及由答案推导的题型仅供评测。图像去元数据后写为 PNG，避免 JPEG 再压缩；重复图像复用 artifact。",
            episodes=len(episodes), turns=len(episodes), evidence=2 * len(episodes),
            scorer=SCORER, targets=len(targets),
            limitations_en="AMA heuristic: normalized whole-answer exact match for normalized yes/no gold; otherwise (EM + token F1)/2. F1 is a token-multiset score averaged per question. Lowercase, delete ASCII punctuation, collapse whitespace, then apply the pinned AutoMedBench yes/no, number and abbreviation maps. Missing/invalid/failed answers score zero. Scores are 0–1; empty groups are null. No LLM judge, S1–S5 or clinical correctness claim. Train/test may share images; row splits are not patient splits. Full data includes train: evaluate held-out performance on test explicitly. Smoke is only a pipeline check. See docs/vqa-rad.md for all differences.",
            limitations_zh="AMA heuristic：规范化后参考答案为 yes/no 时进行整句精确匹配，否则取 (EM + token F1)/2；F1 按 token 多重集计算后逐题平均。转小写、删除 ASCII 标点、合并空白，再按固定 AutoMedBench 表转换 yes/no、数字和缩写。缺失、无效、失败回答计零；分数为 0–1，空组为 null。没有 LLM judge 或 S1–S5，不代表临床正确性。train/test 可能共享图像，行级划分不是患者级划分；完整目录包含 train，评价留出集表现须显式选 test。smoke 仅用于管线验收。完整差异见 docs/vqa-rad.md。",
            artifacts_en="Local-only images in artifacts/. provenance.jsonl maps source rows; import_report.json records selection and source hashes. checksums.json covers every generated file except itself. Do not commit real data or per-row provenance.",
            artifacts_zh="图像位于本地 artifacts/；provenance.jsonl 保存源行映射，import_report.json 记录选择规则和源哈希。checksums.json 覆盖除自身外的所有生成文件；真实数据及逐例来源不提交 Git。",
        )
        errors = validate_dataset(stage)
        if errors:
            raise ValueError("generated dataset invalid: " + "; ".join(errors))
        checksums = {str(path.relative_to(stage)): _sha(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()}
        _json(stage / "checksums.json", {
            "files": checksums,
            "overall_sha256": hashlib.sha256(json.dumps(checksums, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest(),
        })
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output already exists: {out}")
        stage.rename(out)
    return report
