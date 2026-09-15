"""Offline ROCOv2 radiology conversion into single-turn image Episodes."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..dataset_card import write_dataset_cards

_SPLIT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path, value_field: str, key_field: str = "ID") -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or key_field not in reader.fieldnames or value_field not in reader.fieldnames:
            raise ValueError(f"{path.name}: expected {key_field} and {value_field} columns")
        rows: dict[str, str] = {}
        for row in reader:
            image_id = (row.get(key_field) or "").strip()
            if image_id:
                rows[image_id] = (row.get(value_field) or "").strip()
        return rows


def _licenses(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"ID", "PMCID", "Attribution", "Link"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path.name}: expected {sorted(required)} columns")
        return {(r["ID"] or "").strip(): {k: (r[k] or "").strip()
                                               for k in ("PMCID", "Attribution", "Link")}
                for r in reader if (r.get("ID") or "").strip()}


def _cuis(raw: str) -> list[str]:
    return sorted({part.strip().upper() for part in raw.split(";") if part.strip()})


def import_rocov2(source: Path, out: Path, split: str = "test", limit: int | None = None,
                   ids: list[str] | None = None) -> dict[str, Any]:
    source, out = Path(source), Path(out)
    if not _SPLIT_RE.fullmatch(split):
        raise ValueError(f"invalid split: {split!r}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    inputs = {
        "captions": source / f"{split}_captions.csv",
        "concepts": source / f"{split}_concepts_manual.csv",
        "licenses": source / "license_information.csv",
        "mapping": source / "cui_mapping.csv",
    }
    image_dir = source / split
    missing = [str(p) for p in [*inputs.values(), image_dir] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing ROCOv2 input: " + ", ".join(missing))

    captions = _rows(inputs["captions"], "Caption")
    concepts = _rows(inputs["concepts"], "CUIs")
    licenses = _licenses(inputs["licenses"])
    mapping = _rows(inputs["mapping"], "Canonical name", key_field="CUI")
    image_paths = {p.stem: p for p in sorted(image_dir.glob("*.jpg"))}
    complete = sorted(set(captions) & set(concepts) & set(licenses) & set(image_paths))
    if ids is not None:
        selected = []
        for image_id in dict.fromkeys(ids):
            absent = [name for name, rows in (("caption", captions), ("concepts", concepts),
                                                ("license", licenses), ("image", image_paths))
                      if image_id not in rows]
            if absent:
                raise ValueError(f"{image_id}: missing {', '.join(absent)}")
            selected.append(image_id)
    else:
        selected = complete
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no complete ROCOv2 records selected")

    out.mkdir(parents=True, exist_ok=True)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    episodes, targets = [], []
    provenance = []
    for image_id in selected:
        cuis = _cuis(concepts[image_id])
        unknown = [cui for cui in cuis if cui not in mapping]
        if unknown:
            raise ValueError(f"{image_id}: CUI missing from mapping: {unknown}")
        destination = artifacts / image_paths[image_id].name
        shutil.copy2(image_paths[image_id], destination)
        evidence_id = f"image-{image_id}"
        license_row = licenses[image_id]
        provenance.append({"image_id": image_id, **license_row})
        episodes.append({
            "episode_id": image_id,
            "subject_id": image_id,
            "metadata": {"source": "ROCOv2", "split": split, "provenance": license_row},
            "turns": [{
                "turn_id": "t1", "time": None,
                "message": ("Review the released radiology image. Submit state.caption as one concise "
                            "English radiology caption and state.cuis as a list of UMLS CUI strings. "
                            "Cite the image evidence."),
                "evidence": [{"evidence_id": evidence_id, "kind": "image", "text": "",
                              "artifact": f"artifacts/{destination.name}",
                              "source": license_row["Link"], "metadata": {}}],
            }],
        })
        targets.append({"episode_id": image_id, "turns": {"t1": {
            "state": {"caption": captions[image_id], "cuis": cuis},
            "required_evidence": [evidence_id],
        }}})

    dataset = {
        "schema": "ama-dataset-v0", "name": out.name, "version": "0.1",
        "description": "ROCOv2 single-turn radiology image caption and concept prediction.",
        "splits": {split: selected}, "scorer": "rocov2_v0", "license": "CC BY-NC-SA 4.0",
    }
    policy = {"public": {"guidance": (
        "For every image, submit state.caption as a string and state.cuis as a list of UMLS CUI strings."
    )}, "hidden": {}}
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes) + "\n", encoding="utf-8")
    (out / "targets.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in targets) + "\n", encoding="utf-8")
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(source.resolve()), "split": split, "selected_ids": selected,
        "n_complete_source_records": len(complete), "n_imported": len(selected),
        "excluded": {"missing_caption": len(set(image_paths) - set(captions)),
                     "missing_concepts": len(set(image_paths) - set(concepts)),
                     "missing_license": len(set(image_paths) - set(licenses))},
        "source_sha256": {name: _sha(path) for name, path in inputs.items()},
        "provenance": provenance,
    }
    (out / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_cards(
        out, name=out.name,
        source="ROCOv2 radiology (https://github.com/sctg-development/ROCOv2-radiology)",
        license_name="ROCOv2: CC BY-NC-SA 4.0; each image retains its source attribution",
        purpose_en="Single-turn radiology image captioning and UMLS concept prediction research.",
        purpose_zh="用于单轮放射影像描述与 UMLS 概念预测研究。",
        construction_en="Each selected image becomes one Episode with one Turn and one image Evidence item. Captions and CUIs remain evaluator-only targets.",
        construction_zh="每张选定图像转换为一个 Episode、一个 Turn 和一条图像 Evidence；caption 与 CUI 仅保存在 evaluator target 中。",
        episodes=len(episodes), turns=len(episodes), evidence=len(episodes),
        scorer="rocov2_v0", targets=len(targets),
        limitations_en="This derived subset is not an official ROCOv2 benchmark result. Captions have one reference and token overlap does not measure clinical correctness.",
        limitations_zh="该派生子集的结果不是官方 ROCOv2 benchmark 成绩；caption 仅有单一参考，token 重合度不能代表临床正确性。",
        artifacts_en="JPEG images are copied under artifacts/. Per-image PMC links and attributions are recorded in Episode metadata and import_report.json.",
        artifacts_zh="JPEG 图像复制到 artifacts/；逐图 PMC 链接与署名记录在 Episode metadata 和 import_report.json 中。",
    )
    if len(provenance) <= 20:
        en_rows = ["", "## Included-image provenance", "", "| Image | PMCID | Attribution | Source |",
                   "|---|---|---|---|"]
        zh_rows = ["", "## 收录图像来源", "", "| 图像 | PMCID | 署名 | 来源 |",
                   "|---|---|---|---|"]
        for row in provenance:
            values = (row["image_id"], row["PMCID"], row["Attribution"], row["Link"])
            line = f"| {values[0]} | {values[1]} | {values[2]} | [article]({values[3]}) |"
            en_rows.append(line)
            zh_rows.append(line.replace("[article]", "[论文]"))
        with (out / "DATASET_CARD.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(en_rows) + "\n")
        with (out / "DATASET_CARD.zh-CN.md").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(zh_rows) + "\n")
    return report
