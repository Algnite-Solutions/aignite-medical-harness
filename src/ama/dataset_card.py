"""Small, deterministic bilingual Dataset Card writer for importers."""
from __future__ import annotations

from pathlib import Path


def write_dataset_cards(
    out: Path,
    *,
    name: str,
    source: str,
    license_name: str,
    purpose_en: str,
    purpose_zh: str,
    construction_en: str,
    construction_zh: str,
    episodes: int,
    turns: int,
    evidence: int,
    scorer: str,
    targets: int,
    limitations_en: str,
    limitations_zh: str,
) -> None:
    artifacts_en = ("Artifacts are optional paths relative to this dataset directory. "
                    "This import contains no copied artifacts unless stated above.")
    artifacts_zh = "Artifact 是相对本数据集目录的可选路径；除非上文另有说明，本次导入没有复制 artifact。"
    en = f"""# {name} — Dataset Card

## Source and license

Source: {source}

License: {license_name or "Not specified by the importer; consult the source dataset."}

## Research use

{purpose_en}

## Construction

{construction_en}

## Size

- Episodes: {episodes}
- Turns: {turns}
- Evidence items: {evidence}

## Scoring and targets

Scorer: `{scorer}`. Target records: {targets}.

## Artifacts

{artifacts_en}

## Limitations and reporting

{limitations_en}
"""
    zh = f"""# {name}——数据卡

## 来源与许可

来源：{source}

许可：{license_name or "Importer 未注明；请查阅源数据集许可。"}

## 研究用途

{purpose_zh}

## 构造方法

{construction_zh}

## 规模

- Episode：{episodes}
- Turn：{turns}
- Evidence：{evidence}

## 评分与 targets

Scorer：`{scorer}`。Target 记录数：{targets}。

## Artifacts

{artifacts_zh}

## 限制与报告口径

{limitations_zh}
"""
    (Path(out) / "DATASET_CARD.md").write_text(en, encoding="utf-8")
    (Path(out) / "DATASET_CARD.zh-CN.md").write_text(zh, encoding="utf-8")
