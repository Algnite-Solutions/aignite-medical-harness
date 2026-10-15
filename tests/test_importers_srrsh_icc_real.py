"""Standalone ICC 交付包的真实 Excel 集成测试。

真实工作簿位于 Git repo 外、与 repo 同级的目录中。测试只核验导入结构与
聚合计数（不落任何个案临床字段）；生成目录在测试结束后自动删除。
聚合计数基准 = 2026-09-23 盘点报告（ICC临床数据盘点-2026-09-23.md）。
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ama.data import validate_dataset
from ama.importers.srrsh_icc import import_srrsh_icc

DELIVERY_ROOT = Path(__file__).resolve().parents[2]
REAL_XLSX = DELIVERY_ROOT / "ICC临床数据.xlsx"
TEST_PATIENT_ID = "2272542"


@pytest.mark.skipif(not REAL_XLSX.exists(), reason="standalone ICC Excel is outside the repo")
def test_standalone_real_excel_imports_requested_patient():
    with TemporaryDirectory(prefix="icc-real-import-") as temporary_dir:
        out = Path(temporary_dir) / "srrsh_icc"
        report = import_srrsh_icc(REAL_XLSX, out, ids=[TEST_PATIENT_ID])

        assert report["skipped"] == []
        assert report["counts"] == {"episodes": 1, "turns": 1, "evidence": 5}
        assert validate_dataset(out) == []

        episodes = [line for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(episodes) == 1
        assert f'"icc-{TEST_PATIENT_ID}"' in episodes[0]
        assert len((out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        assert not (out / "targets.jsonl").exists()


@pytest.mark.skipif(not REAL_XLSX.exists(), reason="standalone ICC Excel is outside the repo")
def test_standalone_real_excel_full_cohort_reconciles_with_audit():
    """全量导入的聚合计数与 2026-09-23 盘点报告对账（数值正确性闸门）。"""
    with TemporaryDirectory(prefix="icc-real-full-") as temporary_dir:
        out = Path(temporary_dir) / "srrsh_icc"
        report = import_srrsh_icc(REAL_XLSX, out)

        assert report["n_rows"] == 256
        assert report["n_imported"] == 256
        assert report["skipped"] == []
        assert report["counts"] == {"episodes": 256, "turns": 256, "evidence": 1280}
        assert report["table_origin"]["counts"] == {"临床数据总表": 188, "表2": 68}
        # 盘点 §2 锚点列的非空数
        nonnull = report["missing_stats"]["nonnull_by_column"]
        assert nonnull["T"] == 188
        assert nonnull["MVI"] == 6
        assert nonnull["PLT"] == 63
        assert nonnull["TACE"] == 255
        assert nonnull["nerve_invasion"] == 84

        assert validate_dataset(out) == []
        assert len((out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()) == 256
        assert len((out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()) == 256
        assert not (out / "targets.jsonl").exists()
