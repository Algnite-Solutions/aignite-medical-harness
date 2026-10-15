"""ICC NAS 数据的可选只读集成审计。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ama.importers.srrsh_icc import audit_srrsh_icc_multimodal

NAS_ROOT = os.environ.get("SRRSH_ICC_ROOT")


@pytest.mark.skipif(not NAS_ROOT, reason="set SRRSH_ICC_ROOT to audit the external NAS dataset")
def test_nas_multimodal_source_can_be_audited_without_writes():
    root = Path(NAS_ROOT)
    generated = {
        name: (root / name).stat().st_mtime_ns
        for name in ("dataset.json", "episodes.jsonl", "import_report.json")
        if (root / name).exists()
    }

    report = audit_srrsh_icc_multimodal(root)

    assert report["cohort"]["clinical_patients"] > 0
    assert report["cohort"]["report_patients"] > 0
    assert report["cohort"]["mri_patients"] > 0
    assert report["cohort"]["wsi_patients"] > 0
    assert report["coverage"]["clinical_with_all_modalities"] > 0
    assert generated == {
        name: (root / name).stat().st_mtime_ns
        for name in generated
    }
