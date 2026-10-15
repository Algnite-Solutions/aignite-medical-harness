"""SRRSH ICC 多模态文件夹 importer 的合成数据单元测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from ama.cli import main
from ama.data import load_dataset, validate_dataset
from ama.importers.srrsh_icc import (
    CLIN_FIELDS,
    CLINICAL_XLSX,
    EXPECTED_MRI_KEYS,
    LAB_FIELDS,
    MRI_DIR,
    MRI_EVIDENCE_ID,
    MRI_REPORT_PREFIX,
    PATH_FIELDS,
    REPORT_XLSX,
    SHEET_NAME,
    WSI_DIR,
    audit_srrsh_icc_multimodal,
    import_srrsh_icc,
    import_srrsh_icc_multimodal,
)
from ama.runner import execute

PATIENT_ID = "1001"
SECOND_PATIENT_ID = "1002"
CLINICAL_HEADERS = [
    "ID",
    "surg_time",
    "pathology",
    *CLIN_FIELDS,
    *LAB_FIELDS,
    *PATH_FIELDS,
    "TACE",
    "TACE_次数",
]


def _clinical_row(patient_id: str) -> dict:
    row = {header: None for header in CLINICAL_HEADERS}
    row.update({
        "ID": patient_id,
        "surg_time": "2025-01-15",
        "pathology": "ICC",
        "Gender": "Male",
        "Age": 58,
        "height": 179,
        "weight": 79.8,
        "BMI": 24.9,
        "smoke": "Yes",
        "alcohol": "No",
        "HBV": "No",
        "hypertension": "No",
        "diabetes": "No",
        "Cirrhosis": "No",
        "child_pugh_grade": "A",
        "AFP": 2.22,
        "AST": 27,
        "ALP": 73,
        "GGT": 36,
        "ALB": 47.6,
        "BIL": 27.7,
        "PT": 12.8,
        "T": "T2",
        "N": "Nx",
        "M": "Mx",
        "Tumor_size": 4.5,
        "tumor_large_vascular_invasion": "Yes",
        "Tumor_number": 1,
        "nerve_invasion": "Yes",
        "Grade": "High",
        "TACE": "No",
    })
    return row


def _write_clinical(path: Path, rows: list[dict]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_NAME
    sheet.append(["synthetic data"])
    sheet.append(CLINICAL_HEADERS)
    for row in rows:
        sheet.append([row.get(header) for header in CLINICAL_HEADERS])
    workbook.save(path)
    workbook.close()


def _write_reports(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "sheet1"
    sheet.append(["住院号", "检查所见", "诊断"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _mri_filename(sequence: str, variant: str) -> str:
    suffix = {"raw": "", "n4": "_N4", "roi": "_ROI"}[variant]
    return f"{sequence}{suffix}.nii.gz"


def _add_mri(root: Path, patient_id: str, missing: tuple[str, str] | None = None) -> None:
    directory = root / MRI_DIR / f"{patient_id}_synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    for key in EXPECTED_MRI_KEYS:
        if key != missing:
            (directory / _mri_filename(*key)).write_bytes(b"synthetic-binary")


def _add_wsi(root: Path, patient_id: str, suffixes: tuple[str, ...] = ("A",)) -> None:
    directory = root / WSI_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (directory / f"{patient_id}-{suffix}.svs").write_bytes(b"synthetic-binary")


def _make_source(
    root: Path,
    include_second_patient: bool = True,
    include_mri_directory: bool = True,
    include_wsi_directory: bool = True,
) -> Path:
    root.mkdir()
    rows = [_clinical_row(PATIENT_ID)]
    if include_second_patient:
        rows.append(_clinical_row(SECOND_PATIENT_ID))
    _write_clinical(root / CLINICAL_XLSX, rows)
    _write_reports(root / REPORT_XLSX, [
        (PATIENT_ID, "合成的术前 MRI 检查所见。", "合成的 MRI 诊断。"),
        ("9999", "孤立合成报告。", "孤立合成诊断。"),
    ])
    if include_mri_directory:
        (root / MRI_DIR).mkdir(parents=True, exist_ok=True)
        _add_mri(root, PATIENT_ID)
        _add_mri(root, "9999")
        (root / MRI_DIR / "image_ICC.zip").write_bytes(b"synthetic")
        (root / MRI_DIR / f"{PATIENT_ID}_synthetic" / "report.dcm").write_bytes(b"synthetic")
    if include_wsi_directory:
        (root / WSI_DIR).mkdir(parents=True, exist_ok=True)
        _add_wsi(root, PATIENT_ID, ("A", "B", "C"))
        _add_wsi(root, "9999")
        (root / WSI_DIR / ".DS_Store").write_bytes(b"synthetic")
    return root


def test_imports_report_all_nifti_and_multiple_wsi(tmp_path):
    root = _make_source(tmp_path / "source")
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}

    report = import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])

    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert after - before == {
        "DATASET_CARD.md",
        "DATASET_CARD.zh-CN.md",
        "dataset.json",
        "episodes.jsonl",
        "import_report.json",
        "instructions.txt",
        "provenance.jsonl",
    }
    assert report["counts"]["episodes"] == 1
    assert report["counts"]["evidence"] == 27
    assert len(report["artifacts"]) == 21
    assert report["orphans"] == {
        "report_without_clinical": ["9999"],
        "mri_without_clinical": ["9999"],
        "wsi_without_clinical": ["9999"],
    }
    assert any(path.endswith("image_ICC.zip") for path in report["ignored_files"])
    assert any(path.endswith("report.dcm") for path in report["ignored_files"])
    assert validate_dataset(root) == []

    episode = load_dataset(root).episode(f"icc-{PATIENT_ID}")
    assert episode is not None
    assert len(episode.turns) == 1
    evidence = episode.turns[0].evidence
    assert [item.id for item in evidence[:6]] == [
        "clin-1", "lab-1", "tx-tace-1", MRI_EVIDENCE_ID, "tx-surg-1", "path-1",
    ]
    assert evidence[3].text.startswith(MRI_REPORT_PREFIX)
    assert episode.turns[0].observation is None
    assert len([item for item in evidence if item.type == "medical_image"]) == 12
    assert len([item for item in evidence if item.type == "image_mask"]) == 6
    assert len([item for item in evidence if item.type == "whole_slide_image"]) == 3
    for item in evidence[6:]:
        assert item.file is None
        assert "原始文件：" in item.text

    provenance = json.loads(
        (root / "provenance.jsonl").read_text(encoding="utf-8").strip()
    )
    assert set(provenance["evidence"]) == {item.id for item in evidence}
    report_provenance = provenance["evidence"][MRI_EVIDENCE_ID]
    assert report_provenance["relative_time"] == "preoperative"
    assert report_provenance["event_time_missing"] is True
    assert "event_time" not in report_provenance
    assert provenance["evidence"]["mri-dwi-raw"] == {
        "source": f"{MRI_DIR.as_posix()}/{PATIENT_ID}_synthetic/DWI.nii.gz",
        "modality": "MRI",
        "sequence": "DWI",
        "variant": "raw",
        "format": "NIfTI",
        "relative_time": "preoperative",
        "event_time_missing": True,
    }
    assert provenance["evidence"][f"wsi-{PATIENT_ID}-a"]["format"] == "SVS"
    card = (root / "DATASET_CARD.zh-CN.md").read_text(encoding="utf-8")
    assert "脱敏前不得进入仓库" in card
    assert "不复制" in card
    assert "不读取其像素" in card


def test_missing_modalities_and_one_mri_file_are_audited(tmp_path):
    root = _make_source(tmp_path / "source")
    missing_path = root / MRI_DIR / f"{PATIENT_ID}_synthetic" / "T1_DL_ROI.nii.gz"
    missing_path.unlink()

    audit = audit_srrsh_icc_multimodal(root)

    assert audit["missing"]["selected_without_report"] == [SECOND_PATIENT_ID]
    assert audit["missing"]["selected_without_mri"] == [SECOND_PATIENT_ID]
    assert audit["missing"]["selected_without_wsi"] == [SECOND_PATIENT_ID]
    assert audit["missing"]["mri_expected_files"] == {PATIENT_ID: ["T1_DL:roi"]}
    report = import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])
    assert report["counts"]["evidence"] == 26


def test_two_xlsx_only_imports_clinical_and_report(tmp_path):
    root = _make_source(
        tmp_path / "source",
        include_second_patient=False,
        include_mri_directory=False,
        include_wsi_directory=False,
    )

    report = import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])

    assert report["counts"]["evidence"] == 6
    assert report["inputs"]["mri_directory"]["exists"] is False
    assert report["inputs"]["wsi_directory"]["exists"] is False
    assert report["missing"]["missing_source_directories"] == [
        MRI_DIR.as_posix(),
        WSI_DIR.as_posix(),
    ]
    assert report["missing"]["selected_without_mri"] == [PATIENT_ID]
    assert report["missing"]["selected_without_wsi"] == [PATIENT_ID]
    assert report["artifacts"] == []
    assert validate_dataset(root) == []


def test_both_xlsx_inputs_remain_required(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _write_clinical(root / CLINICAL_XLSX, [_clinical_row(PATIENT_ID)])

    with pytest.raises(FileNotFoundError, match="缺少必需输入"):
        audit_srrsh_icc_multimodal(root)


@pytest.mark.parametrize(
    ("include_mri", "include_wsi", "expected_evidence"),
    [(True, False, 24), (False, True, 9)],
)
def test_each_optional_artifact_directory_is_independent(
    tmp_path, include_mri, include_wsi, expected_evidence
):
    root = _make_source(
        tmp_path / "source",
        include_second_patient=False,
        include_mri_directory=include_mri,
        include_wsi_directory=include_wsi,
    )

    report = import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])

    assert report["counts"]["evidence"] == expected_evidence
    assert report["inputs"]["mri_directory"]["exists"] is include_mri
    assert report["inputs"]["wsi_directory"]["exists"] is include_wsi


def test_optional_modality_path_must_be_a_directory(tmp_path):
    root = _make_source(
        tmp_path / "source",
        include_second_patient=False,
        include_mri_directory=False,
        include_wsi_directory=False,
    )
    (root / MRI_DIR).write_bytes(b"not-a-directory")

    with pytest.raises(NotADirectoryError, match="可选模态路径存在但不是目录"):
        audit_srrsh_icc_multimodal(root)


def test_multimodal_preserves_clinical_provenance(tmp_path):
    root = _make_source(tmp_path / "source", include_second_patient=False)
    clinical_out = tmp_path / "clinical"
    import_srrsh_icc(root / CLINICAL_XLSX, clinical_out, ids=[PATIENT_ID])
    import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])

    clinical = json.loads(
        (clinical_out / "provenance.jsonl").read_text(encoding="utf-8").strip()
    )
    multimodal = json.loads(
        (root / "provenance.jsonl").read_text(encoding="utf-8").strip()
    )
    for evidence_id in ("clin-1", "lab-1", "tx-tace-1", "tx-surg-1", "path-1"):
        assert multimodal["evidence"][evidence_id] == clinical["evidence"][evidence_id]


def test_audit_is_read_only(tmp_path):
    root = _make_source(tmp_path / "source")
    before = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    audit = audit_srrsh_icc_multimodal(root, ids=[PATIENT_ID])
    after = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert before == after
    assert audit["cohort"]["selected_patients"] == 1


def test_duplicate_ids_fail(tmp_path):
    root = _make_source(tmp_path / "source", include_second_patient=False)
    _write_clinical(root / CLINICAL_XLSX, [_clinical_row(PATIENT_ID), _clinical_row(PATIENT_ID)])
    with pytest.raises(ValueError, match="临床主表存在重复 ID"):
        audit_srrsh_icc_multimodal(root)

    _write_clinical(root / CLINICAL_XLSX, [_clinical_row(PATIENT_ID)])
    _write_reports(root / REPORT_XLSX, [(PATIENT_ID, "a", "b"), (PATIENT_ID, "c", "d")])
    with pytest.raises(ValueError, match="影像报告表存在重复住院号"):
        audit_srrsh_icc_multimodal(root)


def test_duplicate_mri_directory_and_path_escape_fail(tmp_path):
    root = _make_source(tmp_path / "source", include_second_patient=False)
    (root / MRI_DIR / f"{PATIENT_ID}_duplicate").mkdir()
    with pytest.raises(ValueError, match="MRI 目录存在重复患者 ID"):
        audit_srrsh_icc_multimodal(root)

    (root / MRI_DIR / f"{PATIENT_ID}_duplicate").rmdir()
    artifact = root / MRI_DIR / f"{PATIENT_ID}_synthetic" / "DWI.nii.gz"
    artifact.unlink()
    outside = tmp_path / "outside.nii.gz"
    outside.write_bytes(b"synthetic")
    artifact.symlink_to(outside)
    with pytest.raises(ValueError, match="artifact 路径逃逸"):
        audit_srrsh_icc_multimodal(root)


def test_source_out_match_targets_refused_and_cli_repeated_id(tmp_path):
    root = _make_source(tmp_path / "source")
    with pytest.raises(ValueError, match="--source 与 --out"):
        import_srrsh_icc_multimodal(root, tmp_path / "different")

    assert main([
        "import", "icc-multimodal", "--source", str(root), "--out", str(root),
        "--id", PATIENT_ID, "--id", PATIENT_ID,
    ]) == 0
    assert load_dataset(root).info.splits["all"] == [f"icc-{PATIENT_ID}"]

    (root / "targets.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="targets.jsonl"):
        import_srrsh_icc_multimodal(root, root)


class _ToyModel:
    name = "toy"
    config = {"offline": True}
    timeout = 1
    last_usage = None

    def complete(self, history, tools=None):
        del history, tools
        return {"role": "assistant", "content": json.dumps({
            "turn_id": "t1",
            "answer": {"summary": "Synthetic multimodal summary."},
            "citations": ["clin-1"],
            "reasoning_summary": "Synthetic offline test.",
        })}


def test_ama_run_does_not_decode_nifti_or_svs(tmp_path):
    root = _make_source(tmp_path / "source", include_second_patient=False)
    import_srrsh_icc_multimodal(root, root, ids=[PATIENT_ID])

    run_dir = execute(
        root,
        _ToyModel(),
        episode_ids=[f"icc-{PATIENT_ID}"],
        runs_root=tmp_path / "runs",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["termination"] == "completed"
    decisions = (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(decisions[0])["termination"] == "completed"
