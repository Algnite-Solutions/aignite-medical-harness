"""ICC 临床表 importer 的合成数据单元测试（ama-dataset 契约）。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from openpyxl import Workbook

from ama.cli import main
from ama.data import validate_dataset
from ama.importers.srrsh_icc import (
    CLIN_FIELDS,
    LAB_FIELDS,
    MISSING_TEXT,
    NOT_EVALUATED_MARK,
    ORIGIN_TABLE1,
    ORIGIN_TABLE2,
    PATH_FIELDS,
    SHEET_NAME,
    TURN_ID,
    TURN_MESSAGE,
    _fmt,
    _table_origin,
    build_episode,
    build_pathology,
    import_srrsh_icc,
)

TEST_PATIENT_ID = "2272542"
SYNTHETIC_SOURCE = f"ICC临床数据.xlsx#主表!ID={TEST_PATIENT_ID}"
SYNTHETIC_CONTEXT = {
    "surg_date": date(2025, 1, 15),
    "surg_iso": "2025-01-15T00:00:00+08:00",
    "locator": SYNTHETIC_SOURCE,
}
HEADERS = [
    "ID",
    "surg_time",
    "pathology",
    *CLIN_FIELDS,
    *LAB_FIELDS,
    *PATH_FIELDS,
    "TACE",
    "TACE_次数",
]


def _synthetic_row(**overrides) -> dict:
    row = {
        "ID": TEST_PATIENT_ID,
        "surg_time": "2025-01-15",
        "pathology": "ICC",
        "Gender": "Female",
        "Age": 50,
        "height": 165,
        "weight": 60,
        "BMI": 22,
        "smoke": "No",
        "alcohol": "No",
        "HBV": "No",
        "hypertension": "No",
        "diabetes": "No",
        "Cirrhosis": "No",
        "child_pugh_grade": "A",
        "AFP": 3.1,
        "AST": 20,
        "ALP": 60,
        "GGT": 25,
        "ALB": 42,
        "BIL": 12,
        "PT": 11,
        "PLT": None,
        "INR": None,
        "T": "T1a",
        "N": "Nx",
        "M": "Mx",
        "Tumor_size": 3.2,
        "tumor_large_vascular_invasion": "No",
        "Tumor_number": 1,
        "nerve_invasion": "No",
        "MVI": None,
        "Grade": "Medium",
        "TACE": "No",
        "TACE_次数": None,
    }
    row.update(overrides)
    return row


def _write_synthetic_workbook(path: Path, rows: list[dict]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_NAME
    sheet.append(["synthetic test data", *([None] * (len(HEADERS) - 1))])
    sheet.append(HEADERS)
    for row in rows:
        sheet.append([row.get(header) for header in HEADERS])
    workbook.save(path)
    workbook.close()


EXPECTED_EVIDENCE_TEXTS = {
    "clin-1": (
        "50 岁女性。身高 165 cm，体重 60 kg，BMI 22。吸烟：否；饮酒：否。"
        "HBV（乙肝病毒感染）：阴性。高血压：无；糖尿病：无；肝硬化：无。"
        "Child-Pugh 分级 A 级（肝功能储备评级）。"
    ),
    "lab-1": (
        "围术期血清学检验 panel（采样时点源数据未提供，按术前基线解释；单位未随源数据提供）："
        "AFP 3.1；AST 20；ALP 60；GGT 25；ALB 42；BIL 12；PT 11；"
        "PLT 未提供；INR 未提供。"
    ),
    "tx-tace-1": "术前未行 TACE（肝动脉化疗栓塞，术前辅助介入治疗）。",
    "tx-surg-1": "患者于 2025-01-15 接受手术治疗（术式源数据未收录）。",
    "path-1": (
        "术后病理：肝内胆管细胞癌（ICC）。肿瘤最大径 3.2 cm；肿瘤数目 1；"
        "大血管侵犯：无；神经侵犯：无；MVI（微血管侵犯）：未提供；"
        "组织学分级：中级别（Grade: Medium）。TNM 分期：T1a；"
        "N：Nx（未评估，非阴性）；M：Mx（未评估，非阴性）。"
    ),
}
EXPECTED_EVIDENCE_TYPES = {
    "clin-1": "clinical_baseline",
    "lab-1": "lab_panel",
    "tx-tace-1": "treatment_record",
    "tx-surg-1": "treatment_record",
    "path-1": "pathology_registry",
}


def test_fmt_integral_and_decimal():
    assert _fmt(20.0) == "20"
    assert _fmt(3.1) == "3.1"


def test_episode_is_single_structural_turn_with_five_evidence():
    episode, provenance = build_episode(_synthetic_row())
    assert episode == {
        "id": f"icc-{TEST_PATIENT_ID}",
        "turns": [{
            "id": TURN_ID,
            "observation": TURN_MESSAGE,
            "evidence": [
                {"id": evidence_id, "type": EXPECTED_EVIDENCE_TYPES[evidence_id],
                 "text": EXPECTED_EVIDENCE_TEXTS[evidence_id]}
                for evidence_id in
                ("clin-1", "lab-1", "tx-tace-1", "tx-surg-1", "path-1")
            ],
        }],
    }


def test_provenance_row_carries_timing_and_source():
    _, provenance = build_episode(_synthetic_row())
    assert provenance["source"] == SYNTHETIC_SOURCE
    assert provenance["table_origin"] == ORIGIN_TABLE1
    assert provenance["surg_time"] == "2025-01-15"
    assert provenance["evidence"]["clin-1"]["event_time_basis"] == "proxy_preop_anchor_surg_day"
    assert provenance["evidence"]["tx-surg-1"]["event_time_basis"] == "source_recorded"
    assert provenance["evidence"]["path-1"]["event_time_basis"] == "derived_specimen_from_surg_time"


def test_missing_and_not_evaluated_are_distinct():
    episode, provenance = build_episode(_synthetic_row())
    evidence = {item["id"]: item for item in episode["turns"][0]["evidence"]}
    assert f"PLT {MISSING_TEXT}" in evidence["lab-1"]["text"]
    assert f"N：Nx{NOT_EVALUATED_MARK}" in evidence["path-1"]["text"]
    assert provenance["evidence"]["lab-1"]["missing_fields"] == ["PLT", "INR"]
    assert provenance["evidence"]["path-1"]["not_evaluated"] == ["N", "M"]


def test_table_origin_rule():
    assert _table_origin(_synthetic_row()) == ORIGIN_TABLE1
    assert _table_origin(_synthetic_row(T=None)) == ORIGIN_TABLE2


def test_importer_outputs_contract_files_and_validates(tmp_path):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "srrsh_icc"
    _write_synthetic_workbook(source, [
        _synthetic_row(),
        _synthetic_row(ID="90020002", surg_time="2025-02-20", T=None),
    ])

    report = import_srrsh_icc(source, out)

    assert report["counts"] == {"episodes": 2, "turns": 2, "evidence": 10}
    assert report["skipped"] == []
    assert report["table_origin"]["counts"] == {ORIGIN_TABLE1: 1, ORIGIN_TABLE2: 1}
    assert validate_dataset(out) == []

    dataset = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["schema"] == "ama-dataset"
    assert dataset["name"] == "srrsh_icc"
    assert len(dataset["splits"]["all"]) == 2

    provenance_rows = (out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(provenance_rows) == 2
    assert not (out / "targets.jsonl").exists()


def test_importer_never_reads_outcome_columns(tmp_path):
    """结局列不在 REQUIRED_COLUMNS：缺列的工作簿照常导入，episodes 不含结局信息。"""
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "srrsh_icc"
    _write_synthetic_workbook(source, [_synthetic_row()])

    import_srrsh_icc(source, out)

    episodes_text = (out / "episodes.jsonl").read_text(encoding="utf-8")
    for column in ("RFS_status", "RFS_time", "OS_status", "OS_time", "last_followup"):
        assert column not in episodes_text


def test_duplicate_ids_are_skipped_and_reported(tmp_path):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "srrsh_icc"
    _write_synthetic_workbook(source, [
        _synthetic_row(),
        _synthetic_row(surg_time="2025-03-01"),
    ])

    report = import_srrsh_icc(source, out)

    assert report["counts"]["episodes"] == 1
    assert report["skipped"] == [{"id": TEST_PATIENT_ID, "reason": "duplicate ID"}]
    assert validate_dataset(out) == []


def test_cli_imports_synthetic_workbook(tmp_path, capsys):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "srrsh_icc"
    _write_synthetic_workbook(source, [_synthetic_row()])

    exit_code = main([
        "import",
        "icc-xlsx",
        "--source",
        str(source),
        "--out",
        str(out),
        "--id",
        TEST_PATIENT_ID,
    ])

    assert exit_code == 0
    assert validate_dataset(out) == []
    assert "imported ->" in capsys.readouterr().out
