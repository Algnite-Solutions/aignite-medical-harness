"""ICC 临床表 importer 的合成数据单元测试。"""
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
    TURN_MESSAGE,
    _fmt,
    _table_origin,
    build_clinical_baseline,
    build_episode,
    build_lab_panel,
    build_pathology,
    build_surgery,
    build_tace,
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


EXPECTED_EVIDENCE = [
    {
        "evidence_id": "clin-1",
        "kind": "clinical_baseline",
        "text": (
            "50 岁女性。身高 165 cm，体重 60 kg，BMI 22。吸烟：否；饮酒：否。"
            "HBV（乙肝病毒感染）：阴性。高血压：无；糖尿病：无；肝硬化：无。"
            "Child-Pugh 分级 A 级（肝功能储备评级）。"
        ),
        "source": SYNTHETIC_SOURCE,
        "metadata": {
            "event_time": "2025-01-15T00:00:00+08:00",
            "event_time_basis": "proxy_preop_anchor_surg_day",
            "event_time_note": "状态型资料（年龄/习惯/合并症），截至手术日有效；具体采集日期源数据未提供",
            "table_origin": "临床数据总表",
            "fields": CLIN_FIELDS,
            "units_note": "单位确定：身高 cm、体重 kg。",
        },
    },
    {
        "evidence_id": "lab-1",
        "kind": "lab_panel",
        "text": (
            "围术期血清学检验 panel（采样时点源数据未提供，按术前基线解释；单位未随源数据提供）："
            "AFP 3.1；AST 20；ALP 60；GGT 25；ALB 42；BIL 12；PT 11；"
            "PLT 未提供；INR 未提供。"
        ),
        "source": SYNTHETIC_SOURCE,
        "metadata": {
            "event_time": "2025-01-15T00:00:00+08:00",
            "event_time_basis": "proxy_preop_anchor_surg_day",
            "event_time_note": "真实采样日期源数据未提供",
            "missing_fields": ["PLT", "INR"],
        },
    },
    {
        "evidence_id": "tx-tace-1",
        "kind": "treatment_record",
        "text": "术前未行 TACE（肝动脉化疗栓塞，术前辅助介入治疗）。",
        "source": SYNTHETIC_SOURCE,
        "metadata": {
            "event_time": "2025-01-15T00:00:00+08:00",
            "event_time_basis": "proxy_preop_anchor_surg_day",
            "event_time_note": "阴性治疗史：'截至手术日未行 TACE' 有效，非缺失",
            "tace": "No",
            "missing_fields": ["TACE_次数"],
        },
    },
    {
        "evidence_id": "tx-surg-1",
        "kind": "treatment_record",
        "text": "患者于 2025-01-15 接受手术治疗（术式源数据未收录）。",
        "source": SYNTHETIC_SOURCE,
        "metadata": {
            "event_time": "2025-01-15T00:00:00+08:00",
            "event_time_basis": "source_recorded",
            "missing_fields": ["术式"],
        },
    },
    {
        "evidence_id": "path-1",
        "kind": "pathology_registry",
        "text": (
            "术后病理：肝内胆管细胞癌（ICC）。肿瘤最大径 3.2 cm；肿瘤数目 1；"
            "大血管侵犯：无；神经侵犯：无；MVI（微血管侵犯）：未提供；"
            "组织学分级：中级别（Grade: Medium）。TNM 分期：T1a；"
            "N：Nx（未评估，非阴性）；M：Mx（未评估，非阴性）。"
        ),
        "source": SYNTHETIC_SOURCE,
        "metadata": {
            "event_time": "2025-01-15T00:00:00+08:00",
            "event_time_basis": "derived_specimen_from_surg_time",
            "event_time_note": "取材=手术日（由 surg_time 推导，真实）",
            "missing_fields": ["MVI"],
            "not_evaluated": ["N", "M"],
        },
    },
]


def test_fmt_integral_and_decimal():
    assert _fmt(20.0) == "20"
    assert _fmt(3.1) == "3.1"


def test_build_clinical_baseline():
    assert build_clinical_baseline(_synthetic_row(), SYNTHETIC_CONTEXT) == EXPECTED_EVIDENCE[0]


def test_build_lab_panel():
    assert build_lab_panel(_synthetic_row(), SYNTHETIC_CONTEXT) == EXPECTED_EVIDENCE[1]


def test_build_tace():
    assert build_tace(_synthetic_row(), SYNTHETIC_CONTEXT) == EXPECTED_EVIDENCE[2]


def test_build_surgery():
    assert build_surgery(_synthetic_row(), SYNTHETIC_CONTEXT) == EXPECTED_EVIDENCE[3]


def test_build_pathology():
    assert build_pathology(_synthetic_row(), SYNTHETIC_CONTEXT) == EXPECTED_EVIDENCE[4]


def test_episode_has_one_structural_turn_and_five_evidence():
    episode = build_episode(_synthetic_row())
    assert episode == {
        "episode_id": f"icc-{TEST_PATIENT_ID}",
        "subject_id": TEST_PATIENT_ID,
        "turns": [
            {
                "turn_id": "clinical",
                "message": TURN_MESSAGE,
                "evidence": EXPECTED_EVIDENCE,
            }
        ],
    }


def test_missing_and_not_evaluated_are_distinct():
    evidence = build_episode(_synthetic_row())["turns"][0]["evidence"]
    lab = evidence[1]
    tace = evidence[2]
    pathology = evidence[4]
    assert f"PLT {MISSING_TEXT}" in lab["text"]
    assert tace["metadata"]["tace"] == "No"
    assert f"N：Nx{NOT_EVALUATED_MARK}" in pathology["text"]
    assert pathology["metadata"]["not_evaluated"] == ["N", "M"]


def test_table_origin_rule():
    assert _table_origin(_synthetic_row()) == ORIGIN_TABLE1
    assert _table_origin(_synthetic_row(T=None)) == ORIGIN_TABLE2


def test_importer_uses_only_synthetic_local_workbook(tmp_path):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "dataset"
    _write_synthetic_workbook(source, [_synthetic_row()])

    report = import_srrsh_icc(source, out, ids=[TEST_PATIENT_ID])

    assert report["counts"] == {"episodes": 1, "turns": 1, "evidence": 5}
    assert report["skipped"] == []
    generated = json.loads((out / "episodes.jsonl").read_text(encoding="utf-8"))
    assert generated["turns"][0]["evidence"] == EXPECTED_EVIDENCE
    assert validate_dataset(out) == []


def test_importer_removes_stale_out_of_scope_files(tmp_path):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "dataset"
    out.mkdir()
    for stale_name in ("targets.jsonl", "DATASET_CARD.md", "DATASET_CARD.zh-CN.md"):
        (out / stale_name).write_text("synthetic stale file", encoding="utf-8")
    rows = [
        _synthetic_row(),
        _synthetic_row(ID="synthetic-002", surg_time="2025-02-20", T=None),
    ]
    _write_synthetic_workbook(source, rows)

    report = import_srrsh_icc(source, out)

    assert report["counts"] == {"episodes": 2, "turns": 2, "evidence": 10}
    assert validate_dataset(out) == []
    assert sorted(path.name for path in out.iterdir()) == [
        "dataset.json",
        "episodes.jsonl",
        "import_report.json",
    ]


def test_cli_imports_synthetic_workbook(tmp_path, capsys):
    source = tmp_path / "synthetic_icc.xlsx"
    out = tmp_path / "dataset"
    _write_synthetic_workbook(source, [_synthetic_row()])

    exit_code = main([
        "import",
        "icc-xlsx",
        "--source",
        str(source),
        "--out",
        str(out),
        "--ids",
        TEST_PATIENT_ID,
    ])

    assert exit_code == 0
    assert validate_dataset(out) == []
    assert "imported ->" in capsys.readouterr().out
