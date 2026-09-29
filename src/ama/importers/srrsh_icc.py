"""SRRSH ICC 临床表 → AMA Dataset v0。

映射全部硬编码，不调用 LLM。每位患者生成一个单轮 Episode；该 Turn 只是
AMA Dataset v0 的结构性容器，不再按术前/术后拆分。每例仅输出以下 5 条
临床表 Evidence：临床基线、血清学、术前 TACE、手术记录、术后病理与分期。
MRI、WSI、结局和评分 target 均不在本 importer 范围内。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

SHEET_NAME = "ICC临床数据(合并版)"
HEADER_ROW_INDEX = 1
SOURCE_LOCATOR = "ICC临床数据.xlsx#主表!ID={id}"
TZ = timezone(timedelta(hours=8))

CLIN_FIELDS = [
    "Gender", "Age", "height", "weight", "BMI", "smoke", "alcohol",
    "HBV", "hypertension", "diabetes", "Cirrhosis", "child_pugh_grade",
]
LAB_FIELDS = ["AFP", "AST", "ALP", "GGT", "ALB", "BIL", "PT", "PLT", "INR"]
PATH_FIELDS = [
    "T", "N", "M", "Tumor_size", "tumor_large_vascular_invasion",
    "Tumor_number", "nerve_invasion", "MVI", "Grade",
]
REQUIRED_COLUMNS = set(
    CLIN_FIELDS + LAB_FIELDS + PATH_FIELDS
    + ["ID", "surg_time", "pathology", "TACE", "TACE_次数"]
)

ORIGIN_TABLE1 = "临床数据总表"
ORIGIN_TABLE2 = "表2"
TURN_MESSAGE = "ICC 临床表资料。"

GENDER_TEXT = {"Male": "男性", "Female": "女性"}
YES_NO_TEXT = {
    "smoke": {"Yes": "是", "No": "否"},
    "alcohol": {"Yes": "是", "No": "否"},
    "HBV": {"Yes": "阳性", "No": "阴性"},
    "hypertension": {"Yes": "有", "No": "无"},
    "diabetes": {"Yes": "有", "No": "无"},
    "Cirrhosis": {"Yes": "有", "No": "无"},
    "tumor_large_vascular_invasion": {"Yes": "有", "No": "无"},
    "nerve_invasion": {"Yes": "有", "No": "无"},
    "MVI": {"Yes": "有", "No": "无"},
}
GRADE_TEXT = {
    "Low": "低级别",
    "Low_medium": "低-中级别",
    "Medium": "中级别",
    "Medium_high": "中-高级别",
    "High": "高级别",
}
CHILD_PUGH_TEXT = {
    "A": "Child-Pugh 分级 A 级（肝功能储备评级）",
    "B": "Child-Pugh 分级 B 级（肝功能储备评级）",
}

MISSING_TEXT = "未提供"
NOT_EVALUATED_MARK = "（未评估，非阴性）"
BASIS_SOURCE_RECORDED = "source_recorded"
BASIS_DERIVED_SPECIMEN = "derived_specimen_from_surg_time"
BASIS_PROXY_PREOP = "proxy_preop_anchor_surg_day"
NOTE_CLIN = "状态型资料（年龄/习惯/合并症），截至手术日有效；具体采集日期源数据未提供"
NOTE_LAB = "真实采样日期源数据未提供"
NOTE_TACE_NEG = "阴性治疗史：'截至手术日未行 TACE' 有效，非缺失"
NOTE_PATH = "取材=手术日（由 surg_time 推导，真实）"
UNITS_NOTE_CLIN = "单位确定：身高 cm、体重 kg。"
DIAGNOSIS_CN = "肝内胆管细胞癌（ICC）"

DATASET_DESCRIPTION = "邵逸夫 ICC 临床表离线导入；每位患者 1 个单轮 Episode、5 条临床表 Evidence。"
DATASET_LICENSE = "内部研究使用；含患者相关临床数据，不可再分发；对外发布前必须重编码"


def _fmt(value: Any) -> str:
    """将 Excel 数字格式化为稳定文本，整数值不保留 .0。"""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def _value(value: Any, mapping: dict[str, str] | None = None) -> str:
    if value is None:
        return MISSING_TEXT
    if mapping is not None:
        return mapping.get(str(value), _fmt(value))
    return _fmt(value)


def _table_origin(row: dict[str, Any]) -> str:
    """T 非空的记录来自临床数据总表；否则来自合并表中的表2。"""
    return ORIGIN_TABLE1 if row.get("T") is not None else ORIGIN_TABLE2


def _parse_surg_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def _event_time(surg_date: date) -> str:
    return datetime.combine(surg_date, time.min, tzinfo=TZ).isoformat()


def _missing(row: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if row.get(field) is None]


def _add_missing(metadata: dict[str, Any], missing_fields: list[str]) -> None:
    if missing_fields:
        metadata["missing_fields"] = missing_fields


def build_clinical_baseline(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    age = row.get("Age")
    gender = _value(row.get("Gender"), GENDER_TEXT)
    if age is None:
        head = f"年龄{MISSING_TEXT}，{gender}"
    else:
        head = f"{_fmt(age)} 岁{gender}"

    text = (
        f"{head}。身高 {_value(row.get('height'))} cm，体重 {_value(row.get('weight'))} kg，"
        f"BMI {_value(row.get('BMI'))}。"
        f"吸烟：{_value(row.get('smoke'), YES_NO_TEXT['smoke'])}；"
        f"饮酒：{_value(row.get('alcohol'), YES_NO_TEXT['alcohol'])}。"
        f"HBV（乙肝病毒感染）：{_value(row.get('HBV'), YES_NO_TEXT['HBV'])}。"
        f"高血压：{_value(row.get('hypertension'), YES_NO_TEXT['hypertension'])}；"
        f"糖尿病：{_value(row.get('diabetes'), YES_NO_TEXT['diabetes'])}；"
        f"肝硬化：{_value(row.get('Cirrhosis'), YES_NO_TEXT['Cirrhosis'])}。"
    )
    child_pugh = row.get("child_pugh_grade")
    if child_pugh is None:
        text += f"Child-Pugh 分级：{MISSING_TEXT}。"
    else:
        text += CHILD_PUGH_TEXT.get(
            str(child_pugh), f"Child-Pugh 分级 {_fmt(child_pugh)}"
        ) + "。"

    metadata: dict[str, Any] = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": NOTE_CLIN,
        "table_origin": _table_origin(row),
        "fields": list(CLIN_FIELDS),
        "units_note": UNITS_NOTE_CLIN,
    }
    _add_missing(metadata, _missing(row, CLIN_FIELDS))
    return {
        "evidence_id": "clin-1",
        "kind": "clinical_baseline",
        "text": text,
        "source": ctx["locator"],
        "metadata": metadata,
    }


def build_lab_panel(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    values = "；".join(f"{field} {_value(row.get(field))}" for field in LAB_FIELDS)
    metadata: dict[str, Any] = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": NOTE_LAB,
    }
    _add_missing(metadata, _missing(row, LAB_FIELDS))
    return {
        "evidence_id": "lab-1",
        "kind": "lab_panel",
        "text": (
            "围术期血清学检验 panel（采样时点源数据未提供，按术前基线解释；"
            f"单位未随源数据提供）：{values}。"
        ),
        "source": ctx["locator"],
        "metadata": metadata,
    }


def build_tace(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    tace = row.get("TACE")
    count = row.get("TACE_次数")
    if str(tace) == "No":
        text = "术前未行 TACE（肝动脉化疗栓塞，术前辅助介入治疗）。"
        note = NOTE_TACE_NEG
    elif str(tace) == "Yes":
        count_text = "次数未提供" if count is None else f"次数 {_fmt(count)} 次"
        text = f"术前曾行 TACE（肝动脉化疗栓塞，术前辅助介入治疗），{count_text}。"
        note = "阳性治疗史：截至手术日曾行 TACE"
    elif tace is None:
        text = f"术前 TACE 治疗史：{MISSING_TEXT}。"
        note = "TACE 治疗史源数据未提供"
    else:
        text = f"术前 TACE 治疗史：{_fmt(tace)}。"
        note = "TACE 治疗史按源值保留"

    metadata: dict[str, Any] = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": note,
    }
    if tace is not None:
        metadata["tace"] = str(tace)
    _add_missing(metadata, _missing(row, ["TACE", "TACE_次数"]))
    return {
        "evidence_id": "tx-tace-1",
        "kind": "treatment_record",
        "text": text,
        "source": ctx["locator"],
        "metadata": metadata,
    }


def build_surgery(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": "tx-surg-1",
        "kind": "treatment_record",
        "text": f"患者于 {ctx['surg_date'].isoformat()} 接受手术治疗（术式源数据未收录）。",
        "source": ctx["locator"],
        "metadata": {
            "event_time": ctx["surg_iso"],
            "event_time_basis": BASIS_SOURCE_RECORDED,
            "missing_fields": ["术式"],
        },
    }


def _stage(value: Any, label: str) -> str:
    if value is None:
        return f"{label}：{MISSING_TEXT}"
    value_text = str(value)
    if value_text.lower() == f"{label.lower()}x":
        return f"{label}：{value_text}{NOT_EVALUATED_MARK}"
    return f"{label}：{value_text}"


def build_pathology(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    pathology = row.get("pathology")
    diagnosis = DIAGNOSIS_CN if str(pathology) == "ICC" else _value(pathology)
    grade = row.get("Grade")
    if grade is None:
        grade_text = MISSING_TEXT
    elif str(grade) in GRADE_TEXT:
        grade_text = f"{GRADE_TEXT[str(grade)]}（Grade: {grade}）"
    else:
        grade_text = _fmt(grade)

    t_stage = row.get("T")
    n_stage = row.get("N")
    m_stage = row.get("M")
    text = (
        f"术后病理：{diagnosis}。"
        f"肿瘤最大径 {_value(row.get('Tumor_size'))} cm；"
        f"肿瘤数目 {_value(row.get('Tumor_number'))}；"
        "大血管侵犯："
        f"{_value(row.get('tumor_large_vascular_invasion'), YES_NO_TEXT['tumor_large_vascular_invasion'])}；"
        f"神经侵犯：{_value(row.get('nerve_invasion'), YES_NO_TEXT['nerve_invasion'])}；"
        f"MVI（微血管侵犯）：{_value(row.get('MVI'), YES_NO_TEXT['MVI'])}；"
        f"组织学分级：{grade_text}。"
        f"TNM 分期：{_value(t_stage)}；{_stage(n_stage, 'N')}；{_stage(m_stage, 'M')}。"
    )
    metadata: dict[str, Any] = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_DERIVED_SPECIMEN,
        "event_time_note": NOTE_PATH,
    }
    _add_missing(metadata, _missing(row, PATH_FIELDS))
    not_evaluated = [
        label for label, value in (("N", n_stage), ("M", m_stage))
        if value is not None and str(value).lower() == f"{label.lower()}x"
    ]
    if not_evaluated:
        metadata["not_evaluated"] = not_evaluated
    return {
        "evidence_id": "path-1",
        "kind": "pathology_registry",
        "text": text,
        "source": ctx["locator"],
        "metadata": metadata,
    }


def build_episode(row: dict[str, Any]) -> dict[str, Any]:
    """将一行患者数据转换为包含 5 条 Evidence 的单轮 Episode。"""
    pid = _fmt(row["ID"])
    surg_date = _parse_surg_date(row["surg_time"])
    ctx = {
        "surg_date": surg_date,
        "surg_iso": _event_time(surg_date),
        "locator": SOURCE_LOCATOR.format(id=pid),
    }
    evidence = [
        build_clinical_baseline(row, ctx),
        build_lab_panel(row, ctx),
        build_tace(row, ctx),
        build_surgery(row, ctx),
        build_pathology(row, ctx),
    ]
    return {
        "episode_id": f"icc-{pid}",
        "subject_id": pid,
        "turns": [{"turn_id": "clinical", "message": TURN_MESSAGE, "evidence": evidence}],
    }


def _read_main_table(source: Path) -> list[dict[str, Any]]:
    wb = load_workbook(source, read_only=True, data_only=True)
    try:
        if SHEET_NAME not in wb.sheetnames:
            raise FileNotFoundError(f"sheet '{SHEET_NAME}' not in {source} (have {wb.sheetnames})")
        rows = list(wb[SHEET_NAME].iter_rows(values_only=True))
    finally:
        wb.close()

    header = [str(value).strip() if value is not None else "" for value in rows[HEADER_ROW_INDEX]]
    missing_columns = sorted(REQUIRED_COLUMNS - set(header))
    if missing_columns:
        raise ValueError(f"主表缺少必需列: {missing_columns}")
    data = [
        dict(zip(header, values))
        for values in rows[HEADER_ROW_INDEX + 1:]
        if any(value is not None for value in values)
    ]
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def import_srrsh_icc(
    source: Path,
    out: Path,
    ids: list[str] | None = None,
    dataset_name: str = "srrsh-icc",
) -> dict[str, Any]:
    """把 ICC 临床主表导入为无 target 的 AMA Dataset v0 目录。"""
    source = Path(source)
    out = Path(out)
    rows = _read_main_table(source)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {str(value).strip() for value in ids} if ids else None
    episodes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in sorted(rows, key=lambda item: _fmt(item["ID"])):
        pid = _fmt(row["ID"])
        if wanted is not None and pid not in wanted:
            continue
        if pid in seen_ids:
            skipped.append({"id": pid, "reason": "duplicate ID"})
            continue
        try:
            episode = build_episode(row)
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"id": pid, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        seen_ids.add(pid)
        episodes.append(episode)

    episode_ids = [episode["episode_id"] for episode in episodes]
    dataset = {
        "schema": "ama-dataset-v0",
        "name": dataset_name,
        "version": "0.1",
        "description": DATASET_DESCRIPTION,
        "splits": {"all": episode_ids},
        "scorer": "unscored",
        "license": DATASET_LICENSE,
    }
    _write_json(out / "dataset.json", dataset)
    (out / "episodes.jsonl").write_text(
        "".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in episodes),
        encoding="utf-8",
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sheet": SHEET_NAME,
        "n_rows": len(rows),
        "n_imported": len(episodes),
        "skipped": skipped,
        "ids_filter": sorted(wanted) if wanted is not None else None,
        "counts": {
            "episodes": len(episodes),
            "turns": len(episodes),
            "evidence": 5 * len(episodes),
        },
    }
    _write_json(out / "import_report.json", report)
    return report
