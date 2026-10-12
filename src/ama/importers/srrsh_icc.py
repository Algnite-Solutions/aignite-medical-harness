"""SRRSH ICC 临床表 → AMA Dataset（ama-dataset 契约）。

映射全部硬编码，不调用 LLM。每位患者生成一个单轮 Episode；该 Turn 只是
结构性容器，不按术前/术后拆分。每例输出 5 条临床表 Evidence（id/type/text）：
临床基线、血清学、术前 TACE、手术记录、术后病理与分期。
MRI 报告全文可选：传入 mri_source 时按住院号 join 为第 6 条 evidence
（mri-report-1，完整检查所见+诊断，检查日期未提供按术前解释）。
MRI 序列影像、WSI、结局和评分 target 仍不在本 importer 范围内。

ama-dataset 契约对 Evidence/Turn 禁止额外字段，因此可审计信息不放数据集本体：
- 逐条 Evidence 的 event_time 三件套 / 缺失清单 → provenance.jsonl（按 episode 一行）
- 源 hash / 疑值 / 时间代理规则 / 缺失统计 → import_report.json
两者均不被 run loader 读取。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from ..dataset_card import write_dataset_cards

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
TURN_ID = "t1"
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

DATASET_LICENSE = "内部研究使用；含患者相关临床数据，不可再分发；对外发布前必须重编码"

TIME_PROXY_NOTE = (
    "唯一真实时间 = surg_time（treatment 依据 source_recorded）；基线/检验/TACE 的 "
    "event_time 为手术日代理（proxy_preop_anchor_surg_day）；病理 event_time 为取材推导"
    "（derived_specimen_from_surg_time）。逐条三件套见 provenance.jsonl"
)

# MRI 报告表（ICC核磁影像报告.xlsx）join 用常量：单 sheet、表头在第 0 行、无检查日期列。
MRI_SHEET_NAME = "sheet1"
MRI_HEADER_ROW_INDEX = 0
MRI_ID_COLUMN = "住院号"
MRI_FINDINGS_COLUMN = "检查所见"
MRI_DIAGNOSIS_COLUMN = "诊断"
MRI_REQUIRED_COLUMNS = [MRI_ID_COLUMN, MRI_FINDINGS_COLUMN, MRI_DIAGNOSIS_COLUMN]
MRI_SOURCE_LOCATOR = "ICC核磁影像报告.xlsx#sheet1!住院号={id}"
MRI_EVIDENCE_ID = "mri-report-1"
MRI_EVIDENCE_TYPE = "radiology_report"
MRI_REPORT_PREFIX = "腹部 MRI 检查报告全文（检查日期源数据未提供，按术前检查解释）。"
MRI_TIME_PROXY_NOTE = (
    "MRI 检查日期源数据未提供，按术前检查解释，event_time 锚定手术日仅用于排序"
)
# 术后/随访语境关键词：命中只在 provenance 打标（postop_suspicion_keywords），不改文本。
POSTOP_SUSPICION_KEYWORDS = ["术后", "复发", "随访", "复查", "TACE后", "介入后"]
MRI_POLICY_NOTE = (
    "完整报告（检查所见+诊断）进 evidence（2026-10-07 决策）；诊断列=放射科医生结论，"
    "t1 诊断弱金标因此部分可见，评测侧知悉"
)


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


def build_clinical_baseline(row: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回 (evidence, provenance 片段)。"""
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

    provenance = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": NOTE_CLIN,
        "table_origin": _table_origin(row),
        "units_note": UNITS_NOTE_CLIN,
    }
    missing = _missing(row, CLIN_FIELDS)
    if missing:
        provenance["missing_fields"] = missing
    return {
        "id": "clin-1",
        "type": "clinical_baseline",
        "text": text,
    }, provenance


def build_lab_panel(row: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    values = "；".join(f"{field} {_value(row.get(field))}" for field in LAB_FIELDS)
    provenance = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": NOTE_LAB,
    }
    missing = _missing(row, LAB_FIELDS)
    if missing:
        provenance["missing_fields"] = missing
    return {
        "id": "lab-1",
        "type": "lab_panel",
        "text": (
            "围术期血清学检验 panel（采样时点源数据未提供，按术前基线解释；"
            f"单位未随源数据提供）：{values}。"
        ),
    }, provenance


def build_tace(row: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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

    provenance = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": note,
    }
    if tace is not None:
        provenance["tace"] = str(tace)
    missing = _missing(row, ["TACE", "TACE_次数"])
    if missing:
        provenance["missing_fields"] = missing
    return {
        "id": "tx-tace-1",
        "type": "treatment_record",
        "text": text,
    }, provenance


def build_mri_report(pid: str, report: dict[str, str], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """MRI 报告全文 → (evidence, provenance 片段)。检查所见/诊断保持源文本原样。"""
    findings = report.get(MRI_FINDINGS_COLUMN) or MISSING_TEXT
    diagnosis = report.get(MRI_DIAGNOSIS_COLUMN) or MISSING_TEXT
    evidence = {
        "id": MRI_EVIDENCE_ID,
        "type": MRI_EVIDENCE_TYPE,
        "text": f"{MRI_REPORT_PREFIX}\n检查所见：{findings}\n诊断：{diagnosis}",
    }
    provenance = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_PROXY_PREOP,
        "event_time_note": MRI_TIME_PROXY_NOTE,
        "source": MRI_SOURCE_LOCATOR.format(id=pid),
    }
    # 术后/随访语境仅打标（按常量列表顺序），供评测侧甄别，绝不改写或剔除文本。
    suspected = [
        keyword for keyword in POSTOP_SUSPICION_KEYWORDS
        if keyword in f"{findings}{diagnosis}"
    ]
    if suspected:
        provenance["postop_suspicion_keywords"] = suspected
    return evidence, provenance


def build_surgery(row: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return {
        "id": "tx-surg-1",
        "type": "treatment_record",
        "text": f"患者于 {ctx['surg_date'].isoformat()} 接受手术治疗（术式源数据未收录）。",
    }, {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_SOURCE_RECORDED,
        "missing_fields": ["术式"],
    }


def _stage(value: Any, label: str) -> str:
    if value is None:
        return f"{label}：{MISSING_TEXT}"
    value_text = str(value)
    if value_text.lower() == f"{label.lower()}x":
        return f"{label}：{value_text}{NOT_EVALUATED_MARK}"
    return f"{label}：{value_text}"


def build_pathology(row: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    provenance = {
        "event_time": ctx["surg_iso"],
        "event_time_basis": BASIS_DERIVED_SPECIMEN,
        "event_time_note": NOTE_PATH,
    }
    missing = _missing(row, PATH_FIELDS)
    if missing:
        provenance["missing_fields"] = missing
    not_evaluated = [
        label for label, value in (("N", n_stage), ("M", m_stage))
        if value is not None and str(value).lower() == f"{label.lower()}x"
    ]
    if not_evaluated:
        provenance["not_evaluated"] = not_evaluated
    return {
        "id": "path-1",
        "type": "pathology_registry",
        "text": text,
    }, provenance


def build_episode(
    row: dict[str, Any],
    mri_report: dict[str, str] | None = None,
    mri_source_loaded: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """一行患者 → (episode, provenance 行)。纯函数，无 IO。

    mri_report 提供时在 tx-tace-1 与 tx-surg-1 之间插入 MRI 报告 evidence；
    MRI 源已加载但该患者无报告时不加 evidence，仅在 provenance 标记缺失。
    两者都不提供时与历史行为完全一致。
    """
    pid = _fmt(row["ID"])
    surg_date = _parse_surg_date(row["surg_time"])
    ctx = {
        "surg_date": surg_date,
        "surg_iso": _event_time(surg_date),
        "locator": SOURCE_LOCATOR.format(id=pid),
    }
    built = [
        build_clinical_baseline(row, ctx),
        build_lab_panel(row, ctx),
        build_tace(row, ctx),
    ]
    if mri_report is not None:
        built.append(build_mri_report(pid, mri_report, ctx))
    built.append(build_surgery(row, ctx))
    built.append(build_pathology(row, ctx))
    evidence = [item[0] for item in built]
    episode = {
        "id": f"icc-{pid}",
        "turns": [{"id": TURN_ID, "observation": TURN_MESSAGE, "evidence": evidence}],
    }
    provenance_row = {
        "id": episode["id"],
        "source": ctx["locator"],
        "surg_time": surg_date.isoformat(),
        "table_origin": _table_origin(row),
        "evidence": {item[0]["id"]: item[1] for item in built},
    }
    if mri_report is None and mri_source_loaded:
        provenance_row["mri_report_missing"] = True
    return episode, provenance_row


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


def _read_mri_reports(source: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """读取 MRI 报告表 → (按住院号索引的报告 dict, 跳过清单)。

    报告文本保持源单元格原样（仅去外层空白，内部换行保留）；住院号经 _fmt
    格式化以与临床表 ID 一致。重复住院号保留首条，双列全空行跳过。
    """
    wb = load_workbook(source, read_only=True, data_only=True)
    try:
        if MRI_SHEET_NAME not in wb.sheetnames:
            raise FileNotFoundError(f"sheet '{MRI_SHEET_NAME}' not in {source} (have {wb.sheetnames})")
        rows = list(wb[MRI_SHEET_NAME].iter_rows(values_only=True))
    finally:
        wb.close()

    header = [str(value).strip() if value is not None else "" for value in rows[MRI_HEADER_ROW_INDEX]]
    missing_columns = sorted(set(MRI_REQUIRED_COLUMNS) - set(header))
    if missing_columns:
        raise ValueError(f"MRI 报告表缺少必需列: {missing_columns}")

    def _text(value: Any) -> str | None:
        return str(value).strip() if value is not None else None

    reports: dict[str, dict[str, str]] = {}
    skipped: list[dict[str, Any]] = []
    for values in rows[MRI_HEADER_ROW_INDEX + 1:]:
        if all(value is None for value in values):
            continue
        record = dict(zip(header, values))
        pid = _fmt(record[MRI_ID_COLUMN])
        if pid in reports:
            skipped.append({"id": pid, "reason": "duplicate MRI report ID"})
            continue
        findings = _text(record.get(MRI_FINDINGS_COLUMN))
        diagnosis = _text(record.get(MRI_DIAGNOSIS_COLUMN))
        if not findings and not diagnosis:
            skipped.append({"id": pid, "reason": "empty MRI report"})
            continue
        reports[pid] = {MRI_FINDINGS_COLUMN: findings, MRI_DIAGNOSIS_COLUMN: diagnosis}
    return reports, skipped


def import_srrsh_icc(
    source: Path,
    out: Path,
    ids: list[str] | None = None,
    mri_source: Path | None = None,
) -> dict[str, Any]:
    """把 ICC 临床主表导入为无 target 的 ama-dataset 目录。

    mri_source 提供时按住院号 join MRI 报告全文，作为第 6 条 evidence。
    """
    source = Path(source)
    out = Path(out)
    rows = _read_main_table(source)
    out.mkdir(parents=True, exist_ok=True)

    mri_reports: dict[str, dict[str, str]] = {}
    mri_skipped: list[dict[str, Any]] = []
    if mri_source is not None:
        mri_source = Path(mri_source)
        mri_reports, mri_skipped = _read_mri_reports(mri_source)

    wanted = {str(value).strip() for value in ids} if ids else None
    episodes: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    mri_matched = 0
    patients_without_report: list[str] = []
    postop_keyword_counts: dict[str, int] = {}

    for row in sorted(rows, key=lambda item: _fmt(item["ID"])):
        pid = _fmt(row["ID"])
        if wanted is not None and pid not in wanted:
            continue
        if pid in seen_ids:
            skipped.append({"id": pid, "reason": "duplicate ID"})
            continue
        try:
            mri_report = mri_reports.get(pid)
            episode, provenance_row = build_episode(
                row,
                mri_report=mri_report,
                mri_source_loaded=mri_source is not None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"id": pid, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        seen_ids.add(pid)
        episodes.append(episode)
        provenance.append(provenance_row)
        if mri_report is not None:
            mri_matched += 1
            for keyword in provenance_row["evidence"][MRI_EVIDENCE_ID].get(
                "postop_suspicion_keywords", []
            ):
                postop_keyword_counts[keyword] = postop_keyword_counts.get(keyword, 0) + 1
        elif mri_source is not None:
            patients_without_report.append(pid)

    episode_ids = [episode["id"] for episode in episodes]
    (out / "dataset.json").write_text(
        json.dumps({"schema": "ama-dataset", "name": out.name,
                    "splits": {"all": episode_ids}}, indent=2),
        encoding="utf-8",
    )
    (out / "episodes.jsonl").write_text(
        "".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provenance),
        encoding="utf-8",
    )

    tracked_columns = CLIN_FIELDS + LAB_FIELDS + PATH_FIELDS + ["TACE", "TACE_次数"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sheet": SHEET_NAME,
        "n_rows": len(rows),
        "n_imported": len(episodes),
        "skipped": skipped,
        "ids_filter": sorted(wanted) if wanted is not None else None,
        "table_origin": {
            "rule": "T 列非空 → 临床数据总表（表1，188 例）；否则表2（68 例）",
            "counts": {
                ORIGIN_TABLE1: sum(1 for p in provenance if p["table_origin"] == ORIGIN_TABLE1),
                ORIGIN_TABLE2: sum(1 for p in provenance if p["table_origin"] == ORIGIN_TABLE2),
            },
        },
        "time_proxy": TIME_PROXY_NOTE,
        "outcome_columns": {
            "policy": "RFS/OS/随访列不读取、不进 episodes/provenance（评测侧负责 targets）",
        },
        "missing_stats": {
            "n_rows": len(rows),
            "nonnull_by_column": {c: sum(1 for r in rows if r.get(c) is not None)
                                  for c in tracked_columns},
        },
        "counts": {
            "episodes": len(episodes),
            "turns": len(episodes),
            "evidence": 5 * len(episodes) + mri_matched,
        },
    }
    if mri_source is not None:
        unmatched_report_ids = [pid for pid in mri_reports if pid not in seen_ids]
        report["mri_report"] = {
            "source_file": str(mri_source),
            "source_sha256": hashlib.sha256(mri_source.read_bytes()).hexdigest(),
            "sheet": MRI_SHEET_NAME,
            "n_rows": len(mri_reports) + len(mri_skipped),
            "policy": MRI_POLICY_NOTE,
            "time_proxy": MRI_TIME_PROXY_NOTE,
            "matched": mri_matched,
            "patients_without_report": {
                "count": len(patients_without_report),
                "ids": patients_without_report,
            },
            "unmatched_report_ids": {
                "count": len(unmatched_report_ids),
                "ids": unmatched_report_ids,
            },
            "postop_keyword_counts": postop_keyword_counts,
            "skipped": mri_skipped,
        }
    (out / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mri_source is None:
        purpose_en = ("SRRSH ICC clinical-table import: one single-turn episode per patient with "
                      "5 table-derived evidence items (baseline / lab panel / TACE history / "
                      "surgery / pathology).")
        purpose_zh = "邵逸夫 ICC 临床表导入：每位患者 1 个单轮 Episode、5 条临床表 Evidence。"
        construction_en = ("Hard-coded column mapping, no LLM. Outcome columns (RFS/OS/follow-up) "
                           "are never read; turn design and targets belong to the evaluation side. "
                           "Per-evidence timing and missingness go to provenance.jsonl, "
                           "which the run loader never reads.")
        construction_zh = ("列→Evidence 映射硬编码，无 LLM。结局列（RFS/OS/随访）不读取；"
                           "turn 划分与评测 target 属评测侧。逐条时间与缺失信息在 provenance.jsonl，"
                           "run loader 不读取。")
        limitations_en = ("Time anchors are declared proxies (only surg_time is real). Lab values "
                          "carry no units (source provided none). Output contains real patient "
                          "identifiers: keep out of the repository and do not redistribute before "
                          "re-encoding.")
        limitations_zh = ("时间锚为声明的代理（唯一真实时间=surg_time）。检验值不带单位（源数据未提供）。"
                          "产物含真实患者标识：不入仓库、再分发前必须重编码。")
    else:
        purpose_en = ("SRRSH ICC clinical-table import with joined MRI reports: one single-turn "
                      "episode per patient with 6 evidence items where available (patients with "
                      "a matched MRI report get the full radiology_report text; the rest keep "
                      "the 5 table-derived items).")
        purpose_zh = ("邵逸夫 ICC 临床表 + MRI 报告导入：每位患者 1 个单轮 Episode、"
                      "6 条 evidence（有 MRI 报告的患者含影像报告全文 radiology_report）。")
        construction_en = ("Hard-coded column mapping, no LLM. Outcome columns (RFS/OS/follow-up) "
                           "are never read; turn design and targets belong to the evaluation side. "
                           "MRI findings and impression are joined per patient from "
                           "ICC核磁影像报告.xlsx and included verbatim as a radiology_report "
                           "evidence item. Per-evidence timing and missingness go to "
                           "provenance.jsonl, which the run loader never reads.")
        construction_zh = ("列→Evidence 映射硬编码，无 LLM。结局列（RFS/OS/随访）不读取；"
                           "turn 划分与评测 target 属评测侧。MRI 报告（检查所见+诊断）按住院号 "
                           "join、原文进入 radiology_report evidence。逐条时间与缺失信息在 "
                           "provenance.jsonl，run loader 不读取。")
        limitations_en = ("Time anchors are declared proxies (only surg_time is real). Lab values "
                          "carry no units (source provided none). MRI reports carry no exam date "
                          "(treated as pre-operative proxy) and contain radiologist conclusions "
                          "(诊断列), partially visible to the t1 diagnosis weak gold standard. "
                          "Output contains real patient identifiers: keep out of the repository "
                          "and do not redistribute before re-encoding.")
        limitations_zh = ("时间锚为声明的代理（唯一真实时间=surg_time）。检验值不带单位（源数据未提供）。"
                          "MRI 报告无检查日期，按术前检查代理；报告含放射科结论（诊断列），"
                          "t1 诊断弱金标部分可见。产物含真实患者标识：不入仓库、再分发前必须重编码。")
    write_dataset_cards(
        out,
        name=out.name,
        source=f"{source}#{SHEET_NAME}",
        license_name=DATASET_LICENSE,
        purpose_en=purpose_en,
        purpose_zh=purpose_zh,
        construction_en=construction_en,
        construction_zh=construction_zh,
        episodes=len(episodes), turns=len(episodes),
        evidence=5 * len(episodes) + mri_matched,
        scorer="none", targets=0,
        limitations_en=limitations_en,
        limitations_zh=limitations_zh,
    )
    return report
