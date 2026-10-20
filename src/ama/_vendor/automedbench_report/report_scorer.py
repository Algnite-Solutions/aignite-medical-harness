#!/usr/bin/env python3
"""AutoMedBench lightweight pair metrics; unused corpus backends removed.

See SOURCE.json for the pinned upstream source and local modifications.
"""

from __future__ import annotations

import re

OBSERVATION_SCHEMA = {
    "support_devices": {
        "description": "Support device, tube, or line mention",
        "positive": [
            r"\bendotracheal tube\b",
            r"\bet tube\b",
            r"\bng tube\b",
            r"\bnasogastric tube\b",
            r"\bcordis\b",
            r"\bright ij\b",
            r"\bij cordis\b",
            r"\btip of the endotracheal tube\b",
        ],
    },
    "no_acute_process": {
        "description": "Explicit no-acute-process impression",
        "positive": [
            r"\bno evidence of acute cardiopulmonary process\b",
            r"\bno acute cardiopulmonary process\b",
            r"\bno acute cardiopulmonary abnormalit(?:y|ies)\b",
            r"\bno acute intrathoracic process\b",
            r"\bno new abnormalities within the lungs\b",
            r"\bno radiographic findings to suggest pneumonia\b",
        ],
    },
    "clear_lungs": {
        "description": "Explicit clear lungs statement",
        "positive": [
            r"\blungs are clear\b",
            r"\blungs are fully expanded and clear\b",
            r"\bpleural surfaces are normal\b",
        ],
    },
    "low_lung_volumes": {
        "description": "Low lung volumes",
        "positive": [
            r"\blungs are low in volume\b",
            r"\blow lung volumes\b",
            r"\blower lung volumes\b",
            r"\blungs are low volume\b",
        ],
    },
    "consolidation": {
        "description": "Focal airspace consolidation or infiltrate",
        "positive": [
            r"\bfocal (airspace )?consolidation\b",
            r"\balveolar infiltrate\b",
            r"\bpneumonia\b",
            r"\bairspace consolidation\b",
        ],
        "negative": [
            r"\bno [^.]{0,120}\bconsolidation\b",
            r"\bno focal airspace consolidation\b",
            r"\bno focal consolidation\b",
            r"\bwithout [^.]{0,120}\bconsolidation\b",
            r"\bno radiographic findings to suggest pneumonia\b",
            r"\bno [^.]{0,120}\bpneumonia\b",
        ],
    },
    "pleural_effusion": {
        "description": "Pleural effusion",
        "positive": [r"\bpleural effusion[s]?\b"],
        "negative": [
            r"\bno [^.]{0,120}\bpleural effusion[s]?\b",
            r"\bno pleural effusions\b",
            r"\bwithout [^.]{0,120}\bpleural effusion[s]?\b",
        ],
    },
    "pneumothorax": {
        "description": "Pneumothorax",
        "positive": [r"\bpneumothorax\b", r"\bpneumothoraces\b"],
        "negative": [
            r"\bno [^.]{0,120}\b(?:pneumothorax|pneumothoraces)\b",
            r"\bwithout [^.]{0,120}\b(?:pneumothorax|pneumothoraces)\b",
        ],
    },
    "pulmonary_edema": {
        "description": "Pulmonary edema or fluid overload",
        "positive": [
            r"\bpulmonary edema\b",
            r"\bperihilar edema\b",
            r"\bfluid overload\b",
            r"\bedema\b",
        ],
        "negative": [
            r"\bno [^.]{0,120}\bpulmonary edema\b",
            r"\bno overt pulmonary edema\b",
        ],
    },
    "cardiomegaly": {
        "description": "Enlarged cardiac silhouette",
        "positive": [
            r"\bcardiac enlargement\b",
            r"\bheart size is moderately enlarged\b",
            r"\bcardiomegaly\b",
            r"\benlarged heart\b",
        ],
        "negative": [
            r"\bheart size is normal\b",
            r"\bheart is normal in size\b",
            r"\bcardiomediastinal [^.]{0,20} normal\b",
        ],
    },
    "adenopathy_or_mass": {
        "description": "Hilar/mediastinal adenopathy or masslike finding",
        "positive": [
            r"\badenopathy\b",
            r"\blymph node\b",
            r"\bhilar [^.]{0,20} evident\b",
            r"\bparatracheal stripe\b",
            r"\bparatracheal region\b",
            r"\bmediastinal lymph node\b",
            r"\bsubstantial mass in the right paratracheal region\b",
        ],
    },
    "granuloma_or_calcified_nodule": {
        "description": "Calcified granuloma or calcified nodule",
        "positive": [
            r"\bcalcified granuloma\b",
            r"\bcalcified pulmonary nodule\b",
            r"\bcalcified nodule\b",
            r"\bgranulomatous disease\b",
            r"\bgranuloma\b",
        ],
    },
    "post_surgical_changes": {
        "description": "Prior CABG/sternotomy/post-surgical changes",
        "positive": [
            r"\bsternotomy\b",
            r"\bcabg\b",
            r"\bpost-surgical changes\b",
            r"\bmediastinal wires\b",
            r"\bsurgical clips\b",
            r"\bstatus post median sternotomy\b",
        ],
    },
}

LABEL_ORDER = list(OBSERVATION_SCHEMA.keys())


SECTION_HEADER_PATTERN = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Z /(),-]{1,40})\s*:\s*",
    flags=re.MULTILINE,
)


SECTION_ALIASES = {
    "finding": "findings",
    "findings": "findings",
    "reference findings": "findings",
    "chest, two views": "findings",
    "findings and impression": "impression",
    "findings/impression": "impression",
    "findings/ impression": "impression",
    "impression": "impression",
    "conclusion": "impression",
}


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\r", "\n")
    text = re.sub(r"FINAL REPORT", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalize_text(text))


def split_report_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(SECTION_HEADER_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        raw_name = match.group(1).strip().lower()
        section_name = SECTION_ALIASES.get(raw_name, raw_name)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content and section_name not in sections:
            sections[section_name] = content
    return sections


def extract_findings_text(text: str, *, strict: bool = False) -> str:
    sections = split_report_sections(text)
    findings = sections.get("findings", "").strip()
    if findings:
        return findings
    if strict:
        raise ValueError("Report does not contain a non-empty FINDINGS section")
    return text.strip()


def extract_observations(text: str) -> dict[str, int]:
    norm = normalize_text(text)
    labels: dict[str, int] = {}
    for label, rule in OBSERVATION_SCHEMA.items():
        positive = any(re.search(pattern, norm) for pattern in rule["positive"])
        negative = any(
            re.search(pattern, norm) for pattern in rule.get("negative", [])
        )
        labels[label] = int(positive and not negative)
    return labels


def positive_labels(labels: dict[str, int]) -> list[str]:
    return [label for label in LABEL_ORDER if labels.get(label)]


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for index, token_b in enumerate(b, start=1):
            temp = dp[index]
            if token_a == token_b:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def score_report_pair(
    reference_text: str,
    prediction_text: str,
    reference_labels: dict[str, int] | None = None,
) -> dict:
    reference_text = extract_findings_text(reference_text, strict=False)
    prediction_text = extract_findings_text(prediction_text, strict=False)
    reference_labels = reference_labels or extract_observations(reference_text)
    predicted_labels = extract_observations(prediction_text)

    ref_positive = set(positive_labels(reference_labels))
    pred_positive = set(positive_labels(predicted_labels))

    tp = len(ref_positive & pred_positive)
    fp = len(pred_positive - ref_positive)
    fn = len(ref_positive - pred_positive)
    precision, recall, obs_f1 = _f1(tp, fp, fn)

    return {
        "observation_precision": precision,
        "observation_recall": recall,
        "observation_f1": obs_f1,
        "report_similarity": rouge_l_f1(reference_text, prediction_text),
        "label_exact_match": int(all(
            reference_labels.get(label, 0) == predicted_labels.get(label, 0)
            for label in LABEL_ORDER
        )),
        "reference_labels": reference_labels,
        "predicted_labels": predicted_labels,
        "reference_positive_labels": sorted(ref_positive),
        "predicted_positive_labels": sorted(pred_positive),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_char_count": len(reference_text),
        "predicted_char_count": len(prediction_text),
    }
