"""Transparent topic matching and applicability filters for guideline retrieval.

These lexical rules rank references; they do not establish clinical validity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re


TOPICS = {
    "appendicitis": ["appendicitis", "阑尾炎"],
    "cholecystitis": ["cholecystitis", "gallbladder inflammation", "胆囊炎"],
    "diverticulitis": ["diverticulitis", "憩室炎", "结肠憩室炎"],
    "pancreatitis": ["pancreatitis", "胰腺炎"],
    "chronic kidney disease": ["chronic kidney disease", "CKD", "慢性肾脏病", "慢性肾病"],
}
INTENTS = {
    "diagnosis": ["diagnos", "diagnostic", "differential", "诊断", "鉴别"],
    "investigation": ["imaging", "ultrasound", "computed tomography", "investigation", "laboratory", "检查"],
    "treatment": ["treat", "management", "therapy", "surgery", "治疗"],
}
_STOP = set("and or the a an of for in with about to on guideline guidelines clinical practice consensus recommendations management diagnosis diagnostic treatment investigation adult adults children pediatric acute chronic title abstract 诊断 检查 治疗 急性 慢性 成人 儿童".split())
_FIELDS = ("title", "keywords", "abstract", "key_points", "org")


def contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", text.casefold())) if phrase.isascii() else phrase in text


def text_values(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(text_values(item) for item in value)
    return ""


def topic_groups(query: str) -> list[list[str]]:
    # Backwards-compatible simple field prefixes are syntax, never search terms.
    text = re.sub(r"\b(?:TITLE_ABS|TITLE|ABSTRACT)\s*:", "", query, flags=re.I)
    if ":" in text or re.search(r"\b(?:AND|OR|NOT)\b|[\[\]{}]", text):
        raise ValueError("Use query for the disease/topic and search_expression for raw Europe PMC syntax")
    groups = [aliases for aliases in TOPICS.values() if any(contains(text, alias) for alias in aliases)]
    if groups:
        remainder = text
        for aliases in groups:
            for alias in sorted(aliases, key=len, reverse=True):
                remainder = re.sub(re.escape(alias), " ", remainder, flags=re.I)
        extras = [w for w in re.findall(r"[\w-]+", remainder.casefold()) if w not in _STOP]
        return groups + [[word] for word in extras]
    words = [w for w in re.findall(r"[\w-]+", text.casefold()) if w not in _STOP]
    if not words:
        raise ValueError("query must include a disease or meaningful topic")
    return [[word] for word in words]


def match_groups(text: str, groups: list[list[str]]) -> bool:
    return all(any(contains(text, alias) for alias in group) for group in groups)


def local_guidelines(query: str, path: Path) -> list[dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("title"), str) and isinstance(e.get("url"), str) for e in entries
    ):
        raise ValueError("Guideline index must contain records with title and url")
    groups = topic_groups(query)
    return [dict(entry, source="local_index") for entry in entries
            if match_groups(" ".join(text_values(entry.get(field)) for field in _FIELDS), groups)]


def expression(query: str, search_expression: str, year_to: int | None) -> str:
    groups = topic_groups(query)
    disease = " AND ".join("(" + " OR ".join(f'TITLE_ABS:"{alias}"' for alias in group if alias.isascii()) + ")"
                           if any(alias.isascii() for alias in group) else f'TITLE_ABS:"{group[0]}"' for group in groups)
    # Some society consensus statements are not tagged as publication-type guidelines.
    kinds = '(PUB_TYPE:"guideline" OR PUB_TYPE:"practice guideline" OR TITLE:guideline* OR TITLE:consensus OR TITLE:recommendation*)'
    result = f"({disease}) AND {kinds}"
    if search_expression:
        result += f" AND ({search_expression})"
    if year_to is not None:
        result += f" AND FIRST_PDATE:[1000-01-01 TO {year_to}-12-31]"
    return result


def classify(title: str, pub_types: list[str]) -> tuple[str, list[str]]:
    patterns = [
        ("reporting_guideline", r"\b(stard|prisma|consort|arrive)\b|reporting (guideline|standard)"),
        ("laboratory_protocol", r"laboratory protocol|experimental protocol|assays? for|autophagy|laboratory methods"),
        ("other", r"systematic review|literature review|bibliometric|adherence|appraisal of|quality of.*guideline|"
                  r"limitations of|evaluation of|agreement between|value of|based on|according to|"
                  r"from guidelines to practice|guideline protocol|core outcome set|strengthening the|"
                  r"survey|retrospective|observational study|prospective.*study"),
        ("consensus", r"consensus|position statement"),
        ("clinical_practice_guideline", r"guidelines?|recommendations?"),
    ]
    for kind, pattern in patterns:
        matched = re.search(pattern, title, re.I)
        if matched:
            excluded = {p.casefold() for p in pub_types} & {
                "letter", "comment", "editorial", "observational study", "multicenter study", "case reports"}
            if kind in {"clinical_practice_guideline", "consensus"} and excluded:
                return "other", ["non-guideline publication type: " + ", ".join(sorted(excluded))]
            return kind, [f"title: {matched.group()}"]
    if any(p.casefold() == "practice guideline" for p in pub_types):
        return "clinical_practice_guideline", ["publication_type: Practice Guideline"]
    return "other", ["No clinical guideline or consensus cue in title/publication types"]


def applicability(row: dict) -> dict:
    title = row["title"].casefold()
    populations = list(row.get("population", []))
    if not populations:
        if re.search(r"child|pediatric|paediatric|infant|neonat", title):
            populations.append("pediatric")
        if re.search(r"\badults?\b", title):
            populations.append("adult")
        if re.search(r"elderly|older (?:adult|patient)|geriatric", title):
            populations.append("older_adult")
    subtypes = list(row.get("subtypes", []))
    if not subtypes:
        subtypes = [s for s in ("autoimmune", "chronic", "acute") if contains(title, s)]
    text = " ".join(text_values(row.get(f)) for f in ("title", "abstract", "key_points"))
    intents = list(row.get("intents", [])) or [intent for intent, terms in INTENTS.items()
                                             if any(term in text.casefold() for term in terms)]
    severity = ["severe"] if contains(title, "severe") else []
    return {"population": populations or ["unspecified"], "subtypes": subtypes or ["unspecified"], "severity": severity,
            "intents": intents, "basis": "curated metadata and/or lexical title/abstract cues; not clinical validation"}


def rank_guidelines(rows: list[dict], query: str, *, population: str, subtype: str,
                    intent: str, year_to: int | None, max_results: int) -> list[dict]:
    groups = topic_groups(query)
    candidates = []
    for original in rows:
        row = dict(original)
        title = row.get("title", "")
        # Only title/abstract (plus explicitly curated keywords) establish topic relevance.
        text = " ".join(text_values(row.get(f)) for f in ("title", "abstract", "keywords"))
        if not match_groups(text, groups):
            continue
        disease_groups = [group for group in groups if group in TOPICS.values()] or groups
        if not match_groups(title + " " + text_values(row.get("keywords")), disease_groups):
            # A background mention in an unrelated guideline is not a disease-specific reference.
            continue
        kind, basis = classify(title, row.get("pub_type", []))
        if kind not in {"clinical_practice_guideline", "consensus"}:
            continue
        app = applicability(row)
        if population and population not in app["population"]:
            continue
        if subtype and subtype not in app["subtypes"]:
            continue
        if intent and intent not in app["intents"]:
            continue
        year_match = re.search(r"\b\d{4}\b", str(row.get("year", "")))
        year = int(year_match.group()) if year_match else 0
        if year_to is not None and (not year or year > year_to):
            continue
        row.update(document_type=kind, classification_basis=basis, applicability=app,
                   version={"publication_year": year or None, "latest_status": "not_verified", "year_to": year_to},
                   retrieved_at=datetime.now(timezone.utc).isoformat())
        title_match = match_groups(title, groups)
        # Specific subgroups are retained but not preferred over broad-scope references by default.
        special = ("pediatric" in app["population"] and "adult" not in app["population"]
                   or "older_adult" in app["population"]) if not population else False
        special = special or (not subtype and any(s in app["subtypes"] for s in ("autoimmune", "chronic"))
                              and not any("chronic kidney disease" in group for group in groups))
        special = special or (not subtype and bool(re.search(r"post[- ]?(?:ercp|colonoscopy)", title, re.I)))
        diagnostic_scope = int("diagnosis" in app["intents"]) if not intent else 0
        score = (int(title_match), -int(special), diagnostic_scope, year, int(row.get("source") == "local_index"))
        row["relevance_basis"] = "topic in title" if title_match else "topic in abstract/curated keywords"
        candidates.append((score, row))
    unique, aliases = {}, {}
    for score, row in sorted(candidates, key=lambda item: item[0], reverse=True):
        keys = {f"{field}:{row[field]}" for field in ("pmid", "pmcid", "doi") if row.get(field)}
        keys.add("url:" + row["url"].rstrip("/"))
        key = next((aliases[k] for k in sorted(keys) if k in aliases), sorted(keys)[0])
        if key in unique:
            # Preserve audited society/section metadata if the API copy ranked first.
            if row.get("source") == "local_index":
                for field in ("org", "org_source", "diagnostic_sections", "reference_note"):
                    if row.get(field):
                        unique[key][field] = row[field]
        else:
            unique[key] = row
        for alias in keys:
            aliases[alias] = key
    return list(unique.values())[:max_results]
