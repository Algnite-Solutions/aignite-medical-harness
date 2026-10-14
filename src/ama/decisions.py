"""Deterministic final-answer extraction and independent contract diagnostics."""
from __future__ import annotations

import json
import re

from .data import Decision

FIELDS = {"turn_id", "answer", "citations", "reasoning_summary"}


def output_metrics(rows: list[dict], version: int | None) -> dict:
    names = ("recoverable_answer", "strict_output_compliance", "valid_citations", "summary_present")
    if version != 2:
        return {"contract_version": version, **{name: None for name in names}}
    counts = dict.fromkeys(names, 0)
    for row in rows:
        checks = row.get("output_validation") or {}
        counts["recoverable_answer"] += row.get("decision") is not None
        counts["strict_output_compliance"] += checks.get("strict_compliance") is True
        counts["valid_citations"] += checks.get("citations") == "valid"
        counts["summary_present"] += checks.get("summary_present") is True
    return {"contract_version": version, **{
        name: {"num": count, "den": len(rows), "value": count / len(rows) if rows else None}
        for name, count in counts.items()}}


def _final_content(reply: str) -> tuple[str, int]:
    """Exclude marked reasoning, preserving tag literals inside JSON values."""
    decoder = json.JSONDecoder()
    tags = re.compile(r"</?think>")
    parts = []
    position = depth = blocks = 0
    while position < len(reply):
        if depth:
            tag = tags.search(reply, position)
            if tag is None:
                raise ValueError("unclosed <think> block in model reply")
            depth += 1 if tag.group() == "<think>" else -1
            position = tag.end()
            continue
        if reply.startswith("<think>", position):
            depth = 1
            blocks += 1
            parts.append("\n")
            position += len("<think>")
            continue
        if reply.startswith("</think>", position):
            raise ValueError("unmatched </think> marker in model reply")
        # A tag within a valid JSON string is data, not a reasoning boundary.
        if reply[position] in "{[":
            try:
                _, length = decoder.raw_decode(reply[position:])
            except ValueError:
                pass
            else:
                parts.append(reply[position:position + length])
                position += length
                continue
        parts.append(reply[position])
        position += 1
    if depth:
        raise ValueError("unclosed <think> block in model reply")
    return "".join(parts), blocks


def normalize_decision(reply: str, turn_id: str, visible: set[str]) -> tuple[dict, dict]:
    content, ignored_blocks = _final_content(reply)
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    while position < len(content):
        if content[position] not in "{[":
            position += 1
            continue
        try:
            value, length = decoder.raw_decode(content[position:])
        except ValueError:
            position += 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position += length
    if not objects:
        raise ValueError("no JSON answer object in model reply")

    envelopes = [obj for obj in objects if FIELDS & obj.keys()]
    bare = [obj for obj in objects if not FIELDS & obj.keys()]
    recovered = {}
    if envelopes:
        if any("turn_id" in obj and obj["turn_id"] != turn_id for obj in envelopes):
            raise ValueError("decision turn_id does not match current observation")
        if any("answer" not in obj for obj in envelopes):
            raise ValueError("incomplete Decision envelope without answer")
        complete = [obj for obj in envelopes if {"turn_id", "answer", "citations"} <= obj.keys()]
        candidate = (complete or envelopes)[0]
        for obj in envelopes:
            if obj["answer"] != candidate["answer"]:
                raise ValueError("conflicting conclusions in model reply")
            if any(key in obj and key in candidate and obj[key] != candidate[key]
                   for key in ("citations", "reasoning_summary")):
                raise ValueError("conflicting Decision metadata in model reply")
        answer = candidate["answer"]
        for obj in bare:
            if not isinstance(answer, dict) or not obj or any(
                    key not in answer or answer[key] != value for key, value in obj.items()):
                raise ValueError("conflicting conclusions in model reply")
        format_name = "embedded"
    else:
        candidate = {"answer": bare[0]}
        if not bare[0] or any(obj != bare[0] for obj in bare):
            raise ValueError("empty or conflicting answer objects in model reply")
        format_name = "answer_only"

    errors = []
    missing = sorted(FIELDS - candidate.keys())
    if "turn_id" not in candidate:
        recovered["turn_id"] = "current_turn"
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    extra = sorted(candidate.keys() - FIELDS)
    if extra:
        errors.append("extra fields: " + ", ".join(extra))

    submitted_citations = candidate.get("citations")
    citations = submitted_citations if isinstance(submitted_citations, list) and all(
        isinstance(item, str) and item.strip() for item in submitted_citations) else []
    unknown = sorted(set(citations) - visible)
    if "citations" not in candidate:
        citation_status = "missing"
    elif citations != submitted_citations or unknown or (candidate["answer"] is None and citations):
        citation_status = "invalid"
    else:
        citation_status = "valid" if citations else "empty"
    if citation_status in {"missing", "invalid"}:
        errors.append("citations: " + citation_status)

    summary = candidate.get("reasoning_summary", "")
    if "reasoning_summary" not in candidate:
        match = re.search(r"(?im)^\s*(?:\*\*)?Reasoning summary(?:\*\*)?\s*:\s*(?:\*\*)?", content)
        if match:
            summary = re.split(r"(?im)\n\s*(?:Decision\s*:|```|\{)", content[match.end():], maxsplit=1)[0].strip()
            if summary:
                recovered["reasoning_summary"] = "labeled_block"
    summary_present = isinstance(summary, str) and bool(summary.strip())
    if not summary_present:
        errors.append("reasoning_summary: missing or invalid")
        summary = ""

    try:
        raw = json.loads(reply)
    except ValueError:
        raw = None
    if raw == candidate:
        format_name = "strict" if not missing and not extra else "json_object"
    strict = (format_name == "strict" and summary_present and
              isinstance(submitted_citations, list) and citations == submitted_citations)
    decision = Decision(turn_id=turn_id, answer=candidate["answer"], citations=citations,
                        reasoning_summary=summary).model_dump(mode="json")
    return decision, {
        "format": format_name, "strict_compliance": strict,
        "ignored_reasoning_blocks": ignored_blocks,
        "citations": citation_status, "unknown_citation_ids": unknown,
        "submitted_citations": submitted_citations,
        "summary_present": summary_present, "missing_fields": missing,
        "recovered_fields": recovered, "errors": errors,
    }
