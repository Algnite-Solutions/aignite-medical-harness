import json

import pytest

from ama.agent import Agent
from ama.cli import _eval_run
from ama.decisions import normalize_decision, output_metrics
from ama.importers.mimic_cdm import import_mimic_cdm
from ama.runner import execute
from ama.tools import Tool, ToolResult
from fakes import FakeModel, calls, call
from test_cli import dataset, rows, raw_reply
from test_mimic_cdm import _source


def final(answer=None, **changes):
    return {"turn_id": "t1", "answer": answer or {"diagnosis": "pancreatitis"},
            "citations": ["hpi"], "reasoning_summary": "The findings support the diagnosis.", **changes}


@pytest.mark.parametrize("template", ["{}", "Analysis\n{}", "{}\nExplanation", "```json\n{}\n```"])
def test_json_positions_and_fences(template):
    value = final()
    raw = template.format(json.dumps(value))
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed == value
    assert checks["strict_compliance"] == (template == "{}")


def test_complete_decision_wins_and_labeled_summary_is_recovered():
    value = final()
    del value["reasoning_summary"]
    raw = json.dumps(value) + '\n\nReasoning summary: Findings agree. Alternatives lack support.\n\nDecision: {"diagnosis":"pancreatitis"}'
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed["citations"] == ["hpi"]
    assert parsed["reasoning_summary"] == "Findings agree. Alternatives lack support."
    assert checks["recovered_fields"] == {"reasoning_summary": "labeled_block"}
    assert not checks["strict_compliance"]


def test_answer_only_does_not_invent_citations_or_summary():
    parsed, checks = normalize_decision('Let me analyze.\n{"diagnosis":"pancreatitis"}', "t1", {"hpi"})
    assert parsed["answer"] == {"diagnosis": "pancreatitis"}
    assert parsed["citations"] == [] and parsed["reasoning_summary"] == ""
    assert checks["citations"] == "missing" and not checks["summary_present"]
    assert checks["format"] == "answer_only"


@pytest.mark.parametrize("suffix", [
    '{"diagnosis":"appendicitis"}',
    json.dumps(final(citations=["other"])),
    json.dumps(final(turn_id="wrong")),
])
def test_conflicting_or_wrong_turn_decisions_are_rejected(suffix):
    with pytest.raises(ValueError):
        normalize_decision(json.dumps(final()) + "\n" + suffix, "t1", {"hpi"})


def test_invalid_citations_do_not_erase_answer_or_fall_back_to_shorter_answer():
    raw = json.dumps(final(citations=["imaginary"])) + '\nDecision: {"diagnosis":"pancreatitis"}'
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed["citations"] == ["imaginary"]
    assert checks["citations"] == "invalid" and checks["unknown_citation_ids"] == ["imaginary"]
    assert checks["format"] == "embedded"


def test_only_successful_explicit_tool_metadata_releases_evidence(tmp_path):
    valid = Tool("valid", "valid", {}, lambda: ToolResult("findings", evidence_ids=["lab-1"]))
    catalog = Tool("catalog", "catalog", {}, lambda: {"evidence_id": "untrusted", "report_id": "scan-1"})
    failed = Tool("failed", "failed", {}, lambda: ToolResult("findings", [tmp_path / "absent.png"], ["bad"]))
    agent = Agent(FakeModel(calls(call("valid", "{}", "a"), call("catalog", "{}", "b"),
                                  call("failed", "{}", "c")), "done", "followup"),
                  tools=[valid, catalog, failed])
    agent.chat("start")
    assert agent.evidence_ids == {"lab-1"}
    assert len(agent.evidence_releases) == 1
    assert "Citeable evidence IDs" in agent.history[2]["content"]
    agent.chat("continue")
    assert agent.evidence_ids == {"lab-1"}
    assert Agent(FakeModel(), tools=[valid]).evidence_ids == set()


def test_tool_evidence_persists_between_turns_but_not_episodes(tmp_path):
    folder = dataset(tmp_path, two=True)
    tools = tmp_path / "tools.py"
    tools.write_text('from ama.tools import Tool, ToolResult\n'
                     'TOOLS=[Tool("lab","lab",{},lambda:ToolResult("12",evidence_ids=["lab-1"]))]\n')
    model = FakeModel(calls(call("lab", "{}")), json.dumps(final(citations=["lab-1"])),
                      json.dumps(final(turn_id="t2", citations=["lab-1"])),
                      json.dumps(final(citations=["lab-1"])), json.dumps(final(turn_id="t2")))
    run = execute(folder, model, tools_path=tools, runs_root=tmp_path / "runs")
    records = rows(run)
    assert [row["output_validation"]["citations"] for row in records[:3]] == ["valid", "valid", "invalid"]
    assert len(model.requests) == 5
    assert any(row["kind"] == "evidence_release" for row in rows(run, "diagnostics.jsonl"))
    assert all(row["kind"] not in {"decision", "exchange"} for row in rows(run, "diagnostics.jsonl"))


def test_correct_diagnosis_scores_despite_invalid_citations(tmp_path):
    out = tmp_path / "processed"
    import_mimic_cdm(_source(tmp_path / "source"), out)
    raw = json.dumps(final(answer={"diagnosis": "appendicitis"}, citations=["unknown"]))
    model = FakeModel(raw)
    run = execute(out / "mimic_cdm_interactive", model, episode_ids=["1"], runs_root=tmp_path / "runs")
    assert raw_reply(run, rows(run)[0]) == raw and len(model.requests) == 1
    assert _eval_run(run) == 0
    aggregate = json.loads((run / "metrics.json").read_text())["scored"]["aggregate"]
    assert aggregate["diagnosis_accuracy"]["value"] == 1
    assert aggregate["output_quality"]["valid_citations"]["value"] == 0
    assert aggregate["output_quality"]["summary_present"]["value"] == 1


def test_empty_citations_and_legacy_metrics():
    parsed, checks = normalize_decision(json.dumps(final(citations=[])), "t1", {"hpi"})
    assert output_metrics([{"decision": parsed, "output_validation": checks}], 2)["valid_citations"]["value"] == 0
    assert output_metrics([{"decision": parsed}], None)["summary_present"] is None


def test_thinking_drafts_do_not_override_final_answer(tmp_path):
    draft = json.dumps(final(turn_id="current turn ID"))
    value = final(citations=["unknown"])
    raw = '<think>Template: ' + draft + '\n{"diagnosis":"appendicitis"}</think>\n' + json.dumps(value)
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed == value
    assert checks["ignored_reasoning_blocks"] == 1
    assert not checks["strict_compliance"]
    assert checks["unknown_citation_ids"] == ["unknown"]
    model = FakeModel(raw)
    run = execute(dataset(tmp_path), model, runs_root=tmp_path / "runs")
    assert raw_reply(run, rows(run)[0]) == raw
    assert rows(run)[0]["decision"] == value
    assert rows(run)[0]["model_calls"] == 1
    assert len(model.requests) == 2  # One call for each of the fixture's two turns.


def test_summary_cannot_be_recovered_from_thinking():
    value = final()
    del value["reasoning_summary"]
    raw = '<think>Reasoning summary: Private draft.</think>\n' + json.dumps(value)
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed["reasoning_summary"] == ""
    assert not checks["summary_present"]
    assert "reasoning_summary" not in checks["recovered_fields"]


@pytest.mark.parametrize("raw", [
    '<think>' + json.dumps(final()),
    '<think>' + json.dumps(final()) + '</think>',
    '</think>' + json.dumps(final()),
    '<think>draft</think>' + json.dumps(final(turn_id="wrong")),
    '<think>draft</think>' + json.dumps(final()) + '\n{"diagnosis":"appendicitis"}',
])
def test_reasoning_markers_do_not_mask_invalid_final_answers(raw):
    with pytest.raises(ValueError):
        normalize_decision(raw, "t1", {"hpi"})


def test_thinking_tag_literals_inside_json_are_preserved():
    value = final(reasoning_summary="Literal <think> and </think> text.")
    parsed, checks = normalize_decision(json.dumps(value), "t1", {"hpi"})
    assert parsed == value
    assert checks["strict_compliance"]
    assert checks["ignored_reasoning_blocks"] == 0


def test_multiple_and_nested_thinking_blocks():
    raw = '<think>draft<think>nested</think></think>\n<think>revision</think>\n' + json.dumps(final())
    parsed, checks = normalize_decision(raw, "t1", {"hpi"})
    assert parsed == final()
    assert checks["ignored_reasoning_blocks"] == 2
