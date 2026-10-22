"""Opt-in retrieval endpoint and DeepSeek function-calling smoke tests."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pytest

from ama.agent import Agent
from ama.cli import _load_dotenv
from ama.model import Model
from ama.tools import load_tools
from ama import medical_tools as medical

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_MEDICAL_TOOLS_LIVE_TESTS") != "1",
                    reason="Set RUN_MEDICAL_TOOLS_LIVE_TESTS=1 to contact public retrieval APIs")
@pytest.mark.parametrize("disease", ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"])
def test_four_disease_live_relevance(monkeypatch, tmp_path, disease):
    index = tmp_path / "empty-index.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(index))
    original = medical._get
    responses = []

    def capture(url):
        raw = original(url)
        responses.append({"url": url, "response": json.loads(raw)})
        return raw

    monkeypatch.setattr(medical, "_get", capture)
    rows = medical.guideline_search(disease, max_results=5)
    # Public topic queries only, no patient data or provider credentials.
    output = REPO / "tmp/guideline-review/live"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{disease}.json").write_text(json.dumps({
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "query": disease,
        "raw_responses": responses, "results": rows,
        "scope": "Connectivity/title relevance smoke test, not expert review or clinical accuracy"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    assert isinstance(rows, list) and rows, rows
    assert all(disease in row["title"].casefold() for row in rows)
    assert all(row["source"] == "europepmc" and row["document_type"] in {
        "clinical_practice_guideline", "consensus"} for row in rows)


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_MEDICAL_TOOLS_LIVE_TESTS") != "1",
                    reason="Set RUN_MEDICAL_TOOLS_LIVE_TESTS=1 to contact public retrieval APIs")
@pytest.mark.parametrize("disease", ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"])
def test_reference_search_to_diagnostic_fulltext(monkeypatch, disease):
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    row = medical.guideline_search(disease, live=False, intent="diagnosis")[0]
    section = row["diagnostic_sections"][0]
    result = medical.paper_fulltext(row["pmcid"], section=section["id"], max_chars=3000)
    assert "error" not in result, result
    assert result["sections"][0]["id"] == section["id"]
    assert len(result["sections"][0]["text"]) > 100
    assert result["sections"][0]["source_url"] == section["url"]
    output = REPO / "tmp/guideline-review/live"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{disease}-fulltext.json").write_text(json.dumps({
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "source": row, "result": result
    }, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_MEDICAL_TOOLS_LIVE_TESTS") != "1",
                    reason="Set RUN_MEDICAL_TOOLS_LIVE_TESTS=1 to contact public retrieval APIs")
@pytest.mark.parametrize("name,arguments,result_field", [
    ("pubmed_search", {"query": "psoriasis", "retmax": 1}, "pmid"),
    ("guideline_search", {"query": '"chronic kidney disease"', "max_results": 1}, "title"),
    ("paper_fulltext", {"pmcid": "PMC7092803", "section": "Methods", "max_chars": 2000}, "title"),
    ("clinical_trial_search", {"condition": "psoriasis", "status": "RECRUITING", "max_results": 1}, "nct_id"),
])
def test_public_retrieval_endpoint(name, arguments, result_field, monkeypatch, tmp_path):
    # An empty index ensures guideline success comes from Europe PMC, not the local fallback.
    index = tmp_path / "guidelines.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("AMA_GUIDELINES_PATH", str(index))
    tool = next(tool for tool in load_tools(REPO / "src/ama/medical_tools.py") if tool.name == name)
    result = json.loads(tool.invoke(json.dumps(arguments)).text)
    if name == "paper_fulltext":
        assert "error" not in result, result
        assert result[result_field] and result["sections"][0]["text"]
        assert sum(len(section["text"]) for section in result["sections"]) <= arguments["max_chars"]
    else:
        assert isinstance(result, list) and result, result
        assert result[0][result_field]
        if name == "pubmed_search":
            assert result[0]["abstract"]
        if name == "guideline_search":
            assert result[0]["source"] == "europepmc"
        if name == "clinical_trial_search":
            assert result[0]["status"] == "RECRUITING"


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_API_TESTS") != "1", reason="Set RUN_API_TESTS=1 to call the real API")
@pytest.mark.parametrize("name,arguments,result_field", [
    ("pubmed_search", {"query": "psoriasis", "retmax": 1, "abstract": False}, "pmid"),
    ("guideline_search", {"query": '"chronic kidney disease"', "max_results": 1, "live": True}, "title"),
    ("paper_fulltext", {"pmcid": "PMC7092803", "section": "Methods", "max_chars": 2000}, "title"),
    ("clinical_trial_search", {"condition": "psoriasis", "status": "RECRUITING", "max_results": 1}, "nct_id"),
])
def test_deepseek_calls_medical_tool(name, arguments, result_field, monkeypatch):
    _load_dotenv(REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("Set DEEPSEEK_API_KEY in the environment or local .env")
    monkeypatch.delenv("AMA_GUIDELINES_PATH", raising=False)
    model = Model.from_config("deepseek-flash", REPO / "ama.json", timeout=60)
    model.config["max_tokens"] = 1024
    agent = Agent(model, tools=load_tools(REPO / "src/ama/medical_tools.py"), max_calls=3)

    reply = agent.chat(
        f"Call {name} exactly once with these arguments: {json.dumps(arguments)}. "
        f"Then reply with only the returned {result_field} value (use the first result if a list is returned). "
        "Do not call any other tools or answer before receiving the tool result."
    )

    calls = [call for message in agent.history for call in message.get("tool_calls", [])]
    assert len(calls) == 1, "Expected one tool call"
    assert calls[0]["function"]["name"] == name
    assert json.loads(calls[0]["function"]["arguments"]) == arguments
    observations = [message for message in agent.history if message["role"] == "tool"]
    assert len(observations) == 1
    assert observations[0]["tool_call_id"] == calls[0]["id"]
    results = json.loads(observations[0]["content"])
    if name == "paper_fulltext":
        assert isinstance(results, dict) and "error" not in results
        assert results["sections"] and results["sections"][0]["text"]
        identifier = results[result_field]
    else:
        assert isinstance(results, list) and results, "Expected a successful, nonempty tool result"
        identifier = results[0][result_field]
    assert str(identifier) in reply, "Final answer must use the returned tool result"
    assert len(agent.calls) == 2, "Expected tool selection followed by a final answer"
