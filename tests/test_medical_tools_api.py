"""Opt-in smoke test against the configured DeepSeek API; never runs in default CI."""
import json
import os
from pathlib import Path

import pytest

from ama.agent import Agent
from ama.cli import _load_dotenv
from ama.model import Model
from ama.tools import load_tools

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_API_TESTS") != "1", reason="Set RUN_API_TESTS=1 to call the real API")
@pytest.mark.parametrize("name,arguments,result_field", [
    ("pubmed_search", {"query": "psoriasis", "retmax": 1, "abstract": False}, "pmid"),
    ("guideline_search", {"query": "CKD", "live": False}, "org"),
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
        f"Then reply with only the first result's {result_field} value. "
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
    assert isinstance(results, list) and results, "Expected a successful, nonempty tool result"
    assert str(results[0][result_field]) in reply, "Final answer must use the returned tool result"
    assert len(agent.calls) == 2, "Expected tool selection followed by a final answer"
