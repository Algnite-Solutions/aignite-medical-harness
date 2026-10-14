"""Opt-in end-to-end visual expert smoke test using public demo images."""
import json
import os
from pathlib import Path

import pytest

from ama.agent import Agent
from ama.cli import _load_dotenv
from ama.image_analysis_tools import make_image_analysis_tool
from ama.model import Model

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.api
@pytest.mark.skipif(os.environ.get("RUN_API_TESTS") != "1", reason="Set RUN_API_TESTS=1 to call the real API")
def test_deepseek_visual_expert_roundtrip(monkeypatch):
    _load_dotenv(REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("Set DEEPSEEK_API_KEY in the environment or local .env")
    monkeypatch.setenv("AMA_VISION_MODEL", "deepseek-flash")
    monkeypatch.setenv("AMA_VISION_CONFIG", str(REPO / "ama.json"))
    tool = make_image_analysis_tool()
    tool.on_images({
        "image_01": REPO / "datasets/rocov2_demo/artifacts/ROCOv2_2023_test_000001.jpg",
        "image_02": REPO / "datasets/rocov2_demo/artifacts/ROCOv2_2023_test_000015.jpg",
    })
    query = ("In at most 150 words, describe the visual layout and image type of image_01 and image_02 separately. "
             "These are unrelated research illustrations, not views of one patient. "
             "Mention each image ID and any visual limitations. Do not diagnose a patient.")
    model = Model.from_config("deepseek-flash", REPO / "ama.json", timeout=60)
    model.config["max_tokens"] = 1024
    agent = Agent(model, tools=[tool], max_calls=3)
    reply = agent.chat(
        "Two image evidence IDs are available: image_01, image_02. "
        "Call analyze_images exactly once with image_ids=[\"image_01\",\"image_02\"] and query="
        + json.dumps(query) + ". After the tool returns, give a one-sentence summary of its analysis."
    )
    calls = [call for message in agent.history for call in message.get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "analyze_images"
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments == {"query": query, "image_ids": ["image_01", "image_02"]}
    observations = [row for row in agent.history if row["role"] == "tool"]
    assert len(observations) == 1 and observations[0]["tool_call_id"] == calls[0]["id"]
    result = json.loads(observations[0]["content"].split("\nCiteable")[0])
    assert "error" not in result and result["model"] == "deepseek-flash"
    assert result["image_ids"] == ["image_01", "image_02"] and len(result["images"]) == 2
    assert all(key in result["analysis"] for key in result["image_ids"]), result["analysis"]
    assert result["usage"]["total_tokens"] > 0 and not result["cached"]
    assert reply.strip() and len(agent.calls) == 2
    assert all(isinstance(row["content"], str) or row["content"] is None for row in agent.history)
    assert "base64" not in json.dumps(agent.history)
