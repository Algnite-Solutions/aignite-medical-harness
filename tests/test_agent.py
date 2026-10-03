import json

import pytest

from ama.agent import Agent, CallLimitExceeded
from ama.model import ModelError, image, wire_messages
from ama.tools import Tool, ToolResult, load_tools
from fakes import FakeModel, call, calls


def addition():
    return Tool("add", "Add", {"type": "object"}, lambda a, b: {"sum": a + b})


def test_no_tools_history_and_isolation():
    model = FakeModel("first", "second", "fresh")
    agent = Agent(model, system="system")
    assert agent.chat("hello") == "first"
    assert agent.chat("follow-up") == "second"
    assert [m["role"] for m in model.requests[1][0]] == ["system", "user", "assistant", "user"]
    assert all(tools is None for _, tools in model.requests)
    Agent(model).chat("new episode")
    assert model.requests[2][0] == [{"role": "user", "content": "new episode"}]


def test_multiple_tools_order_ids_and_json():
    model = FakeModel(calls(call(id="A"), call(arguments='{"a": 4, "b": 5}', id="B")), "done")
    agent = Agent(model, tools=[addition()])
    assert agent.chat("calculate") == "done"
    results = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["A", "B"]
    assert [json.loads(m["content"])["sum"] for m in results] == [3, 9]
    assert len(agent.calls) == 2


def test_tool_events_arrive_before_next_model_request():
    events = []

    class CheckingModel(FakeModel):
        def complete(self, history, tools=None):
            if self.requests:
                assert events == ["model_start", "tool_call", "tool_result", "model_start"]
            return super().complete(history, tools)

    agent = Agent(CheckingModel(calls(call()), "done"), tools=[addition()],
                  on_event=lambda kind, _: events.append(kind))
    assert agent.chat("calculate") == "done"


@pytest.mark.parametrize("bad", [call(name="unknown"), call(arguments="invalid"), call(arguments="[]"),
                                 call(arguments='{"wrong":1}')])
def test_tool_error_can_be_corrected(bad):
    model = FakeModel(calls(bad), calls(call()), "recovered")
    agent = Agent(model, tools=[addition()])
    assert agent.chat("go") == "recovered"
    assert "error" in json.loads(agent.history[2]["content"])
    assert json.loads(agent.history[4]["content"]) == {"sum": 3}


def test_all_tool_results_precede_images(tmp_path):
    path = tmp_path / "scan.png"
    path.write_bytes(b"image fixture")
    tool = Tool("scan", "read scan", {"type": "object"}, lambda: ToolResult("scan", [path]))
    model = FakeModel(calls(call("scan", "{}", "A"), call("scan", "{}", "B")), "seen")
    agent = Agent(model, tools=[tool])
    assert agent.chat(["image input", image(path)]) == "seen"
    assert [m["role"] for m in agent.history] == ["user", "assistant", "tool", "tool", "user", "assistant"]
    assert [p["text"] for p in agent.history[4]["content"] if p["type"] == "text"] == [
        "Tool image: call_id=A, name=scan", "Tool image: call_id=B, name=scan"]
    before = json.dumps(agent.history)
    wire = wire_messages(agent.history)
    assert "base64" in json.dumps(wire) and "base64" not in before
    assert json.dumps(agent.history) == before


def test_tool_exception_and_missing_image_return_errors(tmp_path):
    def bad():
        raise RuntimeError("broken tool")
    for handler in [bad, lambda: ToolResult(images=[tmp_path / "absent.png"])]:
        agent = Agent(FakeModel(calls(call("bad", "{}")), "handled"),
                      tools=[Tool("bad", "bad", {}, handler)])
        assert agent.chat("go") == "handled"
        assert "error" in agent.history[2]["content"]


def test_call_limit_keeps_tool_result_and_stops_session():
    agent = Agent(FakeModel(calls(call())), tools=[addition()], max_calls=1)
    with pytest.raises(CallLimitExceeded):
        agent.chat("go")
    assert agent.history[-1]["role"] == "tool"
    with pytest.raises(RuntimeError, match="session has stopped"):
        agent.chat("again")


@pytest.mark.parametrize("error", [ModelError("offline"), KeyboardInterrupt()])
def test_errors_propagate_without_retry(error):
    model = FakeModel(error)
    agent = Agent(model)
    with pytest.raises(type(error)):
        agent.chat("keep this input")
    assert len(model.requests) == 1 and len(agent.calls) == 1
    assert agent.history[-1]["content"] == "keep this input"
    assert agent.calls[0]["error"]


def test_tool_exports_and_duplicate_names(tmp_path):
    assert load_tools(None) == []
    assert load_tools("examples/tools.py")[0].invoke('{"a":2,"b":3}').text == '{"sum": 5}'
    invalid = tmp_path / "invalid.py"
    invalid.write_text("TOOLS = [lambda: 1]")
    with pytest.raises(ValueError, match="TOOLS"):
        load_tools(invalid)
    with pytest.raises(ValueError, match="duplicate"):
        Agent(FakeModel(), tools=[addition(), addition()])


def test_budget_resets_per_input_and_duplicate_ids_are_not_executed():
    model = FakeModel("one", "two")
    agent = Agent(model, max_calls=1)
    assert agent.chat("first") == "one"
    assert agent.chat("second") == "two"
    agent = Agent(FakeModel(calls(call(id="same"), call(id="same"))), tools=[addition()])
    with pytest.raises(ModelError, match="unique"):
        agent.chat("go")
    assert not any(m["role"] == "tool" for m in agent.history)
