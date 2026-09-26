"""Opt-in tools: ama chat datasets/rocov2_demo --episode ID --model qwen36 --tools examples/tools.py.

Tools are trusted Python, not a sandbox. Keep references/future observations out of tools.
"""
from ama.tools import Tool


def add(a: float, b: float):
    return {"sum": a + b}


TOOLS = [Tool(
    name="add", description="Add two numbers.",
    parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"], "additionalProperties": False},
    handler=add,
)]
