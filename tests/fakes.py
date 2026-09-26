"""Deterministic models live ONLY in tests, never in model registration."""
import copy
import json


class FakeModel:
    name = "test-model"
    config = {"model": "deterministic-test-only"}
    timeout = 60
    last_usage = None

    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, history, tools=None):
        self.requests.append((copy.deepcopy(history), copy.deepcopy(tools)))
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}


def call(name="add", arguments='{"a": 1, "b": 2}', id="call-1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def calls(*items):
    return {"role": "assistant", "content": None, "tool_calls": list(items)}


def decision(turn="t1", answer=None, citations=None):
    return json.dumps({"turn_id": turn, "answer": answer, "citations": citations or []})
