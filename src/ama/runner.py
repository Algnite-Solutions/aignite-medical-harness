"""Dataset boundary: observations, decisions and exploration; the Agent knows none of these."""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from .agent import Agent, CallLimitExceeded
from .data import Decision, load_dataset, resolve_dataset_dir, validate_visible_files
from .model import IMAGE_TYPES, ModelError, image
from .recorder import Recorder
from .tools import load_tools

DECISION_PROMPT = '''For each observation, return exactly one JSON object, without Markdown:
{"turn_id": "the current turn ID", "answer": <task answer>, "citations": ["evidence IDs"]}.
Use only evidence released so far. Citations must identify released evidence.
answer=null with citations=[] means abstention. Do not add other top-level fields.'''


def render(turn, folder: Path) -> list:
    parts = [f"Turn: {turn.id}"]
    if turn.available_at:
        parts.append(f"Available at: {turn.available_at.isoformat()}")
    if turn.observation:
        parts.append(turn.observation)
    for evidence in turn.evidence:
        parts.append(f"Evidence: {evidence.id}" + (f" ({evidence.type})" if evidence.type else ""))
        if evidence.text:
            parts.append(evidence.text)
        if evidence.file:
            path = folder / evidence.file
            if mimetypes.guess_type(path.name)[0] in IMAGE_TYPES:
                parts.append(image(path))
            else:
                # Unsupported binary files must not silently become invisible evidence.
                parts.append(path.read_text(encoding="utf-8"))
    return parts


def parse_decision(reply: str, turn_id: str, visible: set[str]) -> dict:
    decision = Decision.model_validate(json.loads(reply))
    if decision.turn_id != turn_id:
        raise ValueError("decision turn_id does not match current observation")
    if set(decision.citations) - visible:
        raise ValueError("citations reference unreleased or unknown evidence")
    if decision.answer is None and decision.citations:
        raise ValueError("abstention requires empty citations")
    return decision.model_dump(mode="json")


def exchange(agent, recorder, message, *, episode_id, turn_id, source):
    """Always persist the history produced so far, including on API error or Ctrl-C."""
    start, calls = len(agent.history), len(agent.calls)
    reply, reason, error = None, "completed", None
    try:
        reply = agent.chat(message)
    except KeyboardInterrupt:
        reason, error = "interrupted", "KeyboardInterrupt"
    except CallLimitExceeded as exc:
        reason, error = "call_limit", str(exc)
    except ModelError as exc:
        reason, error = "api_error", str(exc)
    except Exception as exc:
        reason, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        recorder.history(agent, start, source=source, episode_id=episode_id, turn_id=turn_id)
        for call in agent.calls[calls:]:
            recorder.event("model_call", episode_id=episode_id, turn_id=turn_id, **call)
    new_calls = agent.calls[calls:]
    usage = [c["usage"] for c in new_calls if isinstance(c["usage"], dict)]
    row = {"episode_id": episode_id, "turn_id": turn_id, "reply": reply, "decision": None,
           "termination": reason, "error": error, "model_calls": len(new_calls),
           "duration_ms": sum(c["duration_ms"] for c in new_calls),
           "usage": {k: sum(u.get(k, 0) for u in usage) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
           if usage else None}
    recorder.event("exchange", source=source, **row)
    return row


def execute(dataset_dir, model, *, mode="run", episode_ids=None, split=None, runs_root=Path("runs"),
            instruction_file=None, tools_path=None, max_calls=8, input_fn=None) -> Path:
    dataset = load_dataset(resolve_dataset_dir(dataset_dir))
    validate_visible_files(dataset)
    if episode_ids and split:
        raise ValueError("choose --episode or --split, not both")
    episodes = ([dataset.select(episode_id=e)[0] for e in dict.fromkeys(episode_ids)]
                if episode_ids else dataset.select(split=split))
    if mode == "chat" and (not episode_ids or len(episodes) != 1):
        raise ValueError("chat requires exactly one explicit episode")
    if not episodes:
        raise ValueError("no episodes selected")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    instruction_path = Path(instruction_file) if instruction_file else dataset.folder / "instructions.txt"
    instruction = instruction_path.read_text(encoding="utf-8")
    system = instruction + "\n\n" + (DECISION_PROMPT if mode == "run" else
        "This is exploratory chat. Respond naturally; JSON decisions are not required. "
        "Use only observations released so far; a follow-up is not a new dataset turn.")
    recorder = Recorder(runs_root, dataset=dataset, episodes=episodes, mode=mode, model=model,
                        instruction=instruction, system=system, max_calls=max_calls, tools_path=tools_path)
    print(f"records: {recorder.run_dir}", flush=True)
    termination = "completed"
    interrupted = False
    try:
        for episode in episodes:
            if interrupted:
                for turn in episode.turns:
                    recorder.append("decisions.jsonl", {
                        "episode_id": episode.id, "turn_id": turn.id, "reply": None,
                        "decision": None, "termination": "not_executed", "error": "interrupted",
                        "model_calls": 0, "duration_ms": 0, "usage": None})
                continue
            agent = Agent(model, system=system, tools=load_tools(tools_path), max_calls=max_calls)
            recorder.event("episode_start", episode_id=episode.id,
                           tools=[tool.definition() for tool in agent.tools.values()])
            recorder.history(agent, 0, source="system", episode_id=episode.id, turn_id=None)
            if mode == "chat":
                termination = explore(dataset, episode, agent, recorder, input_fn or input)
                break
            visible = set()
            stopped = None
            for turn in episode.turns:
                if stopped:
                    row = {"episode_id": episode.id, "turn_id": turn.id, "reply": None,
                           "decision": None, "termination": "not_executed", "error": stopped,
                           "model_calls": 0, "duration_ms": 0, "usage": None}
                else:
                    visible.update(e.id for e in turn.evidence)
                    row = exchange(agent, recorder, render(turn, dataset.folder),
                                   episode_id=episode.id, turn_id=turn.id, source="dataset")
                    if row["termination"] == "completed":
                        try:
                            row["decision"] = parse_decision(row["reply"], turn.id, visible)
                        except (ValueError, TypeError) as exc:
                            row.update(termination="invalid_decision", error=str(exc))
                    else:
                        stopped = row["termination"]
                    if row["termination"] != "completed":
                        termination = "completed_with_errors"
                recorder.append("decisions.jsonl", row)
            if stopped == "interrupted":
                termination = "interrupted"
                interrupted = True
    except KeyboardInterrupt:
        termination = "interrupted"
    except Exception as exc:
        termination = "error"
        recorder.event("error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        recorder.finish(termination)
    return recorder.run_dir


def explore(dataset, episode, agent, recorder, input_fn):
    index = 0
    source = "dataset"
    message = render(episode.turns[index], dataset.folder)
    while True:
        turn = episode.turns[index]
        if source == "dataset":
            print(f"\nObservation {index + 1}/{len(episode.turns)}: {turn.id}")
            for part in message:
                print(part if isinstance(part, str) else f"[image] {part['path']}")
        row = exchange(agent, recorder, message, episode_id=episode.id, turn_id=turn.id, source=source)
        if row["termination"] != "completed":
            print(f"Stopped: {row['termination']}: {row['error']}")
            return row["termination"]
        print(row["reply"])
        while True:
            try:
                text = input_fn("(ama) /next /quit > ").strip()
            except EOFError:
                return "eof"
            if text == "/quit":
                return "quit"
            if text == "/next":
                if index + 1 == len(episode.turns):
                    print("没有后续材料；仍可继续追问。")
                    continue
                index += 1
                message, source = render(episode.turns[index], dataset.folder), "dataset"
                break
            if text:
                message, source = text, "human"
                break
