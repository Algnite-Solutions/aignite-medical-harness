# AMA — one small Agent, two dataset workflows

[中文](README.zh-CN.md) | English

[![CI](https://github.com/Algnite-Solutions/aignite-medical-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Algnite-Solutions/aignite-medical-harness/actions/workflows/ci.yml)

The Agent owns conversation history and optional tools. It knows nothing about datasets or scoring.
The experiment layer feeds observations to it: automatically with `run`, interactively with `chat`.

![Recorded single-sample run and Chinese follow-up chat](assets/ama-demo.gif)

[Video (MP4)](assets/ama-demo.mp4) · [Terminal recording (.cast)](assets/ama-demo.cast)

Recorded with `glm-vision` on the public ROCOv2 demo: one sample run, its saved result, and two Chinese follow-ups.
The model's original output is preserved, including an omitted caption and incorrect medical inferences;
this demonstrates the workflow, not diagnostic accuracy. Local paths and waiting times are shortened in the GIF/video.
Replay the terminal recording with `asciinema play assets/ama-demo.cast`.

## Get started

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
# Put your API key in local .env (see .env.example).
ama run datasets/rocov2_demo --model qwen36
ama eval runs/<run-directory>
ama chat datasets/rocov2_demo --episode ROCOv2_2023_test_000001 --model qwen36
```

GitHub Actions runs the deterministic tests and an installed-CLI check on Python 3.11 for PRs to `main`
and updates to `main`. CI requires no model API keys.

Models are aliases in `ama.json`; fake models exist only in tests. A configured alias is not a guarantee
that a provider supports images, tools, or their combination. Requests use standard Chat Completions;
unsupported history is reported, never merged or rewritten.

In the small compatibility check, GLM vision returned a valid Decision
envelope but omitted the caption; the configured Qwen endpoint returned invalid Decisions in the sampled image runs and
rejected a tool-returned-image request. Its text-only tool/follow-up check passed.

`run` starts a new Agent for each episode and requests a JSON Decision per turn.
`chat` sends the first observation immediately. Ordinary text is a follow-up, `/next` releases one
more observation, and `/quit` saves and exits. After the last observation, follow-ups still work.
Chat answers may be natural language; **chat records cannot be scored**.

Both commands automatically check visible data and read the dataset's `instructions.txt`.
Use `--instruction-file path` to replace it. `--max-calls 8` limits API calls per input;
`--timeout 60` sets the per-request timeout. There are no automatic retries or context trimming.

## Optional tools

```bash
ama chat datasets/rocov2_demo --episode ROCOv2_2023_test_000001 --model qwen36 --tools examples/tools.py
```

The Python file explicitly exports `TOOLS: list[Tool]`; both run and chat accept it.
No flag means no tools. Tools are trusted Python, **not sandboxed**: do not expose references or
future observations through them. See [the executable example](examples/tools.py).

Tools return text, JSON or `ToolResult(text="scan", images=["scan.jpg"])`.
Calls execute sequentially with original IDs; all tool results precede image attachments.
Tool errors go back to the model. API errors terminate the episode instead of prompting a repair.

## Read in four steps

1. [model.py](src/ama/model.py): registration, image encoding and one request.
2. [agent.py](src/ama/agent.py): full multi-turn history.
3. [tools.py](src/ama/tools.py) + the Agent loop: optional tool calls.
4. [runner.py](src/ama/runner.py) + [cli.py](src/ama/cli.py): observations, decisions and commands.

The [Chinese code walkthrough](docs/minimal-agent.zh-CN.md) includes runnable review examples.

## Data and results

```bash
ama import rocov2 --source /path/to/ROCOv2 --out datasets/rocov2 --split test --limit 100
```

For a subset, use `ama run datasets/rocov2 --model qwen36 --split test`, or repeat
`--episode ID` instead of `--split`. Use `--runs-root path` to choose the output parent.

ROCOv2 is the built-in `ama import` demo. Each image is one episode/turn; ordered multi-turn datasets
also work. See the [format](docs/ama-dataset.md) and [data card](datasets/rocov2_demo/DATASET_CARD.md).

### Credentialed MIMIC-CDM import

The MIMIC-IV-Ext Clinical Decision Making v1.1 importer is run separately so the core CLI and
its existing dataset workflow stay small. Use a credentialed local source and restricted output:

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm \
  --source /path/to/mimic-iv-ext-clinical-decision-making \
  --out /path/to/processed
ama run /path/to/processed/mimic_cdm_interactive --model MODEL --episode HADM_ID \
  --tools src/ama/importers/mimic_cdm_tools.py --max-calls 24 --runs-root /path/to/restricted-runs
ama run /path/to/processed/mimic_cdm_full_info --model MODEL --episode HADM_ID \
  --runs-root /path/to/restricted-runs
ama eval /path/to/restricted-runs/RUN_ID
```

The importer creates both 2,400-admission datasets and validates source structure. Interactive
episodes start with HPI; the trusted tool file binds examination, complete laboratory results,
microbiology, and imaging to the current admission. Use `list_imaging` to see available reports,
then `imaging(report_id)` to read one. Existing processed datasets work with these tools without
reimporting. Full-information episodes present all those inputs together.
Discharge outcomes remain evaluator-only. Both modes score four-way diagnosis; treatment plans are
recorded for review. Tool results appear in event logs; Decision citations can use episode evidence
IDs or evidence IDs explicitly registered by successful tools. The
credentialed source and model transcripts should remain on restricted storage; importing itself does
not call a model.

For a controlled comparison of model diagnosis and tool use, the
[matched open-answer protocol](docs/mimic-cdm-open-benchmark.md) builds HPI-only,
interactive, and full-information views from the same admissions. It gives all views
the same answer field and hides the four candidate labels from the prompt; source
targets remain four-group, with conservative automatic mapping and a review queue.

Records include `manifest.json` (model configuration, dataset, expected turns and status),
`messages.json` (episode IDs mapped to ordered message arrays, including the actual system prompt),
`diagnostics.jsonl` (tool definitions, model calls, evidence releases and errors), and for run only
`decisions.jsonl` (parsed answers, validation, turn status and totals). No SHA fingerprints are recorded
or checked. New runs use log schema v3; historical logs are not rewritten.
Messages retain their protocol fields without per-message episode/source metadata. Images stay as paths.
Each decision's `message_range: [start, end]` selects its messages by zero-based indices with an exclusive
end in `messages.json[episode_id]`. Model-call diagnostics use `message_index` for the assistant response
position; failed calls may have no response at that position. Raw replies are stored only in the transcript.
Conversation snapshots are replaced atomically after each exchange, including caught errors and Ctrl-C.
A hard process kill during an exchange may lose that exchange, but leaves the preceding snapshot readable.
Final-answer extraction ignores marked `<think>...</think>` blocks, including JSON drafts inside them,
while retaining the raw transcript and recording how many blocks were excluded. Such replies still
fail strict JSON compliance; malformed thinking markers are rejected and citation IDs are not rewritten.
Invalid decisions are kept, without correction dialogues. Failed/missing turns remain in evaluation.
Output contract v2 requests `answer`, `citations`, and a concise `reasoning_summary` alongside
`turn_id`. Recoverable JSON answers are normalized without extra model calls; raw replies and
format/citation/summary diagnostics remain available. Evaluation reports task accuracy separately
from `aggregate.output_quality`. Empty citations do not count as valid supporting references.
API/call-limit failures skip the rest of that episode and continue the batch; Ctrl-C stops the batch.
Partial runs return a nonzero CLI status and remain independently evaluable.

Only `eval` opens references/rules, producing `metrics.json`.
It checks that selected episodes and expected turns still match. Original replies remain in `messages.json`;
references stay in the dataset's `targets.jsonl`. Caption-word and concept-ID overlap are not clinical accuracy.
Compatible historical runs remain evaluable; metrics absent from their output contract are null.
Provenance never enters model input.

AntAngelMed2 endpoints without a server tool parser can opt into
`"tool_call_parser": "antangel"` in their model configuration. This sends tool definitions with
`tool_choice: "none"` (verified on the configured AntAngel endpoint) and parses the model's raw
`<tool_call>` / `<arg_key>` / `<arg_value>` output into standard tool calls. Raw assistant content
is retained. It adds no prompts or model calls and does not alter final answers. This is an explicit
endpoint compatibility mode, not a general fallback for providers that honor `none` by disabling tools.
