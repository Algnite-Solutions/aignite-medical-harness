# `ama probe`: evidence accumulation and post-submission measurements

`ama run` produces clinical trajectories. `ama probe` independently measures an
existing completed run. `ama eval` reads saved outputs and scores offline.
`ama run --probing-plan` has been removed and prints a migration instruction.
No clinical tools are executed by probing.

## Commands

```sh
ama probe SOURCE_RUN --mode cumulative --model MODEL --config ama.json --prepare-only
# Inspect the emitted directory's candidates.json, snapshots.jsonl and requests/.
ama probe --resume PROBE_RUN
ama eval PROBE_RUN

ama probe SOURCE_RUN --mode endpoint --model MODEL --config ama.json
ama probe SOURCE_RUN --mode cumulative --model MODEL --kind sufficient --kind true
```

One source run is sufficient. HPI and interactive runs need not be supplied as a
pair. The source must be a completed standard AMA Decision run with one Decision
per episode. Probe model and provider must match the source. Missing diagnoses
remain in the case denominator but generate no measurement calls.

`--episode ID` may be repeated. `--runs-root`, `--timeout`, `--max-probe-tokens`
and `--top-logprobs` default to `runs`, 60 seconds, 32 tokens and 20 candidates.
Provider-specific options, including thinking configuration, remain inherited
from the chosen alias; configure token limits appropriate to that provider.
The probe sets temperature to zero and requests logprobs. No alias guarantees
provider support for this readout. Missing probabilities stop execution.
`--kind` defaults to `sufficient`; repeat it to include `true`. Duplicate kinds
are errors. `--prepare-only` needs no API credentials and sends no requests.
Resume accepts only the frozen run path, not new configuration flags.

## Cumulative mode

A tool result is one evidence unit, including the entire laboratory panel. A
batch containing examination, laboratory results and an imaging catalog,
followed by one imaging report, produces exactly these four measurement points:

```text
0  HPI
1  HPI + examination
2  HPI + examination + laboratory results
3  HPI + examination + laboratory results + imaging report
```

`list_imaging` is navigation: it is excluded from measurement inputs, point
counts and evidence counts, and retained in `excluded_results.json` for audit.
A repeated real query remains a step marked `duplicate`; error/no-result returns
remain steps marked as query feedback. No-result means no record was available,
not a negative clinical finding. Identical recorder callbacks do not add steps.

Messages are reconstructed from the first patient observation and individual
original tool-result texts in recorded order. Original assistant text, reasoning,
final explanations and subsequent probe answers never enter these inputs. The
fixed independent measurement system replaces the clinical generation system.
A partial native tool-call conversation is never sent to the API.

This is a **controlled evidence-release experiment**. Within-batch order is the
recorded result order, not an observed sequence of agent deliberations. The v1
adapter accepts text from `physical_examination`, `laboratory_results`,
`microbiology`, `list_imaging`, and `imaging`. Images, extra patient turns,
unknown tools and incomplete tool/result pairs fail preparation explicitly.

The candidate is the original final `answer.diagnosis`, used unchanged at every
point. It is a hindsight hypothesis, not the agent's contemporaneous diagnosis.
The field can itself contain explanations: review `candidates.json` and, when
needed, provide an explicit replacement via `--candidates diagnoses.json`:

```json
{"synthetic-case-1": "Acute appendicitis"}
```

This file is a JSON object keyed by selected episode IDs. No gold labels or
model-generated rewrite selects the candidate. Both original and replacement
are recorded, and the override file is hashed. Changing it requires a new task.

## Endpoint mode

The complete original history, including the final assistant answer and
reasoning summary, remains intact. The existing post-submission measurement
question is appended. Sufficient and True each fork the same original history.
This preserves the previous endpoint prompt and must not be interpreted as the
last cumulative point: the candidate may match but the visible context differs.

## Artifacts and execution

| Artifact | Contents |
|---|---|
| `plan.json`, `manifest.json` | Frozen source hashes, settings, mode and execution status |
| `candidates.json` | Original/fixed diagnoses, overrides and missing-candidate reasons |
| `snapshots.jsonl` | Cumulative evidence or complete endpoint history, with source positions |
| `excluded_results.json` | Catalog results omitted from cumulative measurements |
| `jobs.jsonl`, `requests/` | Exact independently constructed measurement jobs and requests |
| `original_submission/` | Byte-exact original manifest and clinical Decisions |
| `probe_calls.jsonl` | Replies, provider logprobs, timing, usage and job identity |
| `transport_attempts.jsonl` | Started/completed/failed requests and bounded rate-limit recovery |
| `metrics.json` | Offline per-point probabilities, adjacent changes and original clinical score |

Source messages are authoritative. If present, `prefixes.jsonl` is checked
against source message prefixes. Source changes, edited requests and snapshot
changes invalidate the frozen task. Keep source runs and source datasets in
place. Original records are never overwritten.

Execution reuses the native one-call Agent, probability reader and paced
transport. A shared lock under the selected runs root serializes probe streams.
For concurrent jobs sharing a provider account, use the same runs root. 429
recovery is bounded; unknown completion and failed/invalid measurements stop the
run. Resume reuses only exactly matching successful records, never silently
resampling an invalid or in-flight request. Different modes are not cross-reused.

The readout sums returned probability mass for A/B label tokens and reports
`p_ab = p_A / (p_A + p_B)`, keeping both provider masses, their sum and output
format diagnostics. Both labels must be available; missing mass is not zero.
Each `delta_p` compares adjacent valid points only, without bridging missing
points. These scores are not assumed to be calibrated probabilities.

`ama eval` scores original clinical Decisions separately using the source dataset
scorer. A source without an evaluation configuration is explicitly unscored;
no clinical scorer is guessed. It never invents per-step diagnoses, and overrides do
not replace the original clinical submission for scoring. Evaluation reads the
source dataset references offline; they are not part of the model input.

## Archived tasks

```sh
ama probe --resume OLD_PROBING_RUN
ama eval OLD_PROBING_RUN
```

Archived plans keep their original hashes, prompts, snapshot boundaries and
contract names. They are not converted to cumulative mode. Historical
`tool_history_replay` is batch-level; `evidence_prefix` and older interactive
probing contracts remain readable. See [archived protocol details](original-output-endpoint-probes.md).

## Offline validation

```sh
python -m pytest tests/test_probe_cli.py tests/test_tool_history_replay.py -q
python -m pytest -m 'not api' -q
```

Tests use synthetic records and mocked provider calls. They cover source
preservation, catalog exclusion, future-evidence leakage, endpoint parity,
repeated/failed results, missing candidates, frozen request integrity, selective
resumption, unknown in-flight usage and archived task compatibility.
