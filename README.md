# AMA — aignite-medical-agent

[中文](README.zh-CN.md) | English

AMA is a minimal research harness for evaluating medical agents. It uses one small abstraction for both single-turn reasoning and multi-turn memory studies:

```text
Episode -> Turn -> Evidence -> Decision
```

An Episode is one independent evaluation unit. It may contain one Turn or many Turns. On each Turn, the harness releases new Evidence and records one structured Decision. The default `direct_decision` protocol shows evidence inline; the optional `tool_agent` protocol allows evidence lookup. A dataset scorer evaluates recorded decisions after the run.

AMA is an evaluation runner, not a clinical system. It does not simulate patients, execute clinical actions, or understand FHIR, oncology workflows, or dataset-specific labels in its core loop.

## Research modes

- **Single-turn grounded reasoning:** one Episode contains one Turn. Use it for grounded question answering, evidence citation, abstention, and structured decision studies.
- **Multi-turn memory:** one Episode contains ordered Turns. New Evidence is released over time while earlier Evidence and conversation history remain available. Use it for belief updating, longitudinal consistency, and long-context or memory studies.

Multi-turn execution is teacher-forced replay. A model decision does not change future observations; the dataset controls what is released next. This makes runs reproducible, but it does not measure the downstream effect of an action.

## How a turn works

In the default `direct_decision` protocol, the model sees each newly released Evidence item and returns `{"turn_id":"t1","answer":...,"citations":[]}`. In `tool_agent`, the model can call two evidence tools before returning the same Decision:

- `list_evidence`: list metadata for Evidence released so far.
- `read_evidence`: read one released Evidence item.

Evidence visibility is cumulative within an Episode. Direct decisions may cite visible Evidence; tool-agent decisions may cite only Evidence actually read. Future Evidence is not visible. Scoring targets are never loaded during a run.

```text
dataset -> release Turn -> protocol -> Decision -> record
targets + eval rules -------------------------------> evaluate
```

## Quick start

Requires Python 3.11 or later.

```bash
python3 -m pip install -e ".[dev]"
pytest -q
ama validate datasets/thyroid_demo
ama run datasets/thyroid_demo --model scripted --protocol tool_agent \
  --instruction-file datasets/thyroid_demo/instructions.txt --runs-root runs
ama eval "$(ls -dt runs/* | head -1)"
```

If `ama` is installed but your shell cannot find it, locate its script directory with:

```bash
python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'
```

Add the printed directory to `PATH`, or use `python3 -m ama.cli` in place of `ama`.

Real OpenAI-compatible models are configured in `ama.json` or `~/.config/ama/models.json`; API keys are read from environment variables.

```bash
ama run datasets/thyroid_demo --episode thyroid_001 --model glm \
  --protocol tool_agent --instruction-file datasets/thyroid_demo/instructions.txt
ama run datasets/thyroid_demo --episode thyroid_001 --model glm --interactive \
  --protocol tool_agent --instruction-file datasets/thyroid_demo/instructions.txt
```

Interactive and batch modes use the same `run_episode()` implementation.

Datasets may also be addressed by name through a local or mounted NAS root:

```bash
export AMA_DATA_ROOT=/Volumes/lab-data/ama
ama validate thyroid_demo
ama run thyroid_demo --model glm
```

An existing explicit path takes precedence. `import --out` always remains an explicit path.

## Dataset layout

AMA Dataset is the format understood by the runner:

```text
datasets/<name>/
  dataset.json       schema, name, and splits
  episodes.jsonl     model-visible Episodes
  targets.jsonl      optional evaluator-only targets
  eval.json          optional evaluator-only scorer and rules
  provenance.jsonl   optional local source locators; never read by the runner
  instructions.txt   optional task prompt, passed explicitly to ama run
  assets/            optional files referenced by Evidence
```

The four core objects are deliberately small:

- **Episode:** one independent evaluation unit.
- **Turn:** one observation and the Evidence newly released with it.
- **Evidence:** the smallest item that should be independently visible, readable, and citable in the study.
- **Decision:** a Turn ID, task-defined `answer`, and Evidence citations.

Evidence may represent a note, report, lab panel, FHIR resource, registry record, image, or PDF. Every item must provide `text`, a safe relative `file`, or both. JPEG, PNG, GIF, and WebP files are delivered as image parts. Other file types remain path references, so importers should include useful text when needed.

Interactive runs keep their compact progress output. Enter `show` to inspect the complete model conversation so far, including system, human, assistant tool calls, and tool observations. Inline image bytes are always shown and recorded as a short artifact descriptor rather than base64.

All fields in `episodes.jsonl` are model-visible. Gold answers and evaluator-only tags belong in `targets.jsonl` or `eval.json`; source locators and patient identifiers belong in `provenance.jsonl`.

See the [AMA Dataset specification](docs/ama-dataset.md) for the complete contract.

## Importing data

New data sources are added with offline importers. An importer converts source records into AMA Dataset and writes a report describing provenance, exclusions, missing fields, and temporal assumptions. Dataset-specific logic must not be added to `run_episode()`.

```bash
ama import medagentbench --source test_data_v2.json --out datasets/medagentbench
ama import episode-folder --source path/to/episodes --out datasets/my_dataset
ama import rocov2 --source /Volumes/Shared/Work/Data/RocoV2 --out datasets/rocov2 --split test --limit 100
```

Original MedAgentBench tasks become single-turn Episodes. Optional FHIR patient timelines become separate multi-turn replay Episodes and currently have no turn-level gold targets. Scores from derived replays must not be reported as original MedAgentBench benchmark scores.

## Scoring and records

Built-in scorers are registered in `src/ama/scorer.py`:

- `exact`: exact answer matching, allowed next steps, required Evidence, and abstention.
- `workflow`: answer fields, transition rules, next-step constraints, required Evidence, and cumulative success.
- `rocov2`: normalized caption token precision/recall/F1 and unordered UMLS CUI precision/recall/F1.
- `unscored`: completion and operational measurements only.

Failed or missing decisions remain in the denominator. A zero denominator is reported as N/A.

Each run records decisions, events, model usage, latency, termination reasons, budgets, dataset hashes, dependency versions, and the Git revision. Trace content can be redacted. Scripted-model scores only verify the runner and scorer; they do not demonstrate medical capability.

## Repository map

```text
src/ama/data.py        schema, loader, and validator
src/ama/model.py       scripted and OpenAI-compatible model clients
src/ama/agent.py       the single Episode replay controller
src/ama/protocols.py   independent direct-decision and tool-agent protocols
src/ama/scorer.py      scorer registry and built-in scorers
src/ama/recorder.py    run artifacts, usage, hashes, and redaction
src/ama/cli.py         validate, inspect, run, eval, and import
src/ama/importers/     offline source converters
datasets/              examples and imported datasets
docs/                  dataset contract
```

## Supported datasets

The checked-in examples cover both single-turn trustworthy reasoning and multi-turn longitudinal memory studies.

| Dataset | Source | Episodes | Turns | Targets | Scorer | Data card |
|---|---|---:|---:|---|---|---|
| `thyroid_demo` | Synthetic oncology workflow | 1 | 3 | Yes | `workflow` | [English](datasets/thyroid_demo/DATASET_CARD.md) · [中文](datasets/thyroid_demo/DATASET_CARD.zh-CN.md) |
| `medagentbench` | Derived MedAgentBench FHIR replay subset | 3 | 12 | No | `unscored` | [English](datasets/medagentbench/DATASET_CARD.md) · [中文](datasets/medagentbench/DATASET_CARD.zh-CN.md) |
| `rocov2_demo` | ROCOv2 radiology image subset | 3 | 3 | Yes | `rocov2` | [English](datasets/rocov2_demo/DATASET_CARD.md) · [中文](datasets/rocov2_demo/DATASET_CARD.zh-CN.md) |
