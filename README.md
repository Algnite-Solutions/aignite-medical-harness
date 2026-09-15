# AMA — aignite-medical-agent

[中文](README.zh-CN.md) | English

AMA is a minimal research harness for evaluating medical agents. It uses one small abstraction for both single-turn reasoning and multi-turn memory studies:

```text
Episode -> Turn -> Evidence -> Decision
```

An Episode is one independent evaluation unit. It may contain one Turn or many Turns. On each Turn, the harness releases new Evidence, lets the model inspect available Evidence through three tools, and records one structured Decision. A dataset scorer evaluates the recorded decisions after the run.

AMA is an evaluation runner, not a clinical system. It does not simulate patients, execute clinical actions, or understand FHIR, oncology workflows, or dataset-specific labels in its core loop.

## Research modes

- **Single-turn grounded reasoning:** one Episode contains one Turn. Use it for grounded question answering, evidence citation, abstention, and structured decision studies.
- **Multi-turn memory:** one Episode contains ordered Turns. New Evidence is released over time while earlier Evidence and conversation history remain available. Use it for belief updating, longitudinal consistency, and long-context or memory studies.

Multi-turn execution is teacher-forced replay. A model decision does not change future observations; the dataset controls what is released next. This makes runs reproducible, but it does not measure the downstream effect of an action.

## How a turn works

The model receives a world observation and can call exactly three tools:

- `list_evidence`: list metadata for Evidence released so far.
- `read_evidence`: read one released Evidence item.
- `submit_decision`: submit the Turn's state, optional action, citations, or an abstention.

Evidence visibility is cumulative within an Episode. A citation is accepted only after the model has read that Evidence item. Future Evidence is not visible. Scoring targets are never loaded during a run.

```text
dataset -> release Turn -> list/read Evidence -> submit Decision -> record
                                                               |
targets + hidden policy ---------------------------------------+-> evaluate
```

## Quick start

Requires Python 3.11 or later.

```bash
python3 -m pip install -e ".[dev]"
pytest -q
ama validate datasets/thyroid_demo
ama run datasets/thyroid_demo --model scripted --runs-root runs
ama eval "$(ls -dt runs/* | head -1)"
```

If `ama` is installed but your shell cannot find it, locate its script directory with:

```bash
python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'
```

Add the printed directory to `PATH`, or use `python3 -m ama.cli` in place of `ama`.

Real OpenAI-compatible models are configured in `ama.json` or `~/.config/ama/models.json`; API keys are read from environment variables.

```bash
ama run datasets/thyroid_demo --episode thyroid_001 --model glm
ama run datasets/thyroid_demo --episode thyroid_001 --model glm --interactive
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

AMA Dataset v0 is the only format understood by the runner:

```text
datasets/<name>/
  dataset.json       metadata, splits, and scorer name
  episodes.jsonl     model-visible Episodes
  targets.jsonl      optional evaluator-only targets
  policy.json        optional public guidance and hidden evaluation policy
  assets/             optional files referenced by Evidence
```

The four core objects are deliberately small:

- **Episode:** one independent evaluation unit and subject.
- **Turn:** one observation and the Evidence newly released with it.
- **Evidence:** the smallest item that should be independently visible, readable, and citable in the study.
- **Decision:** the model's structured output for one Turn.

Evidence may represent a note, report, lab panel, FHIR resource, registry record, image, or PDF. Every Evidence item must provide `text`, `artifact`, or both. `artifact` is a safe relative path inside the dataset directory. After `read_evidence`, JPEG, PNG, GIF, and WebP artifacts are delivered to OpenAI-compatible vision models as an `image_url` content part. Other artifact types remain path references, so importers should provide a useful textual representation when the model needs their content.

Interactive runs keep their compact progress output. Enter `show` to inspect the complete model conversation so far, including system, human, assistant tool calls, and tool observations. Inline image bytes are always shown and recorded as a short artifact descriptor rather than base64.

All fields in `episodes.jsonl`, including `Evidence.metadata`, are model-visible. Gold answers, derived labels, and evaluator-only tags belong in `targets.jsonl` or `policy.hidden`.

See the [AMA Dataset v0 data card](docs/ama-dataset-v0.md) for the complete contract and importer guidance.

## Importing data

New data sources are added with offline importers. An importer converts source records into AMA Dataset v0 and writes a report describing provenance, exclusions, missing fields, and temporal assumptions. Dataset-specific logic must not be added to `run_episode()`.

```bash
ama import medagentbench --source test_data_v2.json --out datasets/medagentbench
ama import episode-folder --source path/to/episodes --out datasets/my_dataset
ama import rocov2 --source /Volumes/Shared/Work/Data/RocoV2 --out datasets/rocov2 --split test --limit 100
```

Original MedAgentBench tasks become single-turn Episodes. Optional FHIR patient timelines become separate multi-turn replay Episodes and currently have no turn-level gold targets. Scores from derived replays must not be reported as original MedAgentBench benchmark scores.

## Scoring and records

Built-in scorers are registered in `src/ama/scorer.py`:

- `exact_v0`: exact state or answer matching, allowed actions, required Evidence, and abstention.
- `workflow_v0`: state fields, transition rules, action constraints, required Evidence, and cumulative success.
- `rocov2_v0`: normalized caption token precision/recall/F1 and unordered UMLS CUI precision/recall/F1.
- `unscored`: completion and operational measurements only.

Failed or missing decisions remain in the denominator. A zero denominator is reported as N/A.

Each run records decisions, events, model usage, latency, termination reasons, budgets, dataset hashes, dependency versions, and the Git revision. Trace content can be redacted. Scripted-model scores only verify the runner and scorer; they do not demonstrate medical capability.

## Repository map

```text
src/ama/data.py        schema, loader, and validator
src/ama/model.py       scripted and OpenAI-compatible model clients
src/ama/agent.py       the single Episode loop and three tools
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
| `thyroid_demo` | Synthetic oncology workflow | 1 | 3 | Yes | `workflow_v0` | [English](datasets/thyroid_demo/DATASET_CARD.md) · [中文](datasets/thyroid_demo/DATASET_CARD.zh-CN.md) |
| `medagentbench` | Derived MedAgentBench FHIR replay subset | 3 | 12 | No | `unscored` | [English](datasets/medagentbench/DATASET_CARD.md) · [中文](datasets/medagentbench/DATASET_CARD.zh-CN.md) |
| `rocov2_demo` | ROCOv2 radiology image subset | 3 | 3 | Yes | `rocov2_v0` | [English](datasets/rocov2_demo/DATASET_CARD.md) · [中文](datasets/rocov2_demo/DATASET_CARD.zh-CN.md) |
