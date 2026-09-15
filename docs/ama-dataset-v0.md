# AMA Dataset v0 — Data Card and Format

[中文](ama-dataset-v0.zh-CN.md) | English

## Purpose

AMA Dataset v0 is the single input format for the AMA runner. It supports single-turn grounded reasoning and multi-turn longitudinal memory experiments without adding domain-specific concepts to the core loop.

```text
Episode -> Turn -> Evidence -> Decision
```

This document is both the format specification and the minimum data card an importer author needs to complete.

## Directory contract

```text
datasets/<name>/
  dataset.json       required; metadata, splits, scorer
  episodes.jsonl     required; one model-visible Episode per line
  targets.jsonl      optional; evaluator-only targets
  policy.json        optional; public and hidden policy
  assets/             optional; files referenced by Evidence
  import_report.json optional; source conversion report
```

The run path loads `dataset.json`, `episodes.jsonl`, and `policy.public`. It does not open or hash `targets.jsonl`. Evaluation loads targets and `policy.hidden`.

## Core objects

### Episode

An Episode is one independent evaluation unit.

```json
{
  "episode_id": "case-001",
  "subject_id": "subject-001",
  "metadata": {},
  "turns": []
}
```

One-turn Episodes are valid. Multi-turn Episodes must have non-null, monotonically non-decreasing Turn times. Model context and previously released Evidence persist within an Episode and reset between Episodes.

`subject_id` may be pseudonymous. Importers must follow the source dataset's privacy and redistribution requirements.

### Turn

A Turn is one world observation plus Evidence newly made available at that point.

```json
{
  "turn_id": "t1",
  "time": "2026-01-01T09:00:00+00:00",
  "message": "A new pathology report is available.",
  "evidence": []
}
```

`time` is the release time used to order a replay. It may be null only for a single-turn Episode. When a source distinguishes event, recorded, and available times, preserve them in Evidence metadata and place the Evidence in the Turn when it became available to the agent.

### Evidence

Evidence is the smallest unit that should be independently visible, readable, and citable in the experiment. It is an evaluation object, not necessarily one database row or one atomic clinical fact.

```json
{
  "evidence_id": "path-001",
  "kind": "pathology_report",
  "text": "Left thyroid nodule: papillary thyroid carcinoma.",
  "artifact": "assets/path-001.pdf",
  "source": "pathology/reports/001",
  "metadata": {
    "event_time": "2026-01-01T08:30:00+00:00",
    "specimen": "left_thyroid"
  }
}
```

| Field | Requirement |
|---|---|
| `evidence_id` | Stable and unique within the Episode. |
| `kind` | A concise source or content type such as `lab_panel`, `clinical_note`, `radiology_report`, `pathology_report`, `fhir_condition`, or `image`. |
| `text` | Model-readable normalized content. May be empty only when `artifact` is present. |
| `artifact` | Optional path relative to the dataset directory. Absolute paths and `..` traversal are invalid. |
| `source` | A provenance locator sufficient to trace the source record under the dataset's access rules. |
| `metadata` | Optional source-specific, model-visible fields. |

Choose a natural unit. A full report or same-time lab panel is often better than splitting every sentence or measurement. An Evidence item should stand on its own without changing its meaning and should be useful as one citation.

All Evidence fields are model-visible after `read_evidence`. Do not place gold answers, derived clinical conclusions, scorer labels, or future information in `metadata`. Put evaluator-only information in `targets.jsonl` or `policy.hidden`.

`artifact` identifies source or derived material such as an image, PDF, waveform, or text file. AMA v0 validates that the path stays inside the dataset and that the file exists. After `read_evidence`, JPEG, PNG, GIF, and WebP files are sent to OpenAI-compatible vision models as an `image_url` content part. The configured model must support that Chat Completions representation. Other file types remain path references; provide `text` when the tested model needs their content. Inline image bytes are never persisted in traces or printed by the interactive shell.

| Source data | Suggested Evidence unit |
|---|---|
| Clinical notes | One note or a clearly identified section |
| Laboratory data | One collection-time panel, or one independently meaningful result |
| Radiology or pathology | One report |
| FHIR | One Resource, or a small group representing one clinical event |
| Registry data | One diagnosis, staging, treatment, or follow-up record |
| Medical QA | One supplied context passage |
| Image, PDF, or waveform | One artifact with a faithful text representation when required |

### Decision

The model submits exactly one accepted Decision per completed Turn:

```json
{
  "turn_id": "t1",
  "state": {},
  "action": {"name": "order_biopsy", "arguments": {}},
  "citations": ["path-001"],
  "abstain": false,
  "note": ""
}
```

`state` and `action.arguments` are dataset-defined JSON objects interpreted by the scorer. Citations must refer to visible Evidence previously read by the model. When `abstain` is true, `state`, `action`, and `citations` must be empty.

## Dataset metadata

`dataset.json` has this shape:

```json
{
  "schema": "ama-dataset-v0",
  "name": "example",
  "version": "0.1",
  "description": "What the dataset measures and where it came from.",
  "splits": {"all": ["case-001"]},
  "scorer": "exact_v0",
  "license": "source dataset license"
}
```

The description should state whether Episodes are single-turn or multi-turn, what is measured, whether examples are original or derived, and which examples are unscored.

## Targets and policy

`targets.jsonl` contains one evaluator-only record per scored Episode:

```json
{"episode_id":"case-001","turns":{"t1":{"answers":["example"]}}}
```

Target payloads are scorer-defined. Episodes without targets remain valid and are reported as unscored.

`policy.json` separates model-visible guidance from evaluator-only rules:

```json
{
  "public": {"guidance": "Model-visible instructions."},
  "hidden": {"evidence_tags": {}, "transitions": []}
}
```

The runner may place `public.guidance` in the system prompt. Hidden policy is available only during evaluation. `workflow_v0` uses `hidden.evidence_tags` to map Evidence IDs to the evaluator-only tags referenced by transition `when` conditions.

Built-in scorers:

- `exact_v0`: state or answer matching, allowed actions, required Evidence, and abstention.
- `workflow_v0`: state fields, deterministic transitions, action constraints, required Evidence, and cumulative success.
- `unscored`: operational measurements only.

## Importer contract

An importer is an offline conversion step. It should:

1. Preserve stable source identifiers and provenance.
2. Define the Evidence unit and Turn release rule explicitly.
3. Preserve known source times; use null for a missing single-turn time rather than guessing.
4. Keep all gold data out of `episodes.jsonl`.
5. Copy artifacts under `assets/` or write safe relative references.
6. Record exclusions, missing fields, truncation, source hashes, and temporal assumptions in `import_report.json`.
7. Run `ama validate` on the output.

Dataset-specific conversion and scoring belong in importers and scorers. They must not add branches to `run_episode()`.

## Dataset Card convention

Published and checked-in datasets should include `DATASET_CARD.md` and `DATASET_CARD.zh-CN.md`. Cards are recommended rather than validator-enforced in v0. Importers generate initial cards with these required sections:

```text
Source and license
Research use
Construction
Size
Scoring and targets
Artifacts
Limitations and reporting
```

Authors should replace generated text with source-specific details before publication. The card must disclose derived examples, missing targets, temporal proxies, truncation, and claims that the resulting scores cannot support.

## Dataset location

CLI commands accept an existing dataset directory. If the argument is not an existing path and `AMA_DATA_ROOT` is set, AMA resolves it below that root. This supports local disks and operating-system-mounted NAS paths without adding a storage protocol:

```bash
export AMA_DATA_ROOT=/Volumes/lab-data/ama
ama validate my_dataset
```

Run manifests store the resolved absolute directory. Import destinations remain explicit paths.

## Study guidance

For single-turn grounded reasoning, use one Turn, release all permitted context there, and score answers, citations, structured state, or abstention. A single-turn Episode may omit time.

For multi-turn memory, group observations by the time they became available, release only new Evidence in each Turn, and keep the sequence fixed. State in the data card whether the study measures recall, belief updating, consistency, or workflow decisions. Teacher-forced replay must not be described as an interactive patient simulation.

For derived benchmark data, distinguish original tasks from constructed Episodes. Scores on constructed longitudinal replays are not scores on the source benchmark unless the source benchmark defines that protocol.
