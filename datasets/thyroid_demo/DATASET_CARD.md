# thyroid_demo — Dataset Card

[中文](DATASET_CARD.zh-CN.md) | English

## Source and license

This is a synthetic toy dataset created for AMA. License: synthetic example data for repository testing.

## Research use

It demonstrates scored multi-turn workflow reasoning: the model updates a thyroid oncology state as ultrasound, contrast imaging, and pathology evidence arrive.

## Construction

One synthetic patient forms one Episode. Three dated reports are released in chronological Turns. Evaluator-only transition tags are stored in `policy.hidden`.

## Size

- Episodes: 1
- Turns: 3
- Evidence items: 3

## Scoring and targets

The dataset has one target record and uses `workflow_v0` to score state fields, transitions, actions, and citations.

## Artifacts

No external artifacts are included. All Evidence has model-readable text.

## Limitations and reporting

This is a toy workflow, not a clinical benchmark. Scripted-model results only test the harness and scorer and must not be presented as evidence of medical capability.
