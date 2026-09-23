# medagentbench — Dataset Card

[中文](DATASET_CARD.zh-CN.md) | English

## Source and license

These examples were derived while importing MedAgentBench and its FHIR environment. Consult the upstream MedAgentBench license and access conditions before redistribution.

## Research use

The checked-in subset is a compact multi-turn example for longitudinal context and memory experiments over FHIR-derived patient records.

## Construction

Three patients were selected during import. Dated FHIR resources were sorted and grouped by day; the latest four populated days form four teacher-forced Turns per patient. Evidence from earlier Turns remains visible.

## Size

- Episodes: 3
- Turns: 12
- Evidence items: 25

## Scoring and targets

The subset uses `unscored` and contains no `targets.jsonl`. It reports completion and operational metrics only.

## Artifacts

No external artifacts are included. Each FHIR-derived Evidence item has a compact textual representation; source locators are in provenance.jsonl.

## Limitations and reporting

The sequence is a derived teacher-forced replay, not the original interactive FHIR task protocol. It has no turn-level clinical gold labels. Results must not be reported as original MedAgentBench benchmark scores.
