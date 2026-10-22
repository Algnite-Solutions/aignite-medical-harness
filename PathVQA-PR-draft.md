# Onboard PathVQA visual question answering task in AMA

## Summary

This PR adds a reproducible PathVQA importer, lexical scorer, and fixed 100-case AMA baseline. The approved read-only dataset is published on NAS at `/mnt/nas/data/processed/PathVQA/pathvqa-task`; no images, targets, demo dataset, or per-question logs are committed.

## Reproduction

Use `scripts/pathvqa.py` through `scripts/run_pathvqa.sh` with `prepare`, `verify`, `run`, `eval`, and `compare`. Runtime requests use public inputs only. Reference targets are loaded during evaluation. The baseline has 49 binary and 51 open questions and is not an official AutoMedBench score.

## Review feedback

- The AMA dataset was moved from `incoming/Automedbench/PathVQA/ama/pathvqa-task` to the read-only `processed/PathVQA/pathvqa-task` release path.
- Repository demo data was removed; runs read the NAS dataset directly.
- The duplicate v1 implementation and historical cohort/result files were removed. `scripts/pathvqa.py` is the only supported entry point.
- `run` verifies public inputs without opening `targets.jsonl`; `eval` and `compare` validate the evaluation identity before scoring.
- Ctrl-C creates one interrupted record and stops the whole batch. Resume requires `--acknowledge-interrupted` and never overwrites the original record.

