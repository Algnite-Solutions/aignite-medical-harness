# PathVQA in AMA

This PR onboards PathVQA as a single-turn visual question answering task in AMA. The approved read-only AMA dataset is `/mnt/nas/data/processed/PathVQA/pathvqa-task`; the repository contains only importer code, locks, data cards, tests, and aggregate results.

The pinned source revision is `7568940e604f3e911441b1f6c7aedb5f1c2866bd`. The frozen local baseline has 100 cases: 49 binary and 51 open questions. It is an AMA sample baseline, not an official AutoMedBench score. `pr100.lock.json` separates public input identity from evaluation identity. `verify`, `run`, and `smoke` do not read `targets.jsonl`; `prepare`, `eval`, and `compare` do.

```bash
export DATASET=/mnt/nas/data/processed/PathVQA/pathvqa-task
export RUN_ROOT=/home/aignitelab/projects/xh/pathvqa-pr100
bash scripts/run_pathvqa.sh prepare
bash scripts/run_pathvqa.sh verify
bash scripts/run_pathvqa.sh run --alias qwen36-nothink
bash scripts/run_pathvqa.sh eval --alias qwen36-nothink
bash scripts/run_pathvqa.sh compare --alias qwen36-nothink
```

A received answer is never regenerated. After Ctrl-C, resume requires explicit `--acknowledge-interrupted`. No demo data is committed.

