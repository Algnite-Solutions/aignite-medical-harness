# PathVQA in AMA

This importer adapts the pinned [`flaviagiammarino/path-vqa`](https://huggingface.co/datasets/flaviagiammarino/path-vqa) Parquet snapshot at revision `7568940e604f3e911441b1f6c7aedb5f1c2866bd`. The source manifest and SHA-256 inventory are in [`source.lock.json`](source.lock.json). The Hugging Face entry is marked MIT; users should check the rights of the underlying pathology images before redistribution. No source image, question, answer, converted episode, or run record is committed here.

Each row becomes one single-turn AMA Episode with its image as Evidence. The answer is stored only in `targets.jsonl`, outside model observations. Image bytes are deduplicated by SHA-256. Source `train`, `validation`, and `test` splits are preserved. `automedbench_test_500` is a deterministic 500-case subset of the source test split, selected with Python `random.Random(42)` and ordered by source index. The PR baseline uses the 100 IDs in [`pr100.lock.json`](pr100.lock.json), stratified to 49 binary and 51 open questions. This is an AMA sample baseline, **not an official AutoMedBench score**.

The release cohort identity is [`pr100.v2.lock.json`](pr100.v2.lock.json). It locks the ordered IDs, complete episode and target files, selected questions, relative image paths and bytes, task instructions, and the effective AMA system prompt. The importer compares every Parquet shard with the separately packaged committed [`source.lock.json`](source.lock.json); editing the downloaded manifest cannot redefine the expected checksums. The source and task adaptation are pinned, but the 500-case and 100-case subsets are local AMA selections, not verified official AutoMedBench cohorts.

## Reproduce the v2 baseline

The three configurations are two models: Qwen3.6-27B with thinking enabled or disabled, and GLM-5.3-Flash at its provider default thinking setting. The public configuration is [`models.v2.json`](models.v2.json); it contains environment variable names, never credentials. Every formal request uses temperature 0, max_tokens 8192, timeout 300 seconds, one model call, no tools, and a one-second gap. A separate image smoke request for each configuration is excluded from the scores. Received incorrect, empty, or malformed responses remain frozen. Only transport failures and HTTP 408, 429, or 5xx get at most three attempts; each attempt is saved.

```bash
python3 -m pip install -e '.[pathvqa]'
export DATASET=/mnt/nas/data/incoming/Automedbench/PathVQA/ama/pathvqa-task
export RUN_ROOT=/home/aignitelab/projects/xh/pathvqa-pr100-v2
bash scripts/run_pathvqa.sh verify
# For a fresh import only: RAW=/path/to/raw OUT=/path/to/new/pathvqa-task bash scripts/run_pathvqa.sh import
# For a new cohort lock only: bash scripts/run_pathvqa.sh prepare
bash scripts/run_pathvqa.sh smoke --alias qwen36-nothink
bash scripts/run_pathvqa.sh smoke --alias qwen36-thinking
bash scripts/run_pathvqa.sh smoke --alias glm-5.3-flash
bash scripts/run_pathvqa.sh run --alias qwen36-nothink
bash scripts/run_pathvqa.sh run --alias qwen36-thinking
bash scripts/run_pathvqa.sh run --alias glm-5.3-flash
bash scripts/run_pathvqa.sh all
```

The `run` command resumes a partially completed cohort without repeating terminal cases. An uncertain in-flight request stops for inspection so that a response cannot silently be generated twice. `all` checks the lock, re-evaluates saved native AMA records, and writes the aggregate comparison. Detailed records stay on controlled storage. The provider token cap standardizes a request setting, not actual effective compute across services. Lexical overlap cannot establish clinical correctness.

## Unified v2 results

The saved [v2 aggregate](baseline.v2.json) covers the same 100 questions for two models and three configurations. Binary accuracy uses all 49 binary questions; open EM/F1 use all 51 open questions, including failures scored as zero.

| Configuration | Recoverable completion | Valid answers | Binary correct | Open EM | Open token F1 | Strict AMA output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B, thinking off | 99/100 | 98/100 | 30/49 | 0/51 | 0.0318 | 21% |
| Qwen3.6-27B, thinking on | 88/100 | 87/100 | 31/49 | 0/51 | 0.0634 | 0% |
| GLM-5.3-Flash, default | 96/100 | 94/100 | 36/49 | 1/51 | 0.1009 | 95% |

Qwen without/with thinking had one/twelve terminal `invalid_decision` records and three/one transport retries. GLM had four terminal HTTP 400 errors and no retries. Valid answers are nonempty normalized strings; binary valid choices must additionally be exactly `yes` or `no`. Strict format compliance and content validity are independent. Code hashes differ across these runs and are retained in the aggregate. Earlier experiments below used different request settings.

## Historical exploratory runs

The earlier [`baseline.json`](baseline.json) and [`qwen36-thinking-pr100.json`](qwen36-thinking-pr100.json) remain as historical records. They used different timeouts and output caps, so the table below is not the unified v2 comparison.

| Model alias | Completed | Valid answers | Binary accuracy (49) | Open exact match (51) | Open token F1 | Strict AMA format |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen36-nothink` | 96/100 | 95/100 | 29/49 (0.5918) | 0/51 (0.0000) | 0.0315 | 21/100 |
| `qwen36-thinking` | 90/100 | 89/100 | 30/49 (0.6122) | 0/51 (0.0000) | 0.0457 | 0/100 |
| `glm-5.3-flash` | 93/100 | 92/100 | 32/49 (0.6531) | 2/51 (0.0392) | 0.1083 | 93/100 |

All used temperature 0 and one API call per Episode. Qwen used a 90-second timeout in both runs; `qwen36-thinking` set `enable_thinking=true`, while `qwen36-nothink` set it to false. GLM retained provider defaults and used a 180-second timeout. The thinking run had 10 API timeouts, the non-thinking run had 4 terminal errors, and GLM had 7; failures stay in every all-case denominator. These settings differ, so this is a preliminary comparison. The open-answer scores are lexical measures and should not be read as clinical accuracy. The thinking run's strict AMA output compliance was 0/100 despite 90 recoverable answers.

On the DGX, the existing source snapshot is `/mnt/nas/data/incoming/Automedbench/PathVQA/raw`, and the staging AMA dataset is `/mnt/nas/data/incoming/Automedbench/PathVQA/ama/pathvqa-task`. Review and manual publication to `/mnt/nas/data/processed` happen after PR approval.

```bash
python -m pip install -e '.[pathvqa]'
ama import pathvqa \
  --source /mnt/nas/data/incoming/Automedbench/PathVQA/raw \
  --out /path/to/new/pathvqa-task
python -c 'from ama.data import validate_dataset; from pathlib import Path; errors = validate_dataset(Path("/path/to/new/pathvqa-task")); print(errors); raise SystemExit(bool(errors))'
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json --prepare-only
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json \
  --model qwen36-nothink --model qwen36 --model glm-5.3-flash
ama eval runs/pathvqa-pr100/<run-id>
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json \
  --summarize-run runs/pathvqa-pr100/<qwen-run-id> \
  --summarize-run runs/pathvqa-pr100/<glm-run-id> \
  --summary-out /path/to/results.json
```

The importer refuses to overwrite an existing destination and verifies source checksums by default. `--limit-per-split 1 --benchmark-count 0` supports a smoke import. The baseline script checks dataset/target hashes and source repository/revision against the frozen cohort before making requests. Use the basename `pathvqa-task` to reproduce the dataset metadata hash; local output paths in the import report do not define cohort identity.

`pathvqa_v1` reports completion and valid-answer rates separately from answer quality. Binary questions use strict normalized `yes`/`no` accuracy. Open questions use normalized exact match and mean per-case token F1, with a token-frequency intersection. Missing, invalid, failed, or nonconforming outputs remain in the appropriate denominator. These lexical metrics can miss clinically equivalent answers and cannot establish medical correctness.

For the Chinese instructions, see [README.zh-CN.md](README.zh-CN.md).
