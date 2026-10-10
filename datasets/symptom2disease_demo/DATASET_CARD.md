# symptom2disease_demo — Dataset Card

## Source and license

Source: [NeuronZero/Symptom2Disease](https://huggingface.co/datasets/NeuronZero/Symptom2Disease), a public copy of the [Kaggle dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease). The source CSV SHA-256 and selected source rows are in `import_report.json`.

License: The Hugging Face copy declares Apache-2.0. The original Kaggle source terms have not been independently verified.

## Research use

Single-turn research classification of symptom text into source disease labels.

## Construction

Six CSV records (rows 2, 52, 102, 152, 202 and 252) were selected from different disease labels. Each becomes one Episode and one Turn. Text is visible Evidence; the label is evaluator-only.

## Size

- Episodes: 6
- Turns: 6
- Evidence items: 6

## Scoring and targets

Scorer: `symptom2disease_v1`. Target records: 6. The task presents the 24 labels from the source CSV and asks for exactly one `answer.disease` choice. The scorer accepts either `{"disease": "<label>"}` or a single label string, ignores case and extra whitespace, and requires an exact source-label match. Other text, aliases, multiple labels, missing answers and failed turns score zero. Top-1 accuracy uses all six expected Episodes as its denominator; completion, valid-choice rate and answer-object format are reported separately.

## Artifacts

No external artifacts; symptom text is embedded in Evidence. Source row numbers are in `import_report.json`.

## Limitations and reporting

This six-example demo is not an official test split or benchmark. The source dataset does not prescribe this AMA 24-choice prompt or scoring protocol. Demo accuracy is only a pipeline smoke check, not a published Symptom2Disease benchmark result or clinical diagnosis accuracy.

## How to test

From the repository root, install the project as described in the README. Validate the committed demo and run the importer tests without a model key:

```bash
python3 -c "from pathlib import Path; from ama.data import validate_dataset; assert not validate_dataset(Path('datasets/symptom2disease_demo'))"
python3 -m pytest -q tests/test_symptom2disease.py
```

To test the importer on the full source CSV, choose a new output directory and run:

```bash
ama import symptom2disease --source /path/to/Symptom2Disease.csv --out /tmp/symptom2disease-check --row 2 --row 52 --row 102 --row 152 --row 202 --row 252
```

For an optional model-backed smoke run, configure a model alias and its key as described in `.env.example`, then run all six demo Episodes. Use the run directory printed by `ama run` for the next command:

```bash
ama run datasets/symptom2disease_demo --model qwen36
ama eval runs/<run-directory>
```

The `all` split includes all six examples. `eval` reports top-1 disease-label accuracy and separate output diagnostics.

## Fixed 100-case, multi-model evaluation

This is an AMA research protocol for the original 24-label `label,text` CSV, not an official Symptom2Disease test split or a clinical diagnosis benchmark. Each Episode shows one symptom description and asks for exactly one of the 24 source labels. `symptom2disease_v1` scores an exact top-1 label match after case and whitespace normalization. Missing, malformed, and out-of-list answers count as wrong; answer format and completion are reported separately.

### One command to prepare, run, score, and compare

From the repository root, with the source CSV available locally and each model key in its configured environment variable:

```bash
export AMA_DATA_PATH=/path/to/data
bash scripts/run_symptom2disease.sh --help
bash scripts/run_symptom2disease.sh all deepseek-v4-flash antangelmed2 qwen36
```

Aliases come from `ama.json`; set their `api_key_env` variables before running. Override `CONFIG` to select another AMA configuration; `PYTHON_BIN` defaults to `.venv/bin/python`. The command does not store key values.

`import` imports the full CSV into `$AMA_DATA_PATH/processed/Symptom2Disease` and rejects an existing destination. `sample` prepares the fixed 100 cases; `run MODEL ...` runs and scores, `eval MODEL ...` re-scores, and `compare MODEL ...` compares completed models. `all MODEL ...` prepares, runs, scores, and compares in sequence. `sample`, `eval`, and `compare` make no model calls.

The fixed cohort is stored at `$AMA_DATA_PATH/processed/Symptom2Disease-fixed100-seed42/dataset/`, a complete dataset usable directly with `ama run` and `ama eval`; the adjacent `benchmark.json` records the sampling protocol. Run records default to local `runs/symptom2disease-fixed100-seed42/`. Override `SOURCE_CSV`, `PROCESSED_DIR`, `BENCHMARK_DIR`, and `RUNS_DIR` as needed. Setting `BENCHMARK_DIR` is sufficient to run prepared cases without the raw CSV. Use new cohort and run directories for a fresh experiment.

The Python entry point also accepts `--benchmark /path/to/processed/cohort --out /path/to/runs --model MODEL`, plus `--prepare-only`, `--run-only`, and `--compare-only`. Omitting `--benchmark` preserves the legacy combined dataset/run layout under `--out`.

### Fixed cohort and run contract

The script excludes repeated symptom texts after whitespace normalization and case folding, keeping the first CSV row. A repeated text with conflicting labels is rejected. With seed 42, it chooses four unique rows from each of 24 labels, then one extra row from four seeded labels. The resulting 100 CSV row numbers and Episode IDs are sorted and recorded in `benchmark.json` with the source SHA-256, label counts, scorer version, and dataset file hashes. Rerunning with a different source or changed dataset fails instead of silently replacing the cohort. The source CSV remains local and is not committed.

Each selected Episode is sent independently through AMA's canonical `execute()` path. By default, runs use one model call per Episode, temperature from the model alias, a 300-second request timeout, a one-second request delay, and at most three transport attempts per case. The same actual system prompt and Episode order are required across compared models. The comparison also requires matching timeout, call limit, temperature, and output-token limit; provider-specific options remain visible in each model's saved configuration.

The batch retries only connection failures, HTTP 408/429/5xx, and provider responses with no usable assistant message, using bounded backoff. If these attempts are exhausted, the command stops, retains completed case records, and can resume with the same command. Once an assistant message is recorded, that case's answer is frozen. On resume, the batch checks saved AMA shards before making another request, including a shard completed just before an interrupted `batch.json` write. A reply AMA cannot parse as a Decision is usually recorded as `invalid_decision`; a malformed assistant message can be recorded as `api_error`. Both count as wrong. Permanent API errors stop the batch and require a corrected configuration and new run directory. No comparison file is produced while transport or provider failures remain unresolved.

### Results and interpretation

`$RUNS_DIR/<alias>/batch.json` tracks attempts and settings; `shards/` preserves individual AMA runs; `merged/` contains the standard AMA run and `metrics.json`. `comparison.json` is written only after every requested model has 100 terminal case records. It validates the cohort, actual prompt, model settings, decision coverage, scorer, and target file hashes, then reports accuracy, completion, formatting, terminations, token usage, and paired case outcomes for each model pair.

The fixed 100 cases are a small sampled evaluation. Reusing them to tune prompts or settings would make a later score on the same cases exploratory. Neither these scores nor pairwise differences establish clinical performance or a definitive model ranking.
