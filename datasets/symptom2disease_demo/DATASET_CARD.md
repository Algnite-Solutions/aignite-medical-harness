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
