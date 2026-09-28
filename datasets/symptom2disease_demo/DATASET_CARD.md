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

Scorer: `unscored`. Target records: 6.

## Artifacts

No external artifacts; symptom text is embedded in Evidence. Source row numbers are in `import_report.json`.

## Limitations and reporting

This six-example demo is not an official test split or benchmark. There is no task accuracy scorer yet. Dataset-label prediction is not clinical diagnosis accuracy.

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

For an optional model-backed smoke run, configure a model alias and its key as described in `.env.example`, then run one demo Episode. Use the run directory printed by `ama run` for the next command:

```bash
ama run datasets/symptom2disease_demo --model qwen36 --episode s2d-000001
ama eval runs/<run-directory>
```

The `all` split includes all six examples. `eval` reports completion only while the scorer is `unscored`.
