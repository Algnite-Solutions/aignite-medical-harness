# MIMIC-CXR — Dataset Card

## Source and purpose

Current-study report generation using [MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
Full reports come from MIMIC-CXR and must be supplied separately. PhysioNet credentialed data
terms apply. This directory contains documentation and a prompt only, with no real images or
reports; it is not a runnable demo dataset.

## Construction and target isolation

One study becomes one single-turn Episode. All current-study images, sorted by dicom_id,
become Evidence with ViewPosition when available. Findings and impression are stored only
in evaluator targets. Official train/validate/test splits are preserved. Prior images,
reports, and disease labels are not inputs.

The custom `explicit-sections-v1` parser requires explicit, nonempty findings and impression
sections, without official-parser aliases or last-paragraph fallback. Extraction failure does
not prove that clinical content is absent. Automatic selection reserves the first eligible
single-image and multi-image studies in numeric study-ID order, then fills in that order.
IDs do not imply chronology. Repeat `--study-id` for explicit selection and set limit to at
least the number of requested IDs.

## Validated size and limitations

Local subset: 3 validation studies, 3 turns, 7 JPGs, and 3 targets. These are prompt-development
cases, not an independent test set. The offline suite passed 83 tests. Three GLM prompt trials
passed output-format validation in 2/3, 2/3, and 3/3 cases respectively; these are not clinical
accuracy scores or evidence of long-term reliability.

Scorer: `unscored`; evaluation aggregate is null. Report-quality scoring and public-baseline
numerical reproduction remain unfinished. Current-only inputs and strict section extraction
are not guaranteed to match a published evaluation protocol.

## Usage

Run from the repository root with Python 3.11+:

```bash
python3 -m pip install -e '.[dev,mimic-cxr]'
python3 -m pytest -q
```

The source must contain `files/`, the split and metadata CSV.gz files, and
`mimic-cxr-reports.zip`. Report ZIP member paths must end in
`files/pXX/p<subject_id>/s<study_id>.txt`. Replace the source path below; output directories
must not already exist.

```bash
python3 scripts/prepare_mimic_cxr_raw.py \
  --source /path/to/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/raw_validate_demo --split validate --limit 3

ama import mimic-cxr \
  --source local_data/mimic_cxr/raw_validate_demo \
  --out local_data/mimic_cxr/processed_validate_demo --split validate
```

The raw package contains actual image/report copies and can be imported independently.
The importer validates staged output before publishing it, generating bilingual cards,
provenance, and import counts. Corrupt images, hash mismatches, conflicting associations,
and existing destinations fail; automatic selection records missing-section exclusions.

Configure a vision-model alias in `ama.json` and its key in `.env`, then replace the alias:

```bash
ama run local_data/mimic_cxr/processed_validate_demo \
  --model YOUR_VISION_MODEL_ALIAS --split validate \
  --instruction-file datasets/mimic_cxr/instructions.txt \
  --runs-root local_data/mimic_cxr/runs --max-calls 1 --timeout 120

ama eval local_data/mimic_cxr/runs/<run-directory>
```

The supplied prompt is the v3 text used in the third trial; `--instruction-file` overrides
the importer's default prompt. Keep real images, reports, and individual run records in
ignored `local_data/`, outside the code submission.
