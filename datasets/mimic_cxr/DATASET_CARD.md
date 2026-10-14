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

The raw importer above remains `unscored`. Use the separate ReXrank preparation route below
for report-quality scoring. Its authoritative references differ; overriding the scorer name
alone does not convert a raw-importer dataset into the ReXrank task.

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

The supplied prompt now specifies only the task and inner answer fields; the runner defines
the Decision envelope. Historical format results do not validate this revised prompt.
Keep real images, reports, and individual run records in
ignored `local_data/`, outside the code submission.

## ReXrank fixed100 and RadGraph scoring

Use `data/mimic-cxr/ReXRank_MIMICCXR_test.json` from the
[pinned ReXrank revision](https://github.com/rajpurkarlab/ReXrank/tree/0a18a41e580992ac7cd32a8a9463009654f0d390).
Sort numeric study IDs, sample 100 without replacement using seed 42, then sort the sample:
100 studies, 164 images. The official `report` is the evaluator-only reference, bypassing
the raw section parser. Missing impression NaN values become empty strings only in the
auxiliary answer fields. Inputs include all current-study images and views, without context/history.

```bash
python3 -m pip install -e '.[dev,mimic-cxr,cxr-radgraph]'
export AMA_DATA_PATH=local_data
export REXRANK_JSON=/path/to/ReXRank_MIMICCXR_test.json
export IMAGE_ROOT=/path/to/mimic-cxr-jpg-2.1.0
bash scripts/run_mimic_cxr.sh freeze
bash scripts/run_mimic_cxr.sh import
bash scripts/run_mimic_cxr.sh setup
bash scripts/run_mimic_cxr.sh validate
```

`IMAGE_ROOT` must contain `files/` and the official split CSV. The complete runnable dataset
is `$AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42/dataset/`, with the frozen
manifest in the adjacent `cohort/`. Override `BENCHMARK_DIR` as needed. Prepared run/eval
does not require the original JSON or NAS images. freeze/import reject existing destinations.

```bash
bash scripts/run_mimic_cxr.sh --help
bash scripts/run_mimic_cxr.sh run YOUR_VISION_MODEL_ALIAS
bash scripts/run_mimic_cxr.sh eval runs/mimic-cxr-rexrank-fixed100-seed42/RUN_ID
# Existing CLI works directly too:
ama run "$AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42/dataset" \
  --model YOUR_VISION_MODEL_ALIAS --split test --max-calls 1 --timeout 300
ama eval runs/RUN_ID
```

The script reuses the AMA runner and adds no envelope contract or batch resume/retry logic.
Run records default to local `runs/mimic-cxr-rexrank-fixed100-seed42`, outside processed data.

`mimic_cxr_radgraph_v1` implements the entity/relation F1 average defined by
[ReXrank-metric](https://github.com/rajpurkarlab/ReXrank-metric/tree/d6be67831ef933dc91f01892303a40ebc6cc9549),
then macro-averages across expected studies. Invalid/missing reports score zero; valid-report
mean, failures and output format quality are reported separately. Extractor failures raise errors.
Inference uses pinned `radgraph==0.1.18`, `model_type=radgraph`, weights and tokenizer/config
on CPU offline. Setup accepts `HF_ENDPOINT` while retaining pinned hash checks; set
`AMA_RADGRAPH_CACHE` for a custom cache directory.

This is an image-only onboarding subset. Official metric functions agree on shared annotations;
equivalence to the legacy extractor remains unverified. Scores are neither full official leaderboard
results nor clinical diagnosis accuracy. Commit code, task instructions, dataset cards and synthetic
tests; keep real data, per-case results, plans and audit artifacts local.
