# Matched MIMIC-CDM diagnosis benchmark

This protocol separates three questions on the **same admissions**:

| View | Evidence available to the model | Main comparison |
| --- | --- | --- |
| HPI | Presenting history only; no tools | Starting diagnostic ability |
| Interactive | HPI, then examination/labs/imaging through tools | Benefit and cost of evidence acquisition |
| Full info | All source evidence in one observation; no tools | Diagnostic ability with complete evidence |

Every view requests the same free-text `answer.diagnosis`, citations, and brief summary.
No candidate labels appear in the prompt. The importer still selects admissions from four
source pathology groups, so this is **unprompted four-group diagnosis**, not an open-world
clinical diagnosis benchmark. It is also not closed-book: every view supplies case evidence.
The HPI view is the closest available low-evidence baseline.

## Create a fixed, paired cohort

The builder reads existing processed MIMIC-CDM datasets; it does not reimport the raw CSVs
or call a model. It samples 25 admissions per source group with seed 42, independent of
model outcomes. All three generated datasets have identical admission order and hidden
targets. `benchmark.json` records that order and sampling rule.

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_benchmark \
  --full-info /path/to/processed/mimic_cdm_full_info \
  --interactive /path/to/processed/mimic_cdm_interactive \
  --out /path/to/restricted/benchmark \
  --per-class 25 --seed 42
```

The generated views are `mimic_cdm_open_hpi`, `mimic_cdm_open_interactive`, and
`mimic_cdm_open_full_info`. Keep them and their run logs on storage authorized for
credentialed MIMIC data. The interactive tool adapter remains
`src/ama/importers/mimic_cdm_tools.py` and reads only that view's selected cases.
Before inspecting pilot outcomes, freeze an independent confirmation cohort:

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_benchmark \
  --full-info /path/to/processed/mimic_cdm_full_info \
  --interactive /path/to/processed/mimic_cdm_interactive \
  --out /path/to/restricted/confirmation \
  --per-class 25 --seed 43 \
  --exclude-benchmark /path/to/restricted/benchmark/benchmark.json
```

## Run the multi-model, three-view matrix

Use the same model configuration within each model's three runs. Set temperature 0,
the same request timeout, and the same episode IDs and order. HPI and full-info have
no tools; interactive uses the MIMIC adapter. The batch runner processes one case at
a time, paces each request, retries HTTP 429, transient HTTP 5xx, and connection
timeouts with bounded backoff, saves each case,
and can resume by repeating the same command. `scripts/run_mimic_ext.sh` also
provides the sampling, run, evaluation, and pairwise comparison commands used
for the pilot below. Example for one model:

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_batch \
  --dataset /path/to/restricted/benchmark/mimic_cdm_open_hpi \
  --model MODEL --out runs/open100/MODEL/hpi --delay 3
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_batch \
  --dataset /path/to/restricted/benchmark/mimic_cdm_open_interactive \
  --model MODEL --out runs/open100/MODEL/interactive --delay 3 \
  --tools src/ama/importers/mimic_cdm_tools.py
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_batch \
  --dataset /path/to/restricted/benchmark/mimic_cdm_open_full_info \
  --model MODEL --out runs/open100/MODEL/full_info --delay 3
```

Repeat for the other models. Each batch writes `batch.json` and a standard evaluated
run in `merged/`; raw one-case runs stay in `shards/`. Repeat a command to retry
failed cases after a quota reset. Do not treat transport failures as diagnostic mistakes.
Report both accuracy over all selected cases and completion/transport failures.
To compare two completed models at a time:

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_compare \
  --model MODEL_A runs/open100/MODEL_A/hpi/merged \
    runs/open100/MODEL_A/interactive/merged runs/open100/MODEL_A/full_info/merged \
  --model MODEL_B runs/open100/MODEL_B/hpi/merged \
    runs/open100/MODEL_B/interactive/merged runs/open100/MODEL_B/full_info/merged \
  --out runs/paired-open-comparison.json
```

The comparison checks model configuration within each model, the actual system prompts,
matching case order and targets, benchmark protocol, timeout, tool availability, and
evaluation coverage. Read the
paired outcomes as:

- **Model difference:** compare models on full-info, with a paired case table.
- **Harness gain:** compare interactive against HPI for each model.
- **Evidence gap:** compare full-info against interactive for each model.

Also compare tool calls, model calls, tokens, citation validity, and output compliance.
The automatic free-text scorer recognizes only conservative equivalents of the four
source labels. It reports mapping coverage and lists ambiguous or unmapped answers
for blinded clinical review; `diagnosis_accuracy_auto` is a lower bound until review.
It scores a diagnosis string directly in `answer` when a model omits the requested
`answer.diagnosis` object, and reports `answer_object_format` separately.
Treatment quality and citation support are not automatically adjudicated.

## Seed-42 pilot results (100 admissions, 2026-10-02)

The fixed cohort has 25 admissions from each of four source pathology groups.
All three models saw the same admissions in the same order across HPI,
interactive, and full-info views. Temperature was 0; batches used a 120-second
request timeout, 24 model calls per case, and a 3-second request delay. All nine
views finished with 100/100 completed decisions after bounded transport retries.

| Model | HPI correct | Interactive correct | Full-info correct | Interactive tool calls | Interactive reported tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| AntAngelMed2 | 55/100 | 81/100 | 83/100 | 478 | 1,480,834 |
| DeepSeek-v4-flash | 60/100 | 85/100 | 87/100 | 562 | 1,822,718 |
| GLM-5.3-flash | 51/100 | 84/100 | 88/100 | 574 | 1,993,335 |

Scores are conservative automatic matches to the source group. Interactive
access improved the score over HPI by 26, 25, and 33 cases respectively;
providing all information directly improved it by another 2, 2, and 4 cases.
The one-case full-info difference between DeepSeek and GLM does not establish a
model ranking. In full-info, 13 AntAngel, 8 DeepSeek, and 9 GLM answers still
need blinded review because they are ambiguous or outside the lexical mapping.

Formatting is a separate result. In the interactive view, strict JSON envelope
compliance was 100/100 for AntAngel, 0/100 for DeepSeek, and 39/100 for GLM.
DeepSeek's 100 interactive decisions were nevertheless recoverable from JSON
following prose. The requested `answer.diagnosis` object appeared in 0/100
AntAngel interactive answers and 100/100 for each other model. All interactive
citations used released IDs, but their support for clinical claims was not
adjudicated. The local comparison report is
`runs/mimic-cdm-open-v1-100-live/review.md`; credentialed case content and raw
run logs are not included in this repository.

## Validity checks before interpreting a model gap

Inspect a blinded sample of the HPI, examination, and imaging text for labels,
discharge information, or later-care clues. The source lacks event timestamps for some
extracted fields. Do not select cases from previous model failures or successes; that
would bias a new comparison. Keep the episode list fixed, review unmapped diagnoses
without knowing model identity, and repeat on a held-out cohort before claiming
general model superiority. Differences in server load, 429 rates, context limits, and
provider thinking settings also limit attribution.
