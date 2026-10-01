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

## Run the two-model, three-view matrix

Use the same model configuration within each model's three runs. Set temperature 0,
the same request timeout, and the same episode IDs and order. HPI and full-info have
no tools; interactive uses the MIMIC adapter. Run each view with one worker and pace
requests to stay below the endpoint's token quota. Give every run enough time to finish
and evaluate it with `ama eval` before comparison. Example for one model:

```bash
ama run /path/to/restricted/benchmark/mimic_cdm_open_hpi \
  --model MODEL --split all --timeout 120 --max-calls 24 --runs-root runs
ama run /path/to/restricted/benchmark/mimic_cdm_open_interactive \
  --model MODEL --split all --timeout 120 --max-calls 24 \
  --tools src/ama/importers/mimic_cdm_tools.py --runs-root runs
ama run /path/to/restricted/benchmark/mimic_cdm_open_full_info \
  --model MODEL --split all --timeout 120 --max-calls 24 --runs-root runs
ama eval runs/HPI_RUN_ID
ama eval runs/INTERACTIVE_RUN_ID
ama eval runs/FULL_RUN_ID
```

Repeat for the second model. `ama run` itself does not pace calls or retry 429s;
do not treat transport failures as diagnostic mistakes. Report both accuracy over
all selected cases and completion/transport failures. To aggregate completed runs:

```bash
PYTHONPATH=src python3 -m ama.importers.mimic_cdm_compare \
  --model MODEL_A runs/A_HPI runs/A_INTERACTIVE runs/A_FULL \
  --model MODEL_B runs/B_HPI runs/B_INTERACTIVE runs/B_FULL \
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
Treatment quality and citation support are not automatically adjudicated.

## Validity checks before interpreting a model gap

Inspect a blinded sample of the HPI, examination, and imaging text for labels,
discharge information, or later-care clues. The source lacks event timestamps for some
extracted fields. Do not select cases from previous model failures or successes; that
would bias a new comparison. Keep the episode list fixed, review unmapped diagnoses
without knowing model identity, and repeat on a held-out cohort before claiming
general model superiority. Differences in server load, 429 rates, context limits, and
provider thinking settings also limit attribution.
