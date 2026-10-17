# MIMIC-CXR data and task onboarding in AMA

## Task

This adapter brings the AutoMedBench MIMIC-CXR findings-generation task into native AMA.
Each study is one episode and one turn, with every supplied JPEG view as image evidence.
The model returns an AMA Decision with `answer={"findings":"..."}`. Reference findings
are evaluator-only in `targets.jsonl`; the inference loader does not read them.

Sources: [paper](https://arxiv.org/abs/2606.01961) and [pinned task configuration](https://github.com/AutoMedBench/AutoMedBench/blob/a4e64e5502c728cee744edce2b596000fcd60fdd/eval_report_gen/mimic-cxr-report-task/config.yaml).
This preserves study grouping, multiple views and findings-only generation while adapting
execution and output to AMA. It does not reproduce the paper's agent protocol or official
cohort. Scoring uses upstream's lightweight metrics. Docker is not required.

## Import from raw data

Install the image dependency first: `python3 -m pip install -e ".[mimic-cxr]"`.
The preparation script and helper default to `.venv/bin/python`; set `PYTHON_BIN`
for the helper if using another environment, and use that interpreter for preparation.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_mimic_cxr_raw.py \
  --source /path/to/MIMIC-CXR-JPG --out local_data/mimic_cxr/raw_findings100 \
  --split test --limit 100 --report-mode findings-only
ama import mimic-cxr --source local_data/mimic_cxr/raw_findings100 \
  --out local_data/processed/MIMIC-CXR-AutoMedBench-findings100-v1/dataset \
  --split test --report-mode findings-only
```

Explicit FINDINGS sections are required; full reports are never substituted for missing
findings. The existing default `findings-impression` mode remains available. Set the mode
on both preparation and import. Automatic raw selection covers single/multiple views and
then follows study order; it is not a random sample. Selection rules and exclusions stay local.

## Reuse frozen AMA CXR cases

```bash
bash scripts/run_mimic_cxr.sh reuse \
  local_data/processed/MIMIC-CXR-ReXrank-fixed100-seed42/dataset
```

This calls `ama import mimic-cxr --source-format ama --report-mode findings-only --split test`.
It preserves episode/evidence IDs, split order, views and image bytes; each JPEG is decoded
and checked against its source provenance hash. References come directly from structured
`targets.answer.findings`, unchanged. Old full-report references, instructions and scorer
are not inherited. Existing destinations are rejected. Optional `--study-id` filtering and
`--limit` prefix selection never resample cases. The previous seed-42 100-study cohort is a
local validation subset, not the official AutoMedBench cohort.

## Run and inspect

```bash
bash scripts/run_mimic_cxr.sh run glm-5.3-flash-coding
bash scripts/run_mimic_cxr.sh eval local_data/automedbench/runs/mimic-cxr-ama/<run-id>
```

The helper wraps native `ama import/run/eval`. Override `DATASET_DIR`, `CONFIG`, `RUNS_DIR`,
`TIMEOUT` or `MAX_CALLS` as needed. Extra AMA run options such as `--episode` pass through.
Keys are read from environment variables or local `.env`, never committed. The runner runs
on the machine launching it; model inference uses the configured API. References are not sent.

The following aliases are available in the committed `ama.json`. Add your own keys
to local `.env` using the variable names in `.env.example`.

| AMA alias | Provider model ID | Key variable |
| --- | --- | --- |
| `glm-5.3-flash-coding` | `glm-5.3-flash` (GLM Coding endpoint) | `GLM_API_KEY` |
| `qwen3.8-flash` | `qwen3.8-flash` (DashScope endpoint) | `DASHSCOPE_API_KEY` |
| `deepseek-v4.1-flash-1017` | `deepseek-flash` (DeepSeek endpoint) | `DEEPSEEK_API_KEY` |

Replace the alias in the run command to select a model. Each run sends study images
to the selected provider; confirm source access terms and permission for that destination.
The existing `glm-5.3-flash` alias still uses the standard GLM endpoint; the separate
Coding alias matches the endpoint used in the local CXR experiment. The DeepSeek
alias records the experiment's version label, but its provider model ID is mutable.

These configurations preserve the recorded request settings: Qwen and DeepSeek
use `max_tokens=8192`; GLM leaves it unset. All request `temperature=0` and leave
thinking behavior to the provider. This does not ensure identical effective
generation settings or token budgets across providers. Use a separate `CONFIG`
for a new controlled comparison and report its settings with the results.

Acceptance checks are valid import structure, correct image/reference pairing, hidden
references and saved predictions from native AMA. Findings-only import defaults to `mimic_cxr_automedbench_v1`, which adds lightweight
content metrics alongside execution/Decision-format diagnostics. The legacy
findings-impression import remains `unscored`.
Nonempty findings, format compliance and clinical correctness are distinct. Keep failed
cases in the denominator.

## Files and limitations

The runnable dataset contains dataset.json, episodes.jsonl, instructions.txt and copied
JPEG artifacts. targets.jsonl stores only the requested reference sections. Local
provenance.jsonl and import_report.json record hashes and source selection; cohort reuse
also retains source file hashes and source_cohort.json when present. Native run records
contain manifest, messages, decisions and diagnostics, with metrics after eval.

Keep real cases, reports, identifiers, provenance and run outputs under ignored local_data/.
Respect source data access terms; do not redistribute clinical cases with code. Retain old
local experiments separately. Migration checks do not establish clinical performance or
comparability with the paper's leaderboard.

## Report scoring

The lightweight pair metrics and format checker are retained from upstream revision
a4e64e5502c728cee744edce2b596000fcd60fdd. Unused backend imports and corpus helpers
are removed; pair-scoring functions are unchanged. Upstream and retained-file hashes,
modifications and the research-use license are recorded in
src/ama/_vendor/automedbench_report/. AMA adapts data access and aggregation; scoring
calls no remote model or judge API and requires no clinical model weights.

This adapter uses upstream's **lightweight** metrics: 12 regex-derived observation labels
and the upstream ROUGE-L tokenizer/LCS/F1 implementation. These are not CheXbert or
RadGraph labels. Upstream pair metrics round to four decimal places.

- mean_observation_f1 and mean_rouge_l_f1: per-case means with invalid cases zero-filled.
- case_mean_score: mean of 0.7 * observation F1 + 0.3 * ROUGE-L F1, an AMA diagnostic.
- valid_case_mean_score: valid-subset mean, always report coverage with it.
- complete_submission_score: applies upstream's all-reports-valid gate; otherwise zero.

Report validation uses upstream's exact 40–8000 character, 20-letter and printable-character
checks. This is distinct from strict AMA JSON compliance. Recovered Decisions can be scored;
unparsed replies are not reinterpreted by the scorer. Missing/invalid outputs receive zero
in AMA case means, including cases whose empty rule-label sets would otherwise earn F1=1.
Higher scores mean greater agreement under these metrics, not verified clinical accuracy.
Do not present lightweight values as the paper's default seven-metric result.

Re-score existing outputs without model API calls:

```bash
bash scripts/run_mimic_cxr.sh eval /path/to/run --scorer mimic_cxr_automedbench_v1
```

## Directory convention

The runnable dataset defaults to `$AMA_DATA_PATH/processed/MIMIC-CXR-AutoMedBench-findings100-v1/dataset`, with `AMA_DATA_PATH=local_data` unless overridden. Run records stay separate. Set `DATASET_DIR` to use an existing dataset. Historical experiment directories are retained so saved absolute run-manifest paths remain valid.
