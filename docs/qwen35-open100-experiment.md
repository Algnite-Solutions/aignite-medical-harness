# Qwen3.6-35B-A3B: PR13 open100 experiment branch

## Material Passport

- Artifact: local deployment patch and offline preparation; 2026-10-04.
- Base: PR #13 merge, `d4ad61095f38517ca7f77994ea3c0cee40d670a1`.
- Cohort: existing seed-42 100 cases, 25 per source group; three views.
- Status: offline preparation. No new patient inference, sampling, training or model download.

## Configuration and scope

Use alias `qwen36-35b-a3b-bailian`; wire model is exactly `qwen3.6-35b-a3b`.
Credentials come from `DASHSCOPE_API_KEY`. The existing `qwen36` alias still means
the MEDAI Qwen3.6-27B endpoint. This patch adds no ModelScope dependency because
the selected deployment is Bailian's existing OpenAI-compatible HTTP service.

The new baseline uses temperature 0, thinking disabled and max_tokens 4096.
Bailian's thinking flag is a top-level wire parameter, as described in
[the official thinking API documentation](https://www.alibabacloud.com/help/en/model-studio/deep-thinking).
This differs from the previous development pilots (temperature 0.01, thinking
enabled); do not pool their results with this baseline. Returned model identity
must match the configured wire model. This is an API provenance check, not an
independent checkpoint attestation. Each diagnostic call records response model,
response ID, finish reason and whether logprobs were returned.

PR13 messages, tool IDs, evidence release, citations, task instructions, scorer
and cohort remain unchanged. No response rewriting or confidence injection is
needed: the existing parser preserves nested `answer.confidence` if provided.
The unchanged PR13 instructions do not require confidence, so the three-view
baseline cannot by itself estimate confidence AUC or the paper's five-repeat
consistency AUC. A separate frozen repeated-generation protocol remains needed
for that research question. Missing tool/final logprobs must remain missing.

## Offline preparation

Reuse `runs/reviewer-open100`; never call the sample command for this experiment.
The checker verifies 100 unique cases, identical case order and target bytes
across all views, and byte equality of dataset, episodes, targets, instructions
and evaluator configuration with the existing PR13 seed-42 cohort. Only counts
and hashes are emitted. Evaluator-only targets stay outside model requests.

```bash
cd ~/aignite-medical-harness
PYTHONPATH=src python3.12 scripts/check_qwen35_open100.py
PYTHONPATH=src python3.12 -m pytest -q -m 'not api'
```

## Pending inference commands (not executed)

Use the already configured environment credential; do not put its value in a
command, patch, log or manifest. A fresh output root prevents mixing old results.
The run command below sends patient content and is listed for a separately
authorized execution step.

```bash
cd ~/aignite-medical-harness
BENCHMARK_DIR=runs/reviewer-open100 \
RUNS_DIR=runs/qwen35-open100-pr13-v1 \
PYTHON_BIN=python3.12 \
bash scripts/run_mimic_ext.sh run qwen36-35b-a3b-bailian
```

The existing PR13 wrapper runs HPI, interactive and full_info sequentially,
with timeout 120 s, delay 3 s and at most 24 logical model calls per case.
For one pass, HPI and full_info should each make 100 calls; interactive can make
up to 2400. At 4096 output tokens per successful call, this is up to 10,649,600
output tokens, plus uncapped input tokens. This is not a monetary ceiling:
transport retries and repeated failed-case resumes can add billed requests.
Inspect truncation, failures, usage and retry logs before any repeat; establish
a provider spending limit before execution. Do not silently lower max_calls or
change evidence to fit a budget.

## Existing Qwen35 evidence

Two saved ten-case development pilots used this exact Bailian wire model, five
generations per case per version. Both had 7/10 majority-correct cases; these
were the same ten cases, not 100 distinct patients. V2 had consistency AUC 1.0,
P AUC 0.52381 and paired delta 0.47619, but only three incorrect cases and delta
bootstrap CI [0, 1]. These preliminary numbers do not establish robust paper
reproduction. The original summary artifacts are in the separate research
workspace under `runs/auc_reproduction/pilot10_full_info_v{1,2}/interface_summary.json`.

This hosted PR13 baseline is exploratory and does not certify on-premise
clinical reliability or reproduce the paper's original agent deployment.

## Local verification receipt

On 2026-10-04: 105 offline tests passed; the cohort checker verified all three
100-case views against the existing PR13 cohort. No new provider requests were
made. `git diff --check` passed. Validation used Python 3.12 from the existing
local environment with pytest API tests excluded.
