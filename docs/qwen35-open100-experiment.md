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

## Budgeted execution follow-up

A subsequent execution request specifies the same 100 cases and three views,
with a CNY 35 ceiling. `scripts/qwen35_open100_budgeted.py` reuses the PR13
Agent, tool adapter, decision normalization, merger and scorer unchanged.
It makes one encounter per case per view, including unsuccessful encounters;
failed answers are not automatically resubmitted. Temperature 0, non-thinking,
4096 output tokens, 24 logical calls and timeout 120 s remain frozen.

The single-process ledger reserves 260096 input tokens plus the configured
maximum output for every HTTP attempt, before dispatch. Official Beijing
prices, checked on 2026-10-04, are input CNY 1.8/M and output CNY 10.8/M.
See [the model specification and prices](https://help.aliyun.com/zh/model-studio/qwen3-6-35b-a3b).
A successful response with valid usage refunds the unused reservation; failures,
unknown usage and crash-in-flight reservations remain charged conservatively.
All retries reserve separately, with at most three transport attempts. A new
request is refused if its reservation could exceed CNY 35. API usage outside
this runner is not part of its ledger. Model identity mismatch or exhausted
transport retries stop the batch. Interrupted runs preserve their records.

Budget fixtures verify crash persistence, refunds, malformed/unknown usage,
pre-dispatch refusal and retry accounting. The full offline suite now passes
110 tests. Execution outputs use owner-only file permissions and a process lock.

The first two launch requests were rejected by automatic approval review;
no synthetic or patient requests were sent. The second rejection states that
the specific payload, destination and cap in cross-thread transcript evidence
are not accepted as trusted authorization. Execution awaits explicit approval
in this thread. The command below remains pending:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3.12 \
  scripts/qwen35_open100_budgeted.py --cohort runs/reviewer-open100 \
  --out '/Users/bran/Documents/Projects/medical model training/runs/qwen35_open100_execution/v1'
```

## Authorized execution started

The user directly supplied the complete payload/destination/three-view/CNY35
authorization in this thread. The subsequent launch passed approval review.
The existing research-workspace credential is named `dashscope_api`; the
budgeted runner maps it to `DASHSCOPE_API_KEY` in process memory only.
A synthetic smoke returned the exact wire model `qwen3.6-35b-a3b`; clinical
execution then started in the fresh `runs/qwen35_open100_execution/v1` output
directory of the separate research workspace. Prior approval failures made no
requests. The execution artifacts, not this static note, are the live status.

## Completed single-pass result

All 300 planned encounters ran. Automatic diagnosis scores are HPI 53/100,
interactive 28/100 and full_info 85/100. There were 328 HTTP attempts including
one synthetic smoke, no transport failures, and one invalid decision caused by
an incorrect turn_id. Reported usage is 669740 input and 40347 output tokens;
the price-based total estimate is CNY 1.6412796 including the smoke.

The interactive view used tools in 11/100 cases. Its 72 noncorrect scores include
17 explicit abstentions, 41 other answers lacking a scorer-readable diagnosis,
9 unmapped answers, 4 different mapped source groups and 1 invalid decision.
Full_info has 10 unmapped answers and 5 different mapped source groups. These
are automatic mapping outcomes, not confirmed clinical error adjudications.
The scorer retains string answers; 57 full_info answers are strings rather than
nested answer.diagnosis objects. This does not change the archived primary score.

Local analysis is reproducible without API calls:

```bash
python3.12 scripts/analyze_qwen35_open100.py \
  --run '/Users/bran/Documents/Projects/medical model training/runs/qwen35_open100_execution/v1'
```

See report.md, analysis.json, budget.json and private/errors.json in that output.
No confidence AUC or repeated-generation consistency AUC is estimated by this
single-pass deployment baseline. No model superiority or training benefit is
established. Raw runs and failures were preserved without answer replacement.
