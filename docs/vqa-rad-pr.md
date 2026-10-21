# Prepared PR

No automatic commit, push or PR edit was performed.

Title:

```text
Onboard AutoMedBench VQA-RAD task and fixed test100 comparison in AMA
```

Description:

```markdown
## Summary

Onboard AutoMedBench's VQA-RAD task in AMA and add a reproducible local test100
comparison using the three model configurations from PR #12. Each QA row becomes
one single-turn image/question episode; reference answers and answer-derived
question types remain evaluator-only.

- Pin and verify the HF source revision; preserve train/test membership.
- Generate full AMA data, bilingual cards, local provenance and checksums.
- Adapt the deterministic VQA heuristic, keeping missing/invalid answers in the denominator.
- Freeze a locally sampled test100 cohort with source indices and file hashes;
  persist native per-case runs, reuse pilots, resume transport failures and freeze received answers.
- Verify identical actual input messages/image hashes, request settings, code versions,
  cohort coverage and native evaluation before writing a comparison.

## Validation

- Full offline suite: **211 passed**, including 25 importer/scoring tests and seven experiment-driver tests.
- Real conversion: train 1,793 / test 451 QA episodes; separate 8-record acceptance subset.
- Source rows, answers, split/order, decoded pixels and generated checksums verified.
- 566 normalization inputs and 5,094 metric pairs matched fixed official reference functions.
- Three actual API batches completed 100 terminal records each. Saved-output evaluation
  reproduced the comparison byte-for-byte without further model calls.
- Pilot responses were reused; no received answers were regenerated. Raw data,
  per-case outputs, provenance and credentials remain in ignored local storage.

## Local test100 results

**The 100 QA records are our seed-42 sample from the 451-record HF test split,
not an official designated AutoMedBench cohort.** They comprise 55 yes/no and
45 open questions with 84 distinct image artifacts.

| Model | Valid answers | Heuristic ↑ | EM ↑ | Token F1 ↑ | Yes/no accuracy ↑ | Strict AMA | Reported tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `glm-5.3-flash` | 97/100 | 0.6601 | 0.6200 | 0.7002 | 48/55 (0.8727) | 93% | 163,286 |
| `qwen3.8-flash` | 99/100 | 0.6713 | 0.6300 | 0.7126 | 49/55 (0.8909) | 100% | 211,193 |
| `deepseek-v4.1-flash-1017` | 100/100 | 0.5712 | 0.5300 | 0.6125 | 42/55 (0.7636) | 100% | 162,157 |

All task means include all 100 expected records. GLM has three malformed
Decisions without `answer` and four recovered answer-only outputs. Qwen's one
explicit abstention is strictly compliant but scores zero, explaining 99 valid
answers versus 100% strict format compliance. Logged transport retry events were
GLM 0, Qwen 3 and DeepSeek 1.

Shared controls: temperature 0, max_calls 1, timeout 180 seconds. GLM uses the
Coding endpoint and no explicit output cap; Qwen/DeepSeek set max_tokens 8192.
Provider thinking defaults are retained, so this is not fully matched decoding.

## Scope

This migrates the underlying VQA task, not the full S1–S5/Docker research-agent
workflow or optional LLM judge. Scores are lexical heuristic results on one local
cohort, not clinical correctness or official leaderboard results. Source
train/test share 202 image-byte hashes; these are QA-row splits, not independent
patient/image splits. The task branch's license notice is preserved separately
from the HF data card's CC0-1.0 declaration.

See docs/vqa-rad.md, docs/vqa-rad.zh-CN.md, docs/vqa-rad-validation.md,
docs/vqa-rad-benchmark.md and docs/vqa-rad-benchmark-results.md.
```

## Manual commit and push

Review and execute manually. Do not force-add local data or run records.

```bash
cd /Users/bob/Projects/Research/aignite-medical-harness-vqa-rad
git status --short
git diff --check
git add scripts/vqa_rad_benchmark.py tests/test_vqa_rad_benchmark.py \
  docs/vqa-rad.md docs/vqa-rad.zh-CN.md docs/vqa-rad-validation.md \
  docs/vqa-rad-benchmark.md docs/vqa-rad-benchmark-results.md docs/vqa-rad-pr.md
git diff --cached --stat
git -c user.name=Bob -c user.email=1004553126@qq.com commit \
  -m "Add fixed VQA-RAD test100 comparison with PR12 models"
git push -u origin onboard-vqa-rad
```
