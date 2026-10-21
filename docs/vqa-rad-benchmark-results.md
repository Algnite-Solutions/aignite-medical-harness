# VQA-RAD locally sampled test100: three-model API results

Executed on **2026-10-21** using the three PR #12 model configurations. These are
actual saved API outputs evaluated with native AMA, not FakeModel/oracle scores.

**This is our own fixed sample of 100 QA records, not an official designated
AutoMedBench 100-case cohort or an official leaderboard run.** We sampled from
451 HF test records using seed 42, sorted the selected indices into source order,
and used the same cohort for every model. It contains **55 yes/no and 45 open
questions, with 84 distinct image artifacts**; these are not 100 independent
patients. The official VQA-RAD staging defaults to the full test split.

Data revision: `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`.
Scorer: `vqa_rad_automedbench_v1`, adapting AutoMedBench task-code commit
`5a19ff72491df7f8c0279438fcfade9364f9a8b7`.

## All-case results

| Model | Valid answers | Heuristic ↑ | EM ↑ | Token F1 ↑ | Yes/no accuracy ↑ | Strict AMA | Reported tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `glm-5.3-flash` | 97/100 | 0.6601 | 0.6200 | 0.7002 | 48/55 (0.8727) | 93% | 163,286 |
| `qwen3.8-flash` | 99/100 | 0.6713 | 0.6300 | 0.7126 | 49/55 (0.8909) | 100% | 211,193 |
| `deepseek-v4.1-flash-1017` | 100/100 | 0.5712 | 0.5300 | 0.6125 | 42/55 (0.7636) | 100% | 162,157 |

All primary task metrics use all 100 expected records, including invalid outputs
and abstention as zero. Normalized yes/no uses strict whole-answer matching;
other gold answers use `(EM + token F1)/2`. F1 is averaged per question, not pooled
across all tokens. Strict AMA output compliance measures the Decision envelope,
not whether an answer is medically correct or even non-abstaining.

| Model | Open questions (45): heuristic | Open EM | Open token F1 |
|---|---:|---:|---:|
| `glm-5.3-flash` | 0.4002 | 0.3111 | 0.4893 |
| `qwen3.8-flash` | 0.4029 | 0.3111 | 0.4946 |
| `deepseek-v4.1-flash-1017` | 0.3361 | 0.2444 | 0.4277 |

## Output failures and transport

- **GLM:** 97 valid task answers. Three single-JSON responses placed `text` at the
  top level alongside envelope metadata but omitted `answer`, causing
  `incomplete Decision envelope without answer`. Four additional answer-only
  responses were recovered and scored, but were not strictly compliant. Thus
  strict output compliance is 93%. The diagnostic valid-answer-only heuristic is
  0.6805; the primary all-case
  score remains 0.6601.
- **Qwen:** 100 strictly compliant Decisions, including **one explicit
  `answer=null` abstention** with empty citations. Hence valid task answers are
  99/100 and the nonempty-valid-citation rate is 99%, while strict format remains
  100%. The abstention scores zero in the primary mean.
- **DeepSeek:** 100 valid answers and 100 strictly compliant Decisions.
- Logged transient connection retry events: GLM 0, Qwen 3, DeepSeek 1. All models
  ultimately produced 100 terminal records. Each has exactly 100 saved shards
  and 100 native logical model calls; the first-question pilot was reused.
  No received answer, including malformed output, was regenerated.

Provider-reported token totals cover all 100 saved terminal records per model.
They do not establish billed usage/cost for failed connection attempts. No price
or billing estimate is inferred.

## Matched inputs and request configuration

The local model configuration files from the MIMIC-CXR worktree matched its saved
PR #12 100-study manifests. GLM uses the **Coding** endpoint, Qwen uses DashScope,
and DeepSeek's request model ID is `deepseek-flash` (the display alias retains
`deepseek-v4.1-flash-1017`). Shared controls are temperature 0, timeout 180 seconds,
`max_calls=1`, one-second request delay and at most three transient transport
attempts. Qwen/DeepSeek set `max_tokens=8192`; GLM leaves the cap unspecified.
Provider thinking defaults are retained. The request identifiers/settings match
PR #12; provider-side weights and defaults are not independently version-pinned.
This is a preliminary API comparison, not fully matched decoding.

The actual saved initial system/user messages and locked image files match across
all three models for all 100 records. References are opened only for preparation
and evaluation and are not included in generation messages. Evaluation checked
cohort membership/order, source/evaluator hashes, code hashes, request settings,
expected turns, decision coverage and all-case denominators.

## Paired descriptive results

| Comparison (left vs right) | Left higher | Right higher | Ties | Mean left − right |
|---|---:|---:|---:|---:|
| `glm-5.3-flash_vs_qwen3.8-flash` | 13 | 14 | 73 | -0.0112 |
| `glm-5.3-flash_vs_deepseek-v4.1-flash-1017` | 24 | 9 | 67 | +0.0889 |
| `qwen3.8-flash_vs_deepseek-v4.1-flash-1017` | 24 | 10 | 66 | +0.1001 |

“Higher” compares each question's heuristic score; ties use tolerance `1e-12`.
These are descriptive results on one locally selected cohort, not significance
claims, clinical accuracy, independent-patient evidence or official AutoMedBench
scores. The source train/test image overlap and lexical scoring limitations in
[the migration guide](vqa-rad.md) still apply. S1–S5 and the optional LLM judge
are not reproduced.

## Reproduction and integrity

Re-running `--compare-only` recomputed native AMA evaluation from saved outputs
and reproduced `comparison.json` **byte-for-byte without model calls**. The final
code files were archived after checking their hashes against all three batches;
dependency versions and Git state are recorded locally. The offline suite passed
**211 tests**, including seven new synthetic experiment-driver tests.

Cohort manifest SHA-256:

```text
2633c46bff1e0a9238d6e5cadfe68ed5e4260f981b57a4e67ba0c029b49481fc
```

Comparison file SHA-256:

```text
ee123b31e986a62b653fd86786798d4a300276eb85b4bf50c19a3f769b18732f
```

The cohort and comparison hashes include local paths and identify this saved
experiment; they are not promised to match a relocated reconstruction. Source
indices, per-record input hashes, raw questions/images, targets, replies,
reasoning, usage records and per-case provenance remain under ignored `data/`.
Only aggregates and overall hashes are published here. See
[the experiment protocol and commands](vqa-rad-benchmark.md).

本次为自行固定抽样的 test100 三模型实验，不是官方指定的 100 例，也不是完整
AutoMedBench 排行榜评测。实际调用与 #12 相同的三个模型配置；主要分数对全部
100 条取平均，GLM 的 3 条解析失败和 Qwen 的 1 条主动弃答均保留分母并计零。
任务分数与格式合规分开报告；原始数据、逐例结果、密钥及来源映射不提交 Git。
