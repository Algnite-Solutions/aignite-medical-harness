# Fixed VQA-RAD comparison using the PR #12 models

This experiment uses 100 QA records from the pinned VQA-RAD test split, sampled
with seed 42 and then kept in source order. The full processed dataset remains
unchanged. A separate cohort manifest freezes selected IDs, explicit source-test
indices, visible file/image hashes and evaluator hashes. Run records are stored
separately under ignored `data/experiments/`.

## Models and request controls

The local configurations were read from the MIMIC-CXR worktree and compared to
the saved 100-study run manifests underlying PR #12. All three matched exactly:

| Display alias | Request model | API | Temperature | Output token cap |
|---|---|---|---:|---:|
| `glm-5.3-flash` | `glm-5.3-flash` | Zhipu **Coding**, `/api/coding/paas/v4` | 0 | Not explicitly set |
| `qwen3.8-flash` | `qwen3.8-flash` | DashScope compatible API | 0 | 8192 |
| `deepseek-v4.1-flash-1017` | `deepseek-flash` | DeepSeek API | 0 | 8192 |

Each episode permits one model call (`max_calls=1`); request timeout is 180 seconds.
Provider thinking defaults are retained. This preserves the PR #12 differences
in decoding configuration and is a preliminary API comparison, not fully matched
decoding. The client reports provider usage; monetary cost is not inferred from
token counts or unverified pricing.

One worker per model may run concurrently, with a one-second delay before each
request and at most three attempts for transient transport failures. Returned
answers, including invalid JSON or failed output extraction, are frozen rather
than regenerated. After exhausted transport retries, that model stops and may
resume later; configuration, code or dataset changes prevent silent resumption.

The native AMA `execute()` function writes one persisted run per episode. The
existing batch merger creates a standard combined run and calls native AMA eval.
The comparison rechecks cohort membership, code/config hashes, actual initial
system/user messages, terminal decisions and evaluation denominators. Missing or
invalid answers remain in all-case means. No reference answer is included in a
generation request. Only preparation and evaluation open target files.

## Commands for this local experiment

From the VQA-RAD worktree, using Python with the `dev,vqa-rad` extras installed:

```bash
export PYTHONPATH=src
BENCHMARK=data/processed/vqa-rad-hf-bcf91e7654fb-test100-seed42-v1
EXPERIMENT=data/experiments/vqa-rad-pr12-test100-seed42-v1

python -m scripts.vqa_rad_benchmark \
  --dataset data/processed/vqa-rad-hf-bcf91e7654fb-v1-full \
  --benchmark "$BENCHMARK" --count 100 --seed 42 --prepare-only
```

The local `models.json` was frozen from the other worktree's configuration files;
it contains endpoint/model settings and API-key environment variable names, never
key values. `reference-configs.json` records the source-file and reference-run
manifest hashes. For a new independent reproduction, the public `ama.json` has a
`glm-5.3-flash-coding` alias for the same GLM endpoint; its ordinary
`glm-5.3-flash` alias uses the different non-Coding endpoint.

When API execution is authorized, start with one selected question per model:

```bash
python -u -m scripts.vqa_rad_benchmark \
  --benchmark "$BENCHMARK" --runs-root "$EXPERIMENT/runs" \
  --config "$EXPERIMENT/models.json" --env-file /path/to/local/.env \
  --model glm-5.3-flash --model qwen3.8-flash \
  --model deepseek-v4.1-flash-1017 --workers 3 --max-new-cases 1
```

Resume the remaining questions with the same command **without**
`--max-new-cases 1`. The saved first answers are reused, so this does not add an
extra paid pilot cohort. Once all models finish, `comparison.json` contains native
AMA task metrics, question-type breakdowns, output quality, terminations, reported
tokens, retry counts and paired score comparisons.

Re-evaluate and reconstruct the comparison without model calls:

```bash
python -m scripts.vqa_rad_benchmark \
  --benchmark "$BENCHMARK" --runs-root "$EXPERIMENT/runs" \
  --model glm-5.3-flash --model qwen3.8-flash \
  --model deepseek-v4.1-flash-1017 --compare-only
```

Each model directory contains `batch.json`, saved `shards/`, a `merged/` native
run after completion, and transport retry diagnostics when applicable. An OS file
lock prevents duplicate workers on one model batch. A crash after a persisted
response but before the batch update recovers that shard instead of querying the
same answer again. An ambiguous/incomplete shard stops for inspection.

## Status and limits

Preparation on 2026-10-21 froze **100 test records: 55 closed yes/no and 45 open**.
The offline repository suite passed **211 tests**, including seven new experiment
driver tests. These test synthetic data, crash/transport recovery, pilot reuse,
frozen invalid responses, evaluator isolation and matched-input comparison.

All three models have now completed 100 records each. Native evaluation and
byte-identical comparison reconstruction passed; see [actual results](vqa-rad-benchmark-results.md).
Oracle and FakeModel scores are not model results. This
experiment does not implement S1–S5 workflow scoring or an LLM judge. The source
train/test image overlap documented in the migration guide remains unchanged.

真实图像、问题、参考答案、病例 ID、逐例请求和模型输出均保留在本地忽略目录。
模型请求只发送图像、问题和 AMA 输出要求，不发送参考答案。配置沿用 #12 的三个
真实运行，GLM 使用 Coding 端点；固定 test100 队列包含 55 条 yes/no 和 45 条开放题。
真实实验结果必须以保存的运行及原生 AMA eval 为依据，不能用合成测试分数替代。
