# VQA-RAD: AMA import and fixed test100 comparison

## 数据位置 / Prepared data

全量数据和固定100条验收包已由维护者上传到实验室 NAS：
`/mnt/nas/data/processed/VQA-rad`。这是内部存储位置，需要相应的 NAS 挂载和访问权限。
此上传状态由维护者提供，本次代码整理没有重新连接 NAS 核验。

| Package | Episodes | Images | Split |
|---|---:|---:|---|
| Full / 全量 | 2,244 | 314 | train 1,793; test 451 |
| Fixed test100 / 固定子集 | 100 | 84 | test only; 55 yes/no, 45 open |

NAS 上分别以包含 `dataset.json` 的目录作为 AMA 数据集路径；若子集有外层归档目录，
应指向其中的 `dataset/`。两包均含图像、问题、evaluator-only 答案、评分配置、双语数据卡、
来源记录和校验清单。无需重新下载或导入即可运行。不要将真实数据、逐例记录或密钥提交 Git。

## Source and migrated semantics

HF source: [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad/blob/bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9/README.md),
revision `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` (dataset card: CC0-1.0).
This is a public dataset; the source version is our reproducibility pin.

Official task code: AutoMedBench `eval_vqa` at
`5a19ff72491df7f8c0279438fcfade9364f9a8b7`:

| Evidence | Semantics retained |
|---|---|
| [Task config](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/tasks/vqa-rad-task/config.yaml) | Image + question, open and yes/no VQA. |
| [HF staging](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/stage_hf_datasets.py) | Defaults to all test rows, without a pinned HF revision. Limited staging selects sorted `random.Random(seed).sample` positions. Strips answers; raw stripped/lowercase `yes` or `no` defines `closed_yes_no`, otherwise `open`. |
| [Normalizer](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/answer_normalizer.py), [metrics](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/answer_metrics.py), [scorer](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/vqa_scorer.py) | Deterministic normalization, EM, token F1 and heuristic combination; missing/invalid answers score zero. |

The adapted scoring code retains the task branch's [license notice](licenses/AutoMedBench-LICENSE.txt):
non-commercial research only. The dataset's CC0 declaration is distinct from the code license.

## Architecture, data flow and failure modes

```text
Pinned local HF snapshot + source-manifest.json
  -> ama import vqa-rad
  -> dataset.json + episodes.jsonl + artifacts/ -> AMA execute -> saved decisions
  -> targets.jsonl + eval.json ----------------> AMA eval
```

- `src/ama/importers/vqa_rad.py`: `_snapshot` checks pinned version, source paths,
  both splits, file sizes and hashes. `import_vqa_rad` reads Parquet batches,
  emits one episode/turn per QA, deduplicates images, generates cards/audit files,
  validates and publishes a new directory. Existing outputs are refused; failed
  staging does not publish a partial package.
- `src/ama/importers/vqa_rad_scoring.py`: deterministic answer normalization and
  scoring, registered through the existing importer/scorer catalogs.
- `scripts/verify_vqa_rad.py`: independently checks questions, answers, source
  row IDs/order, splits, decoded pixels and checksums; oracle/empty checks test
  scoring plumbing, not model performance.
- `scripts/vqa_rad_benchmark.py`: freezes test selection and input/evaluator hashes,
  uses native `execute()` and eval, reuses existing retry/recovery/merge helpers.
  One persisted shard per QA; bounded transport retries, frozen returned answers,
  crash recovery, per-model lock, and mismatch rejection before resumption.
- `scripts/export_vqa_rad_cohort.py`: exports that exact frozen selection without
  resampling. Copies referenced images and unchanged episode/target/provenance
  records, writes a portable receipt and checksums, validates before publication.

Visible input contains only image and question. The task answer is
`{"text": "short English answer"}` inside AMA's Decision envelope. Reference answers
and answer-derived question types remain evaluator-only. The run loader does not
open targets/provenance; this application boundary is not a filesystem sandbox.
Do not give evaluated agents tools that expose evaluator files.

Source hash checks establish consistency with the supplied source manifest, not
independent authenticity. Unknown/duplicate decision IDs or turns and malformed
references fail evaluation; missing/invalid predictions stay in the denominator.

## Scoring and differences from AutoMedBench

Scorer: `vqa_rad_automedbench_v1`; metrics use 0–1 rates with numerator/denominator.

1. Lowercase, delete ASCII punctuation, collapse whitespace and strip.
2. Map whole-answer yes/no synonyms; otherwise normalize number words and
   standalone medical abbreviations using the explicit maps in the scorer.
3. Compute normalized exact match (EM) and multiset token F1. For normalized
   yes/no gold, heuristic = EM; otherwise heuristic = `(EM + F1) / 2`.
4. Macro-average over all expected questions, including failures as zero.

`TRUE!` matches `yes`; `yes, a lesion` does not. No article removal, stemming,
substring credit or semantic medical judge is added. Raw-answer question-type
counts and normalized yes/no scoring groups can differ.

Differences: AMA preserves train and test, uses stable split/source-row IDs,
keeps question type private, and saves metadata-free PNG instead of JPEG quality92.
AMA requires a single-string `text` answer object and completed-turn status.
We migrate the deterministic heuristic, not the optional LLM judge, Docker or
S1–S5 autonomous research workflow. Source train/test share 202 original image
hashes; QA-level test scores are not independent-image/patient estimates.

## Import, validation and export

Python >=3.11; `pip install -e '.[dev,vqa-rad]'`. Prepared NAS packages need no import.
For rebuilding, use the existing pinned local snapshot with its archived
`source-manifest.json` and `snapshot/` files. The initial downloader is retained in
local preparation records, outside this PR. A fresh HF download alone does not
supply the manifest required by the importer.

```bash
ama import vqa-rad --source /path/to/pinned-snapshot --out /path/to/new-full
PYTHONPATH=src python scripts/verify_vqa_rad.py \
  --source /path/to/pinned-snapshot --dataset /path/to/new-full
PYTHONPATH=src python -m pytest -q -m 'not api'
```

Full ordering: train then test, lexicographic shard order, physical Parquet row order.
Full package checksums use SHA256 of the sorted file/hash JSON map, compact
separators `(',', ':')`, excluding `checksums.json`. Runtime/card/name changes can
change whole-package hashes. Explicit source indices and hashes are recorded.

The test100 selection is `sorted(random.Random(42).sample(range(451), 100))`.
It is our fixed comparison subset, not an official designated 100-case cohort.

```bash
export PYTHONPATH=src
python -m scripts.vqa_rad_benchmark --dataset /path/to/full \
  --benchmark /path/to/new-cohort --count 100 --seed 42 --prepare-only
python -m scripts.export_vqa_rad_cohort --cohort /path/to/new-cohort \
  --out /path/to/new-test100/dataset
```

The standalone subset uses relative artifacts and needs no parent dataset.
`source_cohort.json` retains selection positions/IDs/hashes and the original cohort
hash but omits the machine path. Its checksums file states its aggregate algorithm.
The historical `cohort.json` and model runs retain their original paths/hashes;
copying them does not relocate a resumable historical experiment.

## Actual test100 results / 实测结果

Executed 2026-10-21, same frozen 100 QA inputs for all three PR #12 configurations:

| Alias | Request model | API | Max output tokens |
|---|---|---|---:|
| `glm-5.3-flash` | `glm-5.3-flash` | Zhipu Coding `/api/coding/paas/v4` | Unspecified |
| `qwen3.8-flash` | `qwen3.8-flash` | DashScope compatible | 8192 |
| `deepseek-v4.1-flash-1017` | `deepseek-flash` | DeepSeek | 8192 |

Temperature0, timeout180s, one call per episode, one-second delay, at most three
transport attempts. Provider thinking defaults retained. These settings match the
saved #12 manifests; differing caps and mutable provider backends limit comparisons.

| Model | Valid answers | Heuristic | EM | Token F1 | Yes/no | Strict AMA | Reported tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM | 97/100 | 0.6601 | 0.62 | 0.7002 | 48/55 | 93% | 163,286 |
| Qwen | 99/100 | 0.6713 | 0.63 | 0.7126 | 49/55 | 100% | 211,193 |
| DeepSeek | 100/100 | 0.5712 | 0.53 | 0.6125 | 42/55 | 100% | 162,157 |

GLM: three outputs lacked envelope `answer`; four additional answer-only outputs
were recovered but not strict. Qwen: one strict `answer=null` abstention. Failures
and abstention score zero among all100. Transport retry events:0/3/1 respectively;
returned outputs were never regenerated, and pilot outputs were reused.
Strict compliance measures format, not answer correctness. Results are descriptive
on this small subset, not clinical accuracy or official AutoMedBench results.

For a separately authorized new API run, use a local configuration with the above
settings and key environment variable names (never values in Git):

```bash
python -m scripts.vqa_rad_benchmark --benchmark /path/to/cohort \
  --runs-root /path/to/runs --config /path/to/models.json --env-file /path/to/.env \
  --model glm-5.3-flash --model qwen3.8-flash \
  --model deepseek-v4.1-flash-1017 --workers 3
```

The same command with `--max-new-cases 1` runs a reusable pilot. Saved output can
be evaluated without API calls using `--compare-only` instead of execution options.
Comparison checks frozen selection, config/code hashes, actual initial messages,
image hashes, terminal coverage and all-case denominators before writing results.

## Verification evidence

- Offline suite: **213 passed**, including synthetic import/scoring, answer
  isolation, split/sampling, retry/resume, crash recovery and portable export tests.
- Real full conversion: all2,244 source QA records/pixels checked;314 artifacts.
  Full and test100 file hashes, native structure and target isolation passed.
- Actual100-record package copied to a different directory and loaded successfully;
  original cohort bytes unchanged. No extra model API calls during export.
- Official-function comparison:566 normalization inputs and5,094 metric pairs
  matched within1e-12. Saved model outputs reproduced comparison byte-for-byte.
- Full aggregate hash: `30b76770ceb830d2e9f54324083a5550d80f1662af0a0969240aef2e309093ac`.
- Original comparison SHA256: `ee123b31e986a62b653fd86786798d4a300276eb85b4bf50c19a3f769b18732f`.

Detailed source locks, per-case provenance, outputs and local preparation records
remain outside Git. The comparison hash identifies the original path-bearing run,
not a promised byte-identical relocated reconstruction.
