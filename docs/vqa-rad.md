# VQA-RAD migration from AutoMedBench

[中文说明](vqa-rad.zh-CN.md)

This is an offline dataset/task migration into AMA. It does not reproduce the
AutoMedBench autonomous research workflow or establish clinical/model performance.
Only VQA-RAD is implemented here; PathVQA and MIMIC-CXR are separate work.

## Source and official behavior

The source is `flaviagiammarino/vqa-rad`, pinned to
`bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`. The [dataset card at this
revision](https://huggingface.co/datasets/flaviagiammarino/vqa-rad/blob/bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9/README.md)
declares CC0-1.0. Pinning this revision is our reproducibility choice; the official
staging code calls `load_dataset(repo_id, split="test")` without a revision, so this
does **not** establish which source bytes any historical leaderboard run used.

Official code was inspected on `eval_vqa` at
`5a19ff72491df7f8c0279438fcfade9364f9a8b7`:

| Official source | Verified semantics |
|---|---|
| [Task config](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/tasks/vqa-rad-task/config.yaml) | `image+text`, `open_ended`, mixed open and yes/no; smoke min/max 1/10, time limit 3600 s. |
| [HF staging](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/stage_hf_datasets.py) | `_stage_simple` reads test only; VQA-RAD defaults to all rows (`--vqa-rad-count -1`), seed 42. Limited staging uses `sorted(random.Random(seed).sample(range(N), count))`. All-row staging preserves order. IDs are assigned by selected position, `vqarad_{pos:05d}`. Answers are stripped; labels are null. Image/question go under public, answer under private. Images are RGB/L JPEG, quality 92. |
| Same staging, `_vqa_rad_qtype` | `closed_yes_no` only if the stripped/lowercased **raw** answer is exactly `yes` or `no`; otherwise `open`. This is answer-derived, not a separate original annotation. |
| [Task loader](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/task_loader.py) and [runner](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/benchmark_runner.py) | `resolve_question_ids` honors explicit IDs first; smoke/calibration ID files when present; otherwise sorted discovery with optional split filter. `sample_limit` takes a prefix. Therefore staging all 451 test rows does not imply every run evaluated all 451. AMA smoke selection is independently defined below. |
| [Normalizer](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/answer_normalizer.py) and [metrics](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/answer_metrics.py) | Lowercase, delete ASCII punctuation, collapse whitespace. Map whole-answer yes/no synonyms, then number words and standalone medical abbreviation tokens. Compute normalized EM and token-multiset F1. |
| [Scorer](https://github.com/AutoMedBench/AutoMedBench/blob/5a19ff72491df7f8c0279438fcfade9364f9a8b7/eval_vqa/vqa_scorer.py) | Without a judge, normalized yes/no gold uses strict normalized whole-answer accuracy; other gold uses `(EM + F1)/2`. Missing/invalid outputs score zero. Average per question. With a judge enabled, official primary accuracy is promoted to the judge score, which this migration does not implement. |

`stage_data.py` on this branch stages **MedXpertQA**, not VQA-RAD. Its dev/test
calibration seed and subset builder must not be mistaken for VQA-RAD source rules.
The adapted normalization/metric logic carries the upstream
[license notice from the pinned task branch](licenses/AutoMedBench-LICENSE.txt)
(non-commercial research only; this differs from the main branch MIT license).

## Architecture and data flow

```text
pinned HF files -> download script -> local snapshot + source manifest
                                         |
                                ama import vqa-rad
                                         |
                    dataset.json + episodes.jsonl + artifacts/
                              |                    |
                     AMA run / FakeModel           |
                              |                    |
                       decisions.jsonl             |
                              |                    |
                      AMA eval <---- targets.jsonl + eval.json
```

The importer is an offline boundary; the runner remains source-agnostic.

| File/function | Responsibility |
|---|---|
| `scripts/download_vqa_rad.py` | Download this exact revision into a new directory; verify sizes and upstream LFS SHA-256 or Git blob hashes. Uses only stdlib and respects `HTTPS_PROXY`. |
| `src/ama/importers/vqa_rad.py::_snapshot` | Require pinned repo/revision, safe source paths, both splits, listed shards, sizes and manifest SHA-256. This is local consistency validation, not independent manifest authentication. |
| `import_vqa_rad` | Read Parquet batches, retain source split/row identity, extract metadata-free PNGs, emit one Episode/Turn per QA, generate bilingual cards, provenance/report/checksums, validate and publish a staged directory. |
| `src/ama/importers/vqa_rad_scoring.py` | Apply fixed normalization and deterministic heuristic metrics; reject malformed targets/duplicate decision turns; retain failed/missing rows in denominators. |
| `scripts/verify_vqa_rad.py` | Independently compare generated questions, targets, IDs, order, split membership and decoded image pixels to source rows; verify every generated checksum; exercise oracle/empty scorer checks. No model calls. |
| `tests/test_vqa_rad.py` | Synthetic fixtures only; import/CLI tests, evaluator isolation, malformed inputs, split/sampling preservation, scoring and FakeModel run → eval. |

Public input is two Evidence items, `image` and `question`. Task output is
`{"text": "short English answer"}` inside AMA's existing Decision envelope.
Reference text and answer-derived `question_type` occur only in evaluator targets.
The run loader does not open targets, eval config, provenance or import reports;
tests replace those files with invalid content and still execute successfully.
This is an application data boundary, not an OS sandbox against arbitrary tools:
do not give an evaluated agent filesystem tools that can read evaluator files.

Images are deduplicated by original embedded-byte SHA-256 and saved as clean PNGs.
For RGB/L images pixels are unchanged; other modes are converted to RGB, matching
the official mode conversion. The verifier compares every selected image's pixels.

## AMA scoring contract and differences

Scorer: `vqa_rad_automedbench_v1`. All values are 0–1 rate objects (`num`, `den`,
`value`); empty groups have `value: null`. No intermediate rounding is applied.

1. Lowercase; **delete**, rather than replace, characters in `string.punctuation`.
2. Collapse all whitespace and strip edges.
3. Map a whole-answer yes/no synonym (`positive` → `yes`, `absent` → `no`, etc.).
4. Otherwise map number tokens (`two` → `2`, `none` → `0`, `single` → `1`), then
   medical abbreviation tokens (`IVC` → `inferior vena cava`, `l` → `left`, etc.).
   The complete maps are in the scoring module. No Unicode folding, article
   removal, stemming, substring matching or medical semantic judge is added.
5. Compute EM and multiset token F1. Primary `accuracy_heuristic` is EM for
   normalized yes/no gold; otherwise their average. Overall metrics are macro
   averages over expected questions, including missing answers as zero.

Examples: `TRUE!` matches `yes`; `yes, a lesion` does not get yes/no credit.
`lesions` against `two lesions` has F1 `2/3`, EM `0`, heuristic `1/3`.
`left-sided` normalizes to `leftsided`; `the lung` retains its article. These rules
can reward/penalize medically unsuitable wording and are not clinical correctness.

The reported `by_question_type` uses the staging definition (raw `yes`/`no`).
`yes_no_accuracy` uses the **normalized** definition, following the official
scorer. Thus a raw answer `positive` is staged `open` but belongs to the scoring
yes/no denominator. These two groupings intentionally need not coincide.

Differences from the official runtime:

- Preserve both original train and test in the full conversion; evaluate held-out
  QA performance with `--split test` explicitly. No new patient/image split.
- Use stable split + zero-based source-row IDs instead of sampled-position IDs.
- Keep answer-derived question type evaluator-only instead of public input.
- Save lossless PNGs without source metadata instead of recompressed JPEG 92.
- Enforce AMA's single-string `text` object and completed-turn status; null,
  missing, blank, non-string, extra-field and failed answers score zero. No
  coercion from numbers/lists and no official `raw_model_output` placeholder
  detector; AMA output/citation quality remains separately measured by core eval.
- Preserve the deterministic heuristic only: no LLM judge, S1–S5, Docker,
  autonomous model selection/training, official wall-time limits, or workflow score.
- AMA smoke is a pipeline acceptance set, not an asserted official smoke or
  calibration set. No leaderboard equivalence is claimed.

## Reproduce locally

From the `onboard-vqa-rad` worktree, create a dedicated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,vqa-rad]'
```

If the exact snapshot has already been downloaded with its manifest, reuse it.
Otherwise use a new destination (the downloader refuses existing destinations):

```bash
.venv/bin/python scripts/download_vqa_rad.py \
  --out data/incoming/Automedbench/VQArad/hf-bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9
```

If your local network needs a proxy, prefix **that command only** with your
`HTTPS_PROXY` setting. No proxy, login or mount configuration is changed by code.

```bash
SOURCE=data/incoming/Automedbench/VQArad/hf-bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9
FULL=data/processed/vqa-rad-hf-bcf91e7654fb-v1-full
SMOKE=data/processed/vqa-rad-hf-bcf91e7654fb-v1-smoke

.venv/bin/ama import vqa-rad --source "$SOURCE" --out "$FULL"
.venv/bin/ama import vqa-rad --source "$SOURCE" --out "$SMOKE" \
  --subset smoke --smoke-count 8 --seed 42

.venv/bin/python scripts/verify_vqa_rad.py --source "$SOURCE" --dataset "$FULL"
.venv/bin/python scripts/verify_vqa_rad.py --source "$SOURCE" --dataset "$SMOKE"
.venv/bin/python -m pytest -q -m 'not api'
```

All importer outputs must also be new directories; previous outputs are never
overwritten. Import failure removes its staging directory and does not publish a
partial AMA dataset. A failed download leaves its own partial destination for
inspection; preserve it and select a new destination to retry.

Full order is train then test; shards are sorted lexicographically; rows follow
physical Parquet order. Full mode includes every row. Smoke samples only test:
`sorted(random.Random(42).sample(range(N), min(8, N)))`, with train empty. Reports
record explicit selected source-row indices, source file hashes, rule, seed,
runtime versions and counts. Provenance retains source split/row/shard/image hash.
This permits reconstruction beyond a seed alone. Dataset name and runtime versions
affect generated cards/reports/checksums; compare on the recorded environment and
same output basename for identical whole-directory hashes.

`checksums.json` maps every generated file except itself to SHA-256. Its overall
hash is SHA-256 of the UTF-8 JSON file/hash map with sorted keys and separators
`(',', ':')`. The source snapshot hash uses the same canonical map over every
downloaded filename and its SHA-256; download timestamps do not affect it.

No model API is called by downloading, importing, verification or the synthetic
tests. To evaluate an already-existing AMA run, use `ama eval RUN_DIR`; it loads
the scorer from `eval.json`. Initial migration acceptance did not call a model API. The subsequent separately
authorized [three-model test100 experiment](vqa-rad-benchmark-results.md) reports
actual inference results. The FakeModel end-to-end test and real-data
oracle checks establish plumbing, not model accuracy.

## Failure modes and storage

- Wrong revision, missing shard, missing bytes, malformed QA, corrupt selected
  image or checksum mismatch: stop with an error; no records are silently dropped.
- Existing destination or symlink: stop before conversion; no mixing of stale
  artifacts/targets. Optional PyArrow/Pillow dependencies are imported lazily.
- Duplicate images across splits: retain source membership and report overlap;
  do not reinterpret row-level test performance as image-independent evaluation.
- Unknown decision IDs/turns, duplicates or absent/malformed targets: fail eval
  rather than silently shrink the denominator.
- Synthetic test modules skip if optional dependencies are absent; install the
  `vqa-rad` extra to run this migration's full acceptance suite.

Real images, QA, source-row IDs, targets, provenance, run records and dependency
records stay under ignored local `data/` (runs can also use ignored `runs/`).
Do not force-add these paths. The committed demo directory contains documentation
only. Only aggregate versions/counts/hashes are in the public validation summary.
No NAS mount, remote environment, SSH authorization or tmux session is modified.

See [actual validation results](vqa-rad-validation.md) and
[prepared PR text](vqa-rad-pr.md).

The fixed-cohort experiment driver and restart/reproduction commands are in
[the benchmark protocol](vqa-rad-benchmark.md).
