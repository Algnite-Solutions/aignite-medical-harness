# VQA-RAD offline validation / 离线验收

Executed locally on 2026-10-19 in the `onboard-vqa-rad` worktree, based on
`main` commit `82f05f46c65fc783c50a1e2127883332eeee9b53`.
No model API was called. This record contains only versions, aggregate counts,
validation outcomes and overall hashes; private row mappings and data remain local.

本记录为实际执行结果，不是模型推断或性能估计。代码、数据转换和评测管线已离线
验证；没有真实模型性能结果，也没有调用付费 API。

## Versions

| Component | Version |
|---|---|
| HF dataset | `flaviagiammarino/vqa-rad` at `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` |
| AutoMedBench task/scoring code | `eval_vqa` at `5a19ff72491df7f8c0279438fcfade9364f9a8b7` |
| Python | 3.14.3 |
| PyArrow | 26.0.0 |
| Pillow | 13.0.0 |
| Pydantic | 2.14.1 |
| pytest | 9.1.1 |

## Results

| Check | Actual result |
|---|---|
| Source acquisition | 5 files; 34,505,256 bytes. All sizes and upstream LFS SHA-256 / Git blob hashes verified. |
| Full conversion | 2,244 single-turn QA episodes: train 1,793; test 451. |
| Staging question types | Train: closed yes/no 940, open 853. Test: closed yes/no 251, open 200. |
| Distinct original embedded-image byte hashes | Train 313, test 203; union 314, cross-split overlap 202. These are image-byte counts, not patient counts. |
| Full artifacts | 314 clean PNGs, shared by repeated image/question rows. |
| Separate smoke conversion | 8 test episodes, 8 image artifacts; closed yes/no 3, open 5. Train empty; seed 42 and sorted source indices recorded locally. |
| Real-data source comparison | Full and smoke pass: source revision/checksums, questions, stripped target text, row identity, order, split membership, decoded pixels, generated checksums and target completeness. |
| Synthetic tests | 25 passed; includes malformed-source rollback, no-overwrite, fixed sampling, answer isolation, scoring and CLI → FakeModel run → eval. |
| Repository offline suite | 137 passed (`-m 'not api'`); no failures. Final focused suite after data-card/license wording updates: 25 passed. |
| Official reference comparison | 566 distinct normalization inputs, including all distinct real source answers and rule examples; 5,094 prediction/reference metric pairs. All matched fixed official functions within `1e-12`. |
| Scorer sanity checks | Full and smoke: reference-answer oracle heuristic 1.0; absent-decision heuristic 0.0. These are evaluator plumbing checks, not model results. |
| Git whitespace check | `git diff --check` passed. |

The reference comparison executed the downloaded pinned official normalizer and
selected official EM/F1/yes-no functions with their dependencies, compared to the
AMA implementation. It did not execute the official Docker/research workflow or
LLM judge. The inspected official `score_all` formula determines the heuristic
combination; this comparison is not evidence of full leaderboard equivalence.

对应的原文代码、对照脚本、依赖列表与运行汇总保留在本地 `data/offline-checks/`。
逐例 provenance、具体抽样行号、图像和 QA 不纳入 Git。

## Overall SHA-256

Canonical source snapshot file/hash map:

```text
ab3857c8b246e5c19389baca92086ef223a312b15b0dd36fe0cc0b5fdc55c62c
```

Full output (`vqa-rad-hf-bcf91e7654fb-v1-full`):

```text
30b76770ceb830d2e9f54324083a5550d80f1662af0a0969240aef2e309093ac
```

Smoke output (`vqa-rad-hf-bcf91e7654fb-v1-smoke`):

```text
68f2d27b1fb3a00a0bda53010dfe112602d04062d7c074705100123be1e455f8
```

These output hashes cover the generated file/hash map, including bilingual cards,
runtime versions and dataset name, excluding `checksums.json` itself. Their
canonicalization and reproduction conditions are specified in [the guide](vqa-rad.md).
They are integrity records for this conversion, not clinical validation or model
performance claims.
