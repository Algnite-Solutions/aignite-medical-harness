# Onboard PathVQA visual question answering in AMA

## Summary

Add PathVQA visual question answering to AMA. Each source row becomes one single-turn episode containing a pathology image and a question. The model returns a string in `Decision.answer`; reference answers remain evaluator-only targets and are not sent to the generation API.

- Add `ama import pathvqa` for the pinned Parquet snapshot, preserving official train/validation/test splits, stable episode IDs, image bytes, and answer mappings. Verify all 13 shards against a packaged SHA-256 source lock and refuse to overwrite an existing destination.
- Add `pathvqa_v1`: normalized yes/no accuracy for binary questions, and normalized exact match and token-frequency F1 for open questions. Missing, failed, and invalid answers score zero in the corresponding all-case means.
- Report recoverable completion, nonempty normalized answers, binary valid choices, and strict AMA output compliance separately from content scores.
- Reuse a frozen 100-question cohort from the existing local 500-question subset, with ordered IDs and hashes covering questions, images, references, instructions, and the effective system prompt.
- Provide public model configurations, bilingual data cards, and a shell entry point for import, verification, generation, and evaluation. Credentials are read from environment variables.
- Keep complete datasets and detailed run records in controlled storage. Publication to NAS `processed/` remains a manual step after review.

## Validation

- Full offline suite on DGX: **203 passed** in the implementation validation run.
- Reimported the source snapshot and checked episodes, targets, provenance, and image bytes against the staged dataset: **32,632 episodes/targets**, comprising train 19,654, validation 6,259, and test 6,719, with **4,288 unique images**.
- Checked source-lock enforcement, invalid/missing outputs, lexical scoring edge cases, target isolation, retry behavior, and interruption recovery using synthetic tests.
- Checked CLI help, shell syntax, and wheel packaging, including the packaged source checksum lock.
- Recomputed all three configurations from saved native AMA records. Each configuration has 100 terminal records for the same locked cohort. The public aggregate includes request settings, code hashes, retries, token usage, and hashes of saved run files.

## Local validation results

The frozen cohort contains **100 questions: 49 binary and 51 open**. These are **two models with three configurations**.

| Model / configuration | Recoverable completion | Valid answers | Binary accuracy ↑ | Open EM ↑ | Open token F1 ↑ | Strict AMA output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B, thinking off | 99/100 | 98/100 | 30/49 (61.22%) | 0/51 | 0.0318 | 21% |
| Qwen3.6-27B, thinking on | 88/100 | 87/100 | 31/49 (63.27%) | 0/51 | 0.0634 | 0% |
| GLM-5.3-Flash, provider default | 96/100 | 94/100 | 36/49 (73.47%) | 1/51 | 0.1009 | 95% |

Binary accuracy uses all 49 binary questions; open EM and mean token F1 use all 51 open questions. Missing or invalid answers score zero. Normalization applies Unicode NFKC, case folding, punctuation replacement, and whitespace normalization. Token F1 uses the multiset intersection of prediction and reference tokens. A valid answer here means a nonempty normalized string; binary label validity additionally requires exactly `yes` or `no`.

- **Qwen, thinking off:** one terminal `invalid_decision`; 99 recoverable Decisions, of which 98 contain a valid answer. Three transport retries yielded 103 total attempts.
- **Qwen, thinking on:** twelve terminal `invalid_decision` records; 88 recoverable Decisions, of which 87 contain a valid answer. One transport retry yielded 101 total attempts. Zero strict compliance does not mean zero recoverable answers: content scoring uses AMA's recovered Decisions, while strict compliance checks the original output contract.
- **GLM:** four terminal HTTP 400 API errors; 96 recoverable Decisions, of which 94 contain a valid answer. There were 100 attempts and no retries. Strict format compliance (95%) and answer validity (94%) measure different properties.

All configurations used `temperature=0`, `max_tokens=8192`, a 300-second timeout, `max_calls=1`, no tools, and a one-second interval within each configuration. Separate image smoke requests are excluded. Only transport errors or HTTP 408/429/5xx qualify for up to three attempts. Received incorrect, empty, or malformed answers are frozen rather than regenerated.

GLM has the highest binary accuracy and open token F1 in this cohort. Enabling Qwen thinking produced one additional correct binary answer and higher open token F1, alongside lower completion. This small sample does not establish a general model ranking or a causal relationship between formatting and answer quality. Equal output caps do not imply equal compute. Saved code hashes also differ across the runs; the aggregate records these versions rather than claiming one identical implementation snapshot.

## Reproduction and storage

See `datasets/pathvqa/README.md` and `README.zh-CN.md` for the data cards, `models.v2.json` for public request configurations, and `baseline.v2.json` for the machine-readable results. The source is pinned to `flaviagiammarino/path-vqa@7568940e604f3e911441b1f6c7aedb5f1c2866bd`. Reproduction is defined by `source.lock.json` and `pr100.v2.lock.json`, not by a seed alone.

```bash
python3 -m pip install -e '.[pathvqa]'
export DATASET=/mnt/nas/data/incoming/Automedbench/PathVQA/ama/pathvqa-task
# Existing saved runs on DGX; use a new RUN_ROOT for a fresh experiment.
export RUN_ROOT=/home/aignitelab/projects/xh/ama-pathvqa/local_data/pathvqa-v2-sandbox/runs/pathvqa-pr100-v2
bash scripts/run_pathvqa.sh verify
bash scripts/run_pathvqa.sh all --out /tmp/pathvqa-baseline-v2.json
python3 -m scripts.render_pathvqa_results /tmp/pathvqa-baseline-v2.json
```

The commands above validate the frozen dataset and rescore saved records without making model requests. Fresh-generation commands, including separate smoke checks, are in the English data card. The staged source and dataset are under `/mnt/nas/data/incoming/Automedbench/PathVQA/{raw,ama/pathvqa-task}`. The aggregate is also stored at `manifests/pathvqa-results-v2.json` under that root. Detailed run records currently reside in the DGX run directory shown above.

## Scope

This adapts the underlying PathVQA task to AMA. The 500-question subset and its 100-question baseline are local selections, not verified AutoMedBench official cohorts. This does not reproduce the AutoMedBench coding-agent workflow or official scoring. Lexical EM/F1 may miss semantically equivalent answers and do not measure clinical correctness.

Real images, question/answer text, generated datasets, credentials, and per-case outputs stay outside Git. Public manifests include frozen sample IDs and hashes for reproducibility. Earlier v1 experiments remain explicitly historical because their timeout and output-cap settings differ. NAS `processed/` publication and PR merge are separate review steps.
