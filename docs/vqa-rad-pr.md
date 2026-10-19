# Prepared PR

No commit, push or pull request was created automatically.

Title:

```text
feat: migrate VQA-RAD data and task from AutoMedBench into AMA
```

Description:

```markdown
VQA-RAD can now be imported from a pinned Hugging Face snapshot into AMA's
single-turn image/question episodes, preserving source train/test membership.
Reference answers and answer-derived question types remain evaluator-only.

The independent `ama import vqa-rad` command generates complete AMA datasets,
bilingual cards, local provenance, reproducible full/smoke selections and file
checksums. A deterministic scorer adapts the pinned AutoMedBench normalization
and heuristic: strict normalized yes/no match; otherwise mean EM/token F1.
Missing, invalid and failed answers remain in the denominator.

The downloader verifies upstream checksums; the offline verifier compares source
rows, answers, split/order and decoded pixels. Real data and run/provenance records
stay under ignored `data/`. The committed demo is documentation only.

Validation:
- 137 offline tests passed, including 25 VQA-RAD tests and a synthetic FakeModel
  run → eval test; API tests excluded.
- Real snapshot: 2,244 QA episodes (train 1,793; test 451), plus a separately
  generated fixed 8-row test smoke set. Both passed offline integrity checks.
- 566 normalized-answer inputs and 5,094 metric pairs matched fixed official
  reference functions. Oracle/empty scorer sanity checks were 1/0.
- No paid model API calls or real-model performance experiment.

Scope/limits:
- No S1–S5, Docker, LLM judge or autonomous research workflow; no changes to
  PathVQA or MIMIC-CXR.
- Differences (stable source-row IDs, retained train split, evaluator-only type,
  lossless PNGs, AMA answer contract and separately defined smoke selection) are
  documented with pinned official source links.
- Source train/test share 202 original image-byte hashes; this is a QA-row split,
  not an image- or patient-independent benchmark.
- Preserve the task branch's non-commercial research license notice separately
  from the HF dataset card's CC0-1.0 declaration.

See docs/vqa-rad.md, docs/vqa-rad.zh-CN.md and docs/vqa-rad-validation.md.
```

## Manual commit and push

Review the changes and ignored paths before executing these commands yourself.
The data/processed outputs, source downloads and private validation records must
not be force-added.

```bash
cd /Users/bob/Projects/Research/aignite-medical-harness-vqa-rad
git branch --show-current
git diff --check
git status --short

git add .gitignore README.md README.zh-CN.md pyproject.toml \
  src/ama/importers/__init__.py src/ama/importers/vqa_rad.py \
  src/ama/importers/vqa_rad_scoring.py src/ama/scorer.py \
  scripts/download_vqa_rad.py scripts/verify_vqa_rad.py tests/test_vqa_rad.py \
  datasets/vqa_rad_demo/README.md \
  docs/vqa-rad.md docs/vqa-rad.zh-CN.md docs/vqa-rad-validation.md \
  docs/vqa-rad-pr.md docs/licenses/AutoMedBench-LICENSE.txt

git diff --cached --stat
git -c user.name=Bob -c user.email=1004553126@qq.com commit \
  -m "feat: migrate VQA-RAD data and task from AutoMedBench into AMA"
git push -u origin onboard-vqa-rad
```

Open the PR manually against `main` with the title and description above. Do not
target the separate MIMIC-CXR branch.
