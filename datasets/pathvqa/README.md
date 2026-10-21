# PathVQA in AMA

This importer adapts the pinned [`flaviagiammarino/path-vqa`](https://huggingface.co/datasets/flaviagiammarino/path-vqa) Parquet snapshot at revision `7568940e604f3e911441b1f6c7aedb5f1c2866bd`. The source manifest and SHA-256 inventory are in [`source.lock.json`](source.lock.json). The Hugging Face entry is marked MIT; users should check the rights of the underlying pathology images before redistribution. No source image, question, answer, converted episode, or run record is committed here.

Each row becomes one single-turn AMA Episode with its image as Evidence. The answer is stored only in `targets.jsonl`, outside model observations. Image bytes are deduplicated by SHA-256. Source `train`, `validation`, and `test` splits are preserved. `automedbench_test_500` is a deterministic 500-case subset of the source test split, selected with Python `random.Random(42)` and ordered by source index. The PR baseline uses the 100 IDs in [`pr100.lock.json`](pr100.lock.json), stratified to 49 binary and 51 open questions. This is an AMA sample baseline, **not an official AutoMedBench score**.

On the DGX, the existing source snapshot is `/mnt/nas/data/incoming/Automedbench/PathVQA/raw`, and the staging AMA dataset is `/mnt/nas/data/incoming/Automedbench/PathVQA/ama/pathvqa-task`. Review and manual publication to `/mnt/nas/data/processed` happen after PR approval.

```bash
python -m pip install -e '.[pathvqa]'
ama import pathvqa \
  --source /mnt/nas/data/incoming/Automedbench/PathVQA/raw \
  --out /path/to/new/pathvqa-task
python -c 'from ama.data import validate_dataset; from pathlib import Path; errors = validate_dataset(Path("/path/to/new/pathvqa-task")); print(errors); raise SystemExit(bool(errors))'
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json --prepare-only
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json --model qwen36
ama eval runs/pathvqa-pr100/<run-id>
python scripts/pathvqa_baseline.py /path/to/new/pathvqa-task \
  --cohort /path/to/pr100.lock.json \
  --summarize-run runs/pathvqa-pr100/<run-id> --summary-out /path/to/results.json
```

The importer refuses to overwrite an existing destination and verifies source checksums by default. `--limit-per-split 1 --benchmark-count 0` supports a smoke import. The baseline script checks dataset/target hashes and source repository/revision against the frozen cohort before making requests. Use the basename `pathvqa-task` to reproduce the dataset metadata hash; local output paths in the import report do not define cohort identity.

`pathvqa_v1` reports completion and valid-answer rates separately from answer quality. Binary questions use strict normalized `yes`/`no` accuracy. Open questions use normalized exact match and token F1. Missing, invalid, failed, or nonconforming outputs remain in the appropriate denominator. These lexical metrics can miss clinically equivalent answers and cannot establish medical correctness.

## 中文说明

此导入器使用固定 revision 的 PathVQA Parquet 快照；源文件及 SHA-256 见 `source.lock.json`。每条原始记录对应一个单轮 Episode，图片作为 Evidence，参考答案只保存在模型不可见的 `targets.jsonl`。保留官方 train/validation/test 划分，另从 test 固定抽取 500 例；PR 基线从该集合中按二元/开放题比例固定选取 100 例。数据、图片、参考答案和逐题运行记录只保存在受控存储，不进入 Git。

`pathvqa_v1` 分别报告完成率、有效答案率和内容分数：二元题计算严格 yes/no 准确率，开放题计算规范化 exact match 与 token F1。失败和缺失计入分母。这是 AMA 小样本词面评测，不是 AutoMedBench 官方结果，也不能单独证明临床正确性。PR 审核后再由人工核对并发布到 `/mnt/nas/data/processed`。
