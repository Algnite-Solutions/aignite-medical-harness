# VQA-RAD demo policy / Demo 说明

Documentation only. This directory is **not** an AMA dataset and deliberately
contains no `dataset.json`, images, questions, reference answers or partial demo.

Use the fully synthetic fixtures in `tests/test_vqa_rad.py` for offline examples.
For a complete real-data acceptance set, generate the separate 8-row local smoke
dataset described in [the migration guide](../../docs/vqa-rad.md). It remains
under ignored `data/processed/` and must not be committed.

此目录仅为说明文档，不是可运行的 AMA 数据集，不留不完整 demo。合成示例由测试
完整生成；真实数据的 8 条验收集单独生成在本地 `data/processed/`，不提交 Git。
