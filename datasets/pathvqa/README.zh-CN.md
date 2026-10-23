# PathVQA 接入 AMA

本 PR 将 PathVQA 作为单轮视觉问答任务接入 AMA。审核后的只读 AMA 数据集位于 `/mnt/nas/data/processed/PathVQA/pathvqa-task`；仓库只保存 importer、锁文件、数据卡、测试和汇总结果。

固定 source revision 为 `7568940e604f3e911441b1f6c7aedb5f1c2866bd`。本地基线固定 100 例：49 道二元题、51 道开放题，不是 AutoMedBench 官方分数。`pr100.lock.json` 将公开输入身份与评测身份分开：`verify`、`run`、`smoke` 不读取 `targets.jsonl`，`prepare`、`eval`、`compare` 才读取参考答案。

```bash
export DATASET=/mnt/nas/data/processed/PathVQA/pathvqa-task
export RUN_ROOT=/home/aignitelab/projects/xh/pathvqa-pr100
bash scripts/run_pathvqa.sh prepare
bash scripts/run_pathvqa.sh verify
bash scripts/run_pathvqa.sh run --alias qwen36-nothink
bash scripts/run_pathvqa.sh eval --alias qwen36-nothink
bash scripts/run_pathvqa.sh compare --alias qwen36-nothink
```

已收到的答案不会重新生成。Ctrl-C 后续跑必须显式使用 `--acknowledge-interrupted`。仓库不提交 demo 数据。

