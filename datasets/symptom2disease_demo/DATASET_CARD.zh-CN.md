# symptom2disease_demo——数据卡

## 来源与许可

来源：[NeuronZero/Symptom2Disease](https://huggingface.co/datasets/NeuronZero/Symptom2Disease)，是 [Kaggle 原数据集](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) 的公开副本。源 CSV 的 SHA-256 与选取行号见 `import_report.json`。

许可：Hugging Face 副本标注为 Apache-2.0；尚未独立核实 Kaggle 原数据集的许可条款。

## 研究用途

将症状文本映射到源数据疾病标签的单轮研究分类任务。

## 构造方法

从六个不同疾病标签中选取 CSV 第 2、52、102、152、202、252 行。每行生成一个 Episode 和一个 Turn；症状是可见 Evidence，标签仅供评测。

## 规模

- Episode：6
- Turn：6
- Evidence：6

## 评分与 targets

Scorer：`unscored`。Target 记录数：6。

## Artifacts

没有外部 artifact；症状文本直接存于 Evidence，原始行号存于 `import_report.json`。

## 限制与报告口径

这六条样本不是官方测试集或 benchmark；目前也没有任务准确率评分器。数据集标签预测不代表临床诊断准确率。

## 如何测试

在仓库根目录按 README 安装项目。下列命令不需要模型密钥，可检查 demo 并运行 importer 测试：

```bash
python3 -c "from pathlib import Path; from ama.data import validate_dataset; assert not validate_dataset(Path('datasets/symptom2disease_demo'))"
python3 -m pytest -q tests/test_symptom2disease.py
```

如已取得完整源 CSV，可选择一个尚不存在的输出目录，重建相同行号的样本：

```bash
ama import symptom2disease --source /path/to/Symptom2Disease.csv --out /tmp/symptom2disease-check --row 2 --row 52 --row 102 --row 152 --row 202 --row 252
```

如需用真实模型试跑，按 `.env.example` 配置模型别名及密钥。先运行一例，再把 `ama run` 打印的运行目录用于评测：

```bash
ama run datasets/symptom2disease_demo --model qwen36 --episode s2d-000001
ama eval runs/<运行目录>
```

`all` split 包含全部六例。当前 scorer 为 `unscored`，`eval` 只报告完成情况。
