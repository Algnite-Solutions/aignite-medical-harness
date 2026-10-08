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

Scorer：`symptom2disease_v1`。Target 记录数：6。任务列出源 CSV 中的 24 个标签，要求在 `answer.disease` 中选一个。评分器接受 `{"disease": "<标签>"}` 或单个标签字符串；比较时忽略大小写和多余空格，但必须匹配源标签。不在选项中的文本、别名、多个标签、缺失回答和失败的 Turn 均计为错误。Top-1 准确率以全部六个预期 Episode 为分母；完成率、有效选项率和答案对象格式另行报告。

## Artifacts

没有外部 artifact；症状文本直接存于 Evidence，原始行号存于 `import_report.json`。

## 限制与报告口径

这六条样本不是官方测试集或 benchmark。原数据集没有规定此处的 AMA 24 选 1 提示词和评分协议。Demo 准确率只用于验证评测流程，不是公开发表的 Symptom2Disease 基准成绩，也不代表临床诊断准确率。

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

如需用真实模型试跑，按 `.env.example` 配置模型别名及密钥。运行全部六例，再把 `ama run` 打印的运行目录用于评测：

```bash
ama run datasets/symptom2disease_demo --model qwen36
ama eval runs/<运行目录>
```

`all` split 包含全部六例。`eval` 报告疾病标签的 top-1 准确率和独立的输出质量指标。

## 固定 100 例、多模型评测

这是 AMA 针对原始 24 类 `label,text` CSV 定义的研究协议，并非 Symptom2Disease 官方测试集或临床诊断基准。每个 Episode 只给出一段症状描述，要求从源数据的 24 个标签中选一个。`symptom2disease_v1` 忽略大小写和多余空格后精确匹配标签，计算 top-1 准确率。缺失、格式错误和不在选项内的答案均计为错误；答案格式与完成率另行报告。

### 一条命令完成准备、运行、评分与比较

在仓库根目录运行，源 CSV 保存在本地，所选模型的密钥已设置在各自配置的环境变量中：

```bash
.venv/bin/python -m scripts.symptom2disease_benchmark \
  --source /path/to/Symptom2Disease.csv \
  --out runs/symptom2disease-fixed100-seed42 \
  --model deepseek-v4-flash \
  --model antangelmed2 \
  --model qwen36
```

模型别名来自 `ama.json`。也可以用 `--config /path/to/models.json` 选择其他 AMA 兼容配置，无须修改源码。命令不会保存 API 密钥值。`--prepare-only` 只生成并检查样本，不调用模型。运行环境需要能读取配置中 `api_key_env` 对应的环境变量。

### 固定样本与运行约束

抽样前按空格归一化、忽略大小写后的症状文本去重，保留 CSV 中首条记录；相同文本若对应不同标签则报错。使用种子 42，从 24 类中各取四条，再从随机选出的四类中各加一条。最终 100 个 CSV 行号和 Episode ID 按行号排序，连同源文件 SHA-256、各类数量、评分器版本及数据集文件哈希写入 `benchmark.json`。再次运行时若源文件或数据集发生变化，命令会报错，不会覆盖原样本。原始 CSV 保留在本地，不提交到仓库。

每例都通过 AMA 的标准 `execute()` 路径独立运行。默认每例最多调用模型一次；请求超时 300 秒，请求间隔 1 秒，传输问题每例最多尝试三次。温度取模型别名配置。比较时要求实际系统提示词、Episode 顺序、超时、调用次数上限、温度和输出 token 上限一致；各模型特有的服务参数仍记录在配置中。

批处理只对连接失败、HTTP 408/429/5xx，以及没有可用 assistant 消息的服务端响应做有界退避重试。重试耗尽时停止，保留已完成的逐例记录；用相同命令可以续跑。assistant 消息一经落盘，该例回答即冻结。续跑前会核对已有的 AMA shard，包括刚完成单例运行、但尚未写入 `batch.json` 就中断的记录。无法解析为 Decision 时通常记录为 `invalid_decision`；消息内容本身不符合 AMA 要求时可能记录为 `api_error`。这两种情况都计为错误。永久性 API 错误会终止批次，修正配置后需要使用新运行目录。只要有未解决的传输或服务端错误，就不会生成最终比较文件。

### 结果与解释

`runs/<别名>/batch.json` 保存尝试记录和运行设置，`shards/` 保存逐例 AMA 记录，`merged/` 保存合并后的标准运行和 `metrics.json`。只有全部模型都取得 100 条终态记录后，才写入 `comparison.json`。比较过程校验样本、实际提示词、模型参数、Decision 覆盖、评分器和 target 文件哈希，再报告准确率、完成率、输出格式、终止原因、token 用量及各模型之间的逐例配对结果。

这 100 条只是固定抽样。若根据这批样本调整提示词或参数，再在同一批样本上报告分数，只能算探索性结果。分数及其差异均不代表临床表现或确定的模型排序。
