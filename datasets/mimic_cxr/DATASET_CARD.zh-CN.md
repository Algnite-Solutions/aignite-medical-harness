# MIMIC-CXR：数据与任务接入 AMA

本目录只包含说明文档及公开的快照校验摘要，**不是可运行 demo**。
不包含 dataset.json、episodes.jsonl、真实图像、报告或 instructions.txt。
importer 会在受限存储中生成完整 AMA 数据集，以及与所选模式一致的 instructions.txt。

## 任务与来源

复用 MIMIC-CXR importer，将 AutoMedBench 的 MIMIC-CXR findings 生成任务接入 AMA 原生运行流程。
一个 study 是一个 episode、一轮输入；同一 study 的全部 JPEG 胸片及视角作为 evidence。
模型返回 AMA Decision，任务答案为 `answer={"findings":"..."}`。参考 findings 只写入 `targets.jsonl`，推理 loader 不读取。

任务依据：[AutoMedBench 论文](https://arxiv.org/abs/2606.01961) 及其 [官方配置](https://github.com/AutoMedBench/AutoMedBench/blob/a4e64e5502c728cee744edce2b596000fcd60fdd/eval_report_gen/mimic-cxr-report-task/config.yaml)。
本接入保留 findings-only、按 study 分组和多视角输入；使用 AMA 的 JSON Decision 和 API 调用方式。
本接入不复现论文的 agent 研究流程、Docker 环境或官方测试病例；使用上游提供的轻量评分，无需 Docker。

## 两种导入来源

### 独立 raw package

先安装图像依赖：`python3 -m pip install -e ".[mimic-cxr]"`。
以下准备命令及辅助脚本默认使用 `.venv/bin/python`；使用其他环境时，准备命令改用对应解释器，辅助脚本设置 `PYTHON_BIN`。

先从已取得访问权限的 MIMIC-CXR-JPG 源目录准备数据：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_mimic_cxr_raw.py \
  --source /path/to/MIMIC-CXR-JPG --out local_data/mimic_cxr/raw_example10 \
  --split test --limit 10 --report-mode findings-only

ama import mimic-cxr --source local_data/mimic_cxr/raw_example10 \
  --out local_data/processed/MIMIC-CXR-example10/dataset \
  --split test --report-mode findings-only
```

raw importer 仅提取明确的 FINDINGS 段落，缺失时中止导入；不会将完整报告当成 findings。
默认 `--report-mode findings-impression` 保留旧 importer 行为。准备与导入时均须指定 findings-only。
自动选取按照已有的单图/多图覆盖及 study 顺序规则，不等于随机采样；具体规则和排除原因保留在本地选择报告。
此示例产生新子集；单独指定 --limit 100 或 --study-id 均不能重建历史 seed-42 实验及其冻结参考字段。

## 复现固定 100 例实验

使用 NAS 中受限的 `processed/mimic-cxr-jpg-2.1.0/findings100-v1/` 快照，
或其未经修改的本地副本，不重新 raw 导入。访问权限需单独取得，仓库不分发真实病例。
快照含完整 `dataset/` 及同级 `transfer-manifest.json`，固定了病例顺序、
164 张图像、任务提示和评测专用参考 findings。
公开的 [快照校验摘要](findings100-v1.lock.json) 固定该 manifest 的 SHA256；
受限 manifest 再固定全部 174 个数据文件的大小和 SHA256，仓库不公开病例 ID。

安装依赖、取得快照后，在仓库根目录执行：

```bash
# 替换为可正常读取的 NAS 挂载或本地副本的绝对路径。
SNAPSHOT=/path/to/data/processed/mimic-cxr-jpg-2.1.0/findings100-v1
PYTHONPATH=src .venv/bin/python scripts/verify_mimic_cxr_snapshot.py \
  --dataset "$SNAPSHOT/dataset" --manifest "$SNAPSHOT/transfer-manifest.json"
export DATASET_DIR="$SNAPSHOT/dataset"
# 配好自己的 key，并确认向该服务商发送图像的授权后：
bash scripts/run_mimic_cxr.sh run glm-5.3-flash-coding --split test
# 使用上一步实际打印的运行目录：
bash scripts/run_mimic_cxr.sh eval /absolute/path/to/run
```

校验失败即停止，不重新生成参考答案，也不另选 100 例替代。
测试其他模型时，将别名替换为 `qwen3.8-flash` 或 `deepseek-v4.1-flash-1017`，
数据和快照内的 instructions.txt 保持不变。

历史参数：temperature=0、timeout=180 秒、max_calls=1、无工具；
Qwen 和 DeepSeek 请求 max_tokens=8192，GLM 未设置。
历史本地运行采用两个并发 worker，复用 Symptom2Disease 的重试包装器，
对可重试的传输/服务端错误最多发起三次 HTTP 请求，已收到的答案不重试。
上述原生运行命令为顺序执行，不含该包装器；复现的是冻结输入、任务和评分，
不是历史并发/重试驱动。连接失败及显式重试应如实记录，不能静默替换模型输出。
API model ID 和服务商默认值可能变化，重新运行不保证分数逐位一致。
原始轻量分数为 GLM 0.3352（93/100 合格）、Qwen 0.3008（100/100）、
DeepSeek 0.2326（99/100），不是 AutoMedBench 官方榜单分数。

历史 seed-42 子集来自 ReXrank 固定 revision `0a18a41e580992ac7cd32a8a9463009654f0d390`
中的 2,347 个 study / 3,858 张图；仅提供 seed 并不能定义完整样本集合。
此处以冻结快照而非重新抽样作为所报结果的复现标准。官方 AutoMedBench pilot
和后来整理的完整 test 合格集是独立 cohort，不能替换原实验集合。
后续严格 raw 导入排查发现，这 100 例中 11 例不满足当前显式 FINDINGS 解析规则，
但冻结参考文本仍可在原始报告中找到。重新解析会改变病例集或参考字段；
复现该历史实验应保留快照内已有的结构化 targets。

## 可选：转换其他冻结的 AMA CXR 病例

```bash
bash scripts/run_mimic_cxr.sh reuse \
  /absolute/path/to/frozen-cxr/dataset
```

此入口用于数据转换；直接使用上述 NAS 快照无需执行 reuse。
等价于 `ama import mimic-cxr --source-format ama --report-mode findings-only --split test ...`。
此入口保留已有 split 内的病例顺序、episode/evidence ID、全部图片和视角；逐张解码 JPEG 并校验原 provenance 的 SHA256。
参考字段直接复制已有结构化 `targets.answer.findings`，不重新切分旧的完整报告，也不继承旧 scorer、提示词或模型输出。
输出目录必须是新目录。可用 `--study-id` 筛选或 `--limit` 取顺序前缀，不重新随机抽样。
此前 seed-42 的 100 例可作为本地验证子集；它是沿用旧选样规则的子集，不是 AutoMedBench 官方病例集。

## 原生 AMA 运行与验收

```bash
bash scripts/run_mimic_cxr.sh run glm-5.3-flash-coding
bash scripts/run_mimic_cxr.sh eval local_data/automedbench/runs/mimic-cxr-ama/<run-id>
```

脚本只包装 `ama import/run/eval`。通过 `DATASET_DIR`、`CONFIG`、`RUNS_DIR`、`TIMEOUT`、`MAX_CALLS` 修改运行参数。
可在 `run MODEL` 后追加 `--episode` 等 AMA 参数。API key 由相应环境变量或本地 `.env` 提供，不能提交到仓库。
运行在发起命令的机器上，模型推理通过配置中的外部 API 完成；参考报告不发送到模型服务。

仓库中的 `ama.json` 已提供以下别名；按照 `.env.example` 中的变量名，将自己的 key 填入本地 `.env`。

| AMA 别名 | 接口请求的 model ID | 密钥环境变量 |
| --- | --- | --- |
| `glm-5.3-flash-coding` | `glm-5.3-flash`（GLM Coding 接口） | `GLM_API_KEY` |
| `qwen3.8-flash` | `qwen3.8-flash`（DashScope 接口） | `DASHSCOPE_API_KEY` |
| `deepseek-v4.1-flash-1017` | `deepseek-flash`（DeepSeek 接口） | `DEEPSEEK_API_KEY` |

替换运行命令中的别名即可切换模型。运行会向所选服务商发送胸片，应确认数据访问条款及向该目的地发送的授权。
原有 `glm-5.3-flash` 别名仍使用标准 GLM 接口，新增 Coding 别名对应本地 CXR 实验使用的接口。
DeepSeek 别名记录本次实验的版本标签，但实际请求的服务商 model ID 会随服务更新。

配置保留已有实验的请求参数：Qwen、DeepSeek 为 `max_tokens=8192`，GLM 未设置；三者均请求 `temperature=0`，思考行为使用服务商默认值。
这不保证服务商实际生效的生成设置或 token 预算一致。新的严格对照实验应通过独立 `CONFIG` 明确设置，并随结果报告参数。

验收包括：导入结构有效、输入/标注对应正确、参考答案隔离、AMA 能保存模型预测。
findings-only 导入默认注册 `mimic_cxr_automedbench_v1`，`ama eval` 同时计算轻量内容指标与独立的 AMA 格式诊断。旧 findings-impression 模式仍为 `unscored`。
非空 findings、格式合规与临床正确性应分开报告，失败病例也保留在原分母中。

## 输出与限制

- `dataset.json`、`episodes.jsonl`、`instructions.txt`：推理可读数据。
- `targets.jsonl`：只包含本任务要求的参考答案。
- `artifacts/`：校验后复制的 JPEG。
- `provenance.jsonl`、`import_report.json`：本地来源与校验记录；复用时还保存源文件 hash 和可用的 `source_cohort.json`。
- AMA 运行目录：manifest、消息、decisions、diagnostics；eval 后生成 metrics。

真实图片、报告、病例标识、来源记录和运行产物留在忽略的 `local_data/`。遵守原始数据访问协议，不随代码分发。
已有本地历史实验保留，不与新任务结果混用。当前验证不能证明临床质量或与论文排行榜可比。

## 报告内容评分

轻量逐例评分与格式检查来自 AutoMedBench 固定 revision `a4e64e5502c728cee744edce2b596000fcd60fdd`。
已删除未使用的后端导入和批量评分辅助函数，逐例评分函数保持不变。上游及当前文件 SHA256、修改说明和研究使用许可见 `src/ama/_vendor/automedbench_report/`。
AMA 适配数据读写和汇总；评分不调用外部 judge 或报告生成 API，也不需要临床模型权重。

默认使用上游提供的 **lightweight backend**：

- `mean_observation_f1`：官方 12 类规则标签 F1，标签从参考/生成文本中用正则提取，不是 CheXbert，也不是 RadGraph。
- `mean_rouge_l_f1`：官方 tokenizer、最长公共子序列和 F1 实现；每例按上游规则保留四位小数。
- `case_mean_score`：逐例 `0.7 × observation_f1 + 0.3 × ROUGE-L F1` 后平均；这是 AMA 的逐例诊断汇总。
- `valid_case_mean_score`：只对合格报告计算的辅助均值，必须同时报告覆盖率。
- `complete_submission_score`：沿用官方整批提交规则，只有所有报告都合格时才保留分数，否则为零。

报告合格条件原样使用官方检查：40–8000 个字符、至少 20 个字母、UTF-8 可读且字符属于官方 printable 集合。
这些条件与 AMA 四字段 JSON 严格格式不同：可恢复为 Decision 的答案仍可参与内容评分；无法解析的 Decision 不从原始文本重新猜测答案。
缺失、解析失败或未通过报告检查的病例在 AMA 逐例指标中均计零，留在分母内。此规则是显式适配，避免缺失/正常报告的空标签 F1 获得分数。
分数越高代表此指标下更相似，不等同经验证的临床正确性。不得将轻量分数称为论文默认七指标分数。

已有的 `unscored` 运行可以不重新生成，直接执行：

```bash
bash scripts/run_mimic_cxr.sh eval /path/to/existing/run \
  --scorer mimic_cxr_automedbench_v1
```

## 目录约定

与其他 importer 一致，可运行数据默认位于 `$AMA_DATA_PATH/processed/MIMIC-CXR-AutoMedBench-findings100-v1/dataset`（`AMA_DATA_PATH` 默认 `local_data`），运行记录与数据分开。可通过 `DATASET_DIR` 指定现有数据目录。之前的实验目录无需移动，历史 run manifest 的绝对路径仍可使用。
