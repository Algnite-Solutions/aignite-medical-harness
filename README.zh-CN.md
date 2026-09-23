# AMA — aignite-medical-agent

中文 | [English](README.md)

AMA 是医疗模型的研究评测运行器，不是临床系统。它用同一个 `Episode → Turn → Evidence → Decision` 数据抽象支持单轮问答和多轮信息更新。多轮是固定信息释放的 teacher-forced replay；模型决策不改变后续观察。

## 两种交互协议

默认 `direct_decision`：每轮直接展示新证据，模型输出 `{"turn_id":"t1","answer":...,"citations":[]}`。`answer` 是任务定义的 JSON，`null` 表示弃答。

可选 `tool_agent`：模型先用 `list_evidence` 和 `read_evidence` 查看当前及此前已释放的证据，最后输出同样的 Decision。只有读过的证据才能引用。工具调用只记录在轨迹中，不是 Decision 字段。两种协议使用同一个回放控制器，但提示、解析和引用约束互不混杂。未来轮证据始终不可见；推理阶段不读取标准答案。

## 快速开始

需要 Python 3.11+。

```bash
python3 -m pip install -e ".[dev]"
pytest -q
ama validate datasets/thyroid_demo
ama run datasets/thyroid_demo --model scripted --protocol tool_agent \
  --instruction-file datasets/thyroid_demo/instructions.txt --runs-root runs
ama eval "$(ls -dt runs/* | head -1)"
```

模型别名在 `ama.json` 或 `~/.config/ama/models.json` 中配置；密钥由环境变量读取。`--interactive` 与批量运行共用回放控制器。显式数据集路径优先；也可设 `AMA_DATA_ROOT` 按名称查找已挂载数据集。

## 数据集

运行器只接受 `ama-dataset`。最小目录包含 `dataset.json`（仅 `schema`、`name`、`splits`）和模型可见的 `episodes.jsonl`。可选的 `targets.jsonl`、`eval.json` 只用于评分；`provenance.jsonl` 保存本地源记录定位，运行和评分都不读取；`instructions.txt` 中的任务说明需通过 `--instruction-file` 显式传入。

Episode 含 `id` 与有序 `turns`；Turn 含 `id`、可选 `observation` 和真实 `available_at`、新释放的 `evidence`；Evidence 含 `id`、可选 `type`、`text`、相对 `file`，其中 `text` 或 `file` 至少存在一个。没有真实时间时不补造日期。诊断、鉴别、治疗计划、下一步行动等均在任务自己的 `answer` schema 中，不进入通用 Decision 字段。

完整格式、示例、文件安全与边界见 [AMA Dataset 中文规范](docs/ama-dataset.zh-CN.md)。

## 导入与评分

```bash
ama import medagentbench --source test_data_v2.json --out datasets/medagentbench
ama import episode-folder --source path/to/episodes --out datasets/my_dataset
ama import rocov2 --source /path/to/ROCOv2 --out datasets/rocov2 --split test
```

三个内置导入器均输出统一数据格式，另写导入报告和数据卡。内置 scorer 为 `exact`、`workflow`、`rocov2` 及 `unscored`。失败或缺失决策仍计入适用指标的分母。

每次运行记录协议、完整任务说明、模型配置、模型可见输入的哈希、事件、决策、使用量和终止原因。标准答案、评分规则及 provenance 不进入模型调用或运行清单。
