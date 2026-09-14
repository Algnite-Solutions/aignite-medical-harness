# AMA — aigne-medical-agent

中文 | [English](README.md)

AMA 是一个用于评测医疗 Agent 的最小研究运行器。单轮推理研究和多轮记忆研究共用同一个小型抽象：

```text
Episode -> Turn -> Evidence -> Decision
```

Episode 是一次独立评测，可以只有一个 Turn，也可以包含多个 Turn。每轮中，运行器释放新的 Evidence，模型通过三个工具查看当前可见证据，然后提交一个结构化 Decision。运行结束后，由数据集指定的 scorer 评分。

AMA 是评测运行器，不是临床系统。它不模拟患者，不真正执行临床动作；核心循环也不理解 FHIR、肿瘤工作流或某个数据集的标签。

## 研究形式

- **单轮可信推理：** 一个 Episode 只有一个 Turn，适合研究基于证据的问答、引用、拒答和结构化决策。
- **多轮长效记忆：** 一个 Episode 包含有序的多个 Turn。新 Evidence 随时间释放，先前证据和对话历史继续保留，适合研究判断更新、纵向一致性以及长上下文或记忆能力。

多轮执行采用 teacher-forced replay。模型的决策不会改变后续观察，下一轮内容始终由数据集决定。这能保证回放可复现，但不能衡量一个动作对真实后续状态的影响。

## 一轮如何运行

模型收到一条 world observation，并且只能调用三个工具：

- `list_evidence`：列出截至当前轮已经释放的 Evidence 元数据。
- `read_evidence`：读取一条已经释放的 Evidence。
- `submit_decision`：提交本轮状态、可选动作、引用，或者拒答。

同一 Episode 内 Evidence 累计可见。模型只有先读取一条 Evidence，才能引用它；未来轮次的 Evidence 不可见。运行阶段永远不会加载评分答案。

```text
数据集 -> 释放一轮 -> 列出/读取证据 -> 提交决策 -> 保存轨迹
                                                  |
targets + hidden policy --------------------------+-> 评分
```

## 快速开始

需要 Python 3.11 或更高版本。

```bash
python3 -m pip install -e ".[dev]"
pytest -q
ama validate datasets/thyroid_demo
ama run datasets/thyroid_demo --model scripted --runs-root runs
ama eval "$(ls -dt runs/* | head -1)"
```

如果已经安装但 shell 找不到 `ama`，可以查询脚本安装目录：

```bash
python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'
```

把输出目录加入 `PATH`，或者用 `python3 -m ama.cli` 代替 `ama`。

真实的 OpenAI-compatible 模型配置写在 `ama.json` 或 `~/.config/ama/models.json`，API key 从环境变量读取。

```bash
ama run datasets/thyroid_demo --episode thyroid_001 --model glm
ama run datasets/thyroid_demo --episode thyroid_001 --model glm --interactive
```

交互和批量模式使用同一个 `run_episode()`。

本地目录或已挂载 NAS 中的数据集也可以通过名称访问：

```bash
export AMA_DATA_ROOT=/Volumes/lab-data/ama
ama validate thyroid_demo
ama run thyroid_demo --model glm
```

已经存在的显式路径优先；`import --out` 始终使用显式路径。

## 数据集目录

Runner 只理解 AMA Dataset v0：

```text
datasets/<name>/
  dataset.json       元信息、split 和 scorer 名称
  episodes.jsonl     模型可见的 Episode
  targets.jsonl      可选，仅供 evaluator 读取
  policy.json        可选，包含公开指导和隐藏评测规则
  assets/             Evidence 引用的可选文件
```

四个核心对象保持最小：

- **Episode：** 一次独立评测及其主体。
- **Turn：** 一次观察，以及本轮新释放的 Evidence。
- **Evidence：** 在当前研究中需要独立展示、读取和引用的最小信息单元。
- **Decision：** 模型在一轮中提交的结构化结果。

Evidence 可以是一份病历、影像或病理报告、一组检验、一个 FHIR Resource、一条登记记录、一张图片或一个 PDF。每条 Evidence 必须至少包含 `text` 或 `artifact`。`artifact` 是数据集目录内的安全相对路径。v0 runner 只返回这个路径，不解析或渲染文件；如果模型需要理解其内容，importer 应同时提供有用的文本表示。

`episodes.jsonl` 中的全部字段，包括 `Evidence.metadata`，都对模型可见。标准答案、派生标签和仅供评分器使用的 tag 必须放进 `targets.jsonl` 或 `policy.hidden`。

完整约定和 importer 指南见 [AMA Dataset v0 中文数据卡](docs/ama-dataset-v0.zh-CN.md)。

## 导入数据

新数据源通过离线 importer 接入。Importer 将原始记录转换成 AMA Dataset v0，并输出报告，说明来源、排除记录、缺失字段和时间处理假设。数据集特有逻辑不应进入 `run_episode()`。

```bash
ama import medagentbench --source test_data_v2.json --out datasets/medagentbench
ama import episode-folder --source path/to/episodes --out datasets/my_dataset
ama import rocov2 --source /Volumes/Shared/Work/Data/RocoV2 --out datasets/rocov2 --split test --limit 100
```

原始 MedAgentBench task 会成为单轮 Episode。可选的 FHIR 患者时间线会成为独立的多轮回放 Episode，目前没有逐轮 gold target。此类派生回放的分数不能作为原始 MedAgentBench benchmark 成绩报告。

## 评分与运行记录

`src/ama/scorer.py` 注册了三个内置 scorer：

- `exact_v0`：精确状态或答案、允许动作、必需 Evidence 和拒答。
- `workflow_v0`：状态字段、迁移规则、动作约束、必需 Evidence 和累计成功率。
- `rocov2_v0`：规范化 caption token 与无序 UMLS CUI 的 precision、recall 和 F1。
- `unscored`：只报告完成情况和运行指标。

失败或未提交的轮次仍保留在分母中；分母为零时记为 N/A。

每次运行会记录决策、事件、模型 usage、延迟、终止原因、预算、数据 hash、依赖版本和 Git revision，并支持 trace 脱敏。scripted 模型的满分只证明 runner 与 scorer 可以工作，不代表模型具有医疗能力。

## 仓库结构

```text
src/ama/data.py        schema、loader 和 validator
src/ama/model.py       scripted 与 OpenAI-compatible 模型客户端
src/ama/agent.py       唯一 Episode 循环和三个工具
src/ama/scorer.py      scorer 注册表和内置 scorer
src/ama/recorder.py    运行产物、usage、hash 和脱敏
src/ama/cli.py         validate、inspect、run、eval 和 import
src/ama/importers/     离线数据转换器
datasets/              示例和已导入的数据集
docs/                  数据集约定
```

## 已支持数据集

仓库内置样例同时覆盖单轮可信推理与多轮长效记忆研究。

| 数据集 | 来源 | Episode | Turn | Targets | Scorer | 数据卡 |
|---|---|---:|---:|---|---|---|
| `thyroid_demo` | 合成肿瘤工作流 | 1 | 3 | 有 | `workflow_v0` | [中文](datasets/thyroid_demo/DATASET_CARD.zh-CN.md) · [English](datasets/thyroid_demo/DATASET_CARD.md) |
| `medagentbench` | MedAgentBench FHIR 派生回放子集 | 3 | 12 | 无 | `unscored` | [中文](datasets/medagentbench/DATASET_CARD.zh-CN.md) · [English](datasets/medagentbench/DATASET_CARD.md) |
| `rocov2_demo` | ROCOv2 放射影像子集 | 3 | 3 | 有 | `rocov2_v0` | [中文](datasets/rocov2_demo/DATASET_CARD.zh-CN.md) · [English](datasets/rocov2_demo/DATASET_CARD.md) |
