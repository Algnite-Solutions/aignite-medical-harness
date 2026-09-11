# AMA Dataset v0——数据卡与格式约定

中文 | [English](ama-dataset-v0.md)

## 目的

AMA Dataset v0 是 AMA runner 唯一理解的输入格式。它用同一个最小核心支持单轮可信推理和多轮纵向记忆实验，不把领域概念加入核心循环。

```text
Episode -> Turn -> Evidence -> Decision
```

本文既是格式规范，也是数据导入者需要填写的最小数据卡。

## 目录约定

```text
datasets/<name>/
  dataset.json       必需；元信息、split 和 scorer
  episodes.jsonl     必需；每行一个模型可见的 Episode
  targets.jsonl      可选；仅供 evaluator 读取
  policy.json        可选；公开和隐藏 policy
  assets/             可选；Evidence 引用的文件
  import_report.json 可选；源数据转换报告
```

运行阶段加载 `dataset.json`、`episodes.jsonl` 和 `policy.public`，不会打开或计算 `targets.jsonl` 的 hash。评测阶段才加载 targets 和 `policy.hidden`。

## 核心对象

### Episode

Episode 是一次独立的评测单元。

```json
{
  "episode_id": "case-001",
  "subject_id": "subject-001",
  "metadata": {},
  "turns": []
}
```

单轮 Episode 合法。多轮 Episode 的 Turn 时间不能为空，并且必须单调不减。模型上下文和已释放 Evidence 在同一 Episode 内保留，在不同 Episode 之间重置。

`subject_id` 可以使用假名。Importer 必须遵守源数据的隐私与再分发要求。

### Turn

Turn 是一次 world observation，以及在这个时间点新开放的 Evidence。

```json
{
  "turn_id": "t1",
  "time": "2026-01-01T09:00:00+00:00",
  "message": "新的病理报告已到达。",
  "evidence": []
}
```

`time` 是安排回放顺序的释放时间，只能在单轮 Episode 中为 null。如果源数据区分事件时间、记录时间和可见时间，应将它们保存在 Evidence metadata 中，并按照 Evidence 对 Agent 真正可见的时间把它放入对应 Turn。

### Evidence

Evidence 是实验中需要独立展示、读取和引用的最小信息单元。它是评测对象，不一定对应数据库的一行，也不一定是一条原子临床事实。

```json
{
  "evidence_id": "path-001",
  "kind": "pathology_report",
  "text": "甲状腺左叶结节：甲状腺乳头状癌。",
  "artifact": "assets/path-001.pdf",
  "source": "pathology/reports/001",
  "metadata": {
    "event_time": "2026-01-01T08:30:00+00:00",
    "specimen": "left_thyroid"
  }
}
```

| 字段 | 要求 |
|---|---|
| `evidence_id` | 在 Episode 内稳定且唯一。 |
| `kind` | 简洁的来源或内容类型，例如 `lab_panel`、`clinical_note`、`radiology_report`、`pathology_report`、`fhir_condition` 或 `image`。 |
| `text` | 模型可读的规范化内容；只有存在 `artifact` 时才可以为空。 |
| `artifact` | 可选的数据集目录内相对路径；禁止绝对路径和 `..` 路径逃逸。 |
| `source` | 在数据访问规则允许的范围内，足以追溯原始记录的来源定位。 |
| `metadata` | 可选、数据源特有并且对模型可见的字段。 |

应选择自然的信息粒度。一份完整报告或同一采样时间的一组检验通常比逐句、逐值拆分更合适。一条 Evidence 应能独立呈现而不改变原意，并适合作为一次引用。

模型调用 `read_evidence` 后可以看到 Evidence 的全部字段。不要在 `metadata` 中放入标准答案、派生临床结论、评分标签或未来信息；这些内容应放在 `targets.jsonl` 或 `policy.hidden`。

`artifact` 用于标识图片、PDF、波形或文本文件等原始或派生材料。AMA v0 只验证路径位于数据集内并且文件存在，不负责解析或渲染。如果被测模型需要文本输入，importer 应同时提供忠实的 `text` 表示。未来的多模态 backend 可以使用同一个 artifact 引用，而不需要改变 Episode 模型。

| 源数据 | 建议的 Evidence 单元 |
|---|---|
| 临床笔记 | 一份 note 或边界清楚的 section |
| 检验数据 | 同一采样时间的一组 panel，或具有独立意义的单项结果 |
| 影像或病理 | 一份报告 |
| FHIR | 一个 Resource，或表示同一临床事件的一小组 Resource |
| 登记数据 | 一次诊断、分期、治疗或随访记录 |
| 医学 QA | 一段给定的上下文材料 |
| 图片、PDF 或波形 | 一个 artifact；需要时附带忠实文本表示 |

### Decision

模型在每个完成的 Turn 中提交恰好一个被接受的 Decision：

```json
{
  "turn_id": "t1",
  "state": {},
  "action": {"name": "order_biopsy", "arguments": {}},
  "citations": ["path-001"],
  "abstain": false,
  "note": ""
}
```

`state` 和 `action.arguments` 是由数据集定义、由 scorer 解释的 JSON object。Citation 必须指向当前可见且已经被模型读取的 Evidence。`abstain=true` 时，`state`、`action` 和 `citations` 必须为空。

## 数据集元信息

`dataset.json` 的结构如下：

```json
{
  "schema": "ama-dataset-v0",
  "name": "example",
  "version": "0.1",
  "description": "数据测量目标和来源。",
  "splits": {"all": ["case-001"]},
  "scorer": "exact_v0",
  "license": "源数据许可证"
}
```

Description 应说明 Episode 是单轮还是多轮、研究测量什么、样本是原始还是派生，以及哪些样本不评分。

## Targets 与 policy

`targets.jsonl` 每行保存一个被评分 Episode 的 evaluator-only 记录：

```json
{"episode_id":"case-001","turns":{"t1":{"answers":["example"]}}}
```

Target 内容由 scorer 定义。没有 target 的 Episode 仍然合法，评测结果中会标记为 unscored。

`policy.json` 分离模型可见指导和仅供 evaluator 使用的规则：

```json
{
  "public": {"guidance": "模型可见的说明。"},
  "hidden": {"evidence_tags": {}, "transitions": []}
}
```

Runner 可以把 `public.guidance` 加入 system prompt。Hidden policy 只在评测阶段使用。`workflow_v0` 使用 `hidden.evidence_tags` 把 Evidence ID 映射到 transition 的 `when` 条件所引用的 evaluator-only tag。

内置 scorer：

- `exact_v0`：状态或答案匹配、允许动作、必需 Evidence 和拒答。
- `workflow_v0`：状态字段、确定性迁移、动作约束、必需 Evidence 和累计成功率。
- `unscored`：只统计运行指标。

## Importer 约定

Importer 是离线转换步骤，应当：

1. 保留稳定的源数据 ID 和 provenance。
2. 明确说明 Evidence 单元和 Turn 释放规则。
3. 保留已知源时间；单轮数据缺少时间时使用 null，不要猜测。
4. 保证 gold 数据不会进入 `episodes.jsonl`。
5. 把 artifact 复制到 `assets/`，或写入安全的相对引用。
6. 在 `import_report.json` 中记录排除项、缺失字段、截断、源 hash 和时间处理假设。
7. 对输出运行 `ama validate`。

数据集特有的转换和评分逻辑属于 importer 与 scorer，不应给 `run_episode()` 增加分支。

## Dataset Card 约定

公开发布和提交到仓库的数据集应包含 `DATASET_CARD.md` 与 `DATASET_CARD.zh-CN.md`。v0 将数据卡作为推荐标准，validator 暂不强制。Importer 会生成包含以下固定章节的初始数据卡：

```text
来源与许可
研究用途
构造方法
规模
评分与 targets
Artifacts
限制与报告口径
```

发布前，数据作者应把自动生成内容替换为源数据特有的信息。数据卡必须说明派生样本、缺失 targets、时间代理、截断情况，以及结果不支持哪些结论。

## 数据集位置

CLI 命令接受已经存在的数据集目录。如果参数不是现有路径，并且设置了 `AMA_DATA_ROOT`，AMA 会在该根目录下解析数据集名称。这样可以支持本地磁盘和由操作系统挂载的 NAS，不需要增加新的存储协议：

```bash
export AMA_DATA_ROOT=/Volumes/lab-data/ama
ama validate my_dataset
```

Run manifest 保存解析后的绝对目录；import 输出仍然要求显式路径。

## 不同研究的建议

单轮可信推理使用一个 Turn，在该轮释放所有允许的上下文，对答案、引用、结构化状态或拒答评分。单轮 Episode 可以没有时间。

多轮记忆研究按信息真正可见的时间组织观察，每轮只放新 Evidence，并固定回放顺序。数据卡应说明研究测量的是回忆、判断更新、一致性还是工作流决策。Teacher-forced replay 不应描述成可交互的患者模拟。

对于派生 benchmark 数据，应明确区分原始任务和人工构造的 Episode。除非源 benchmark 本身定义了这一协议，否则纵向派生回放的成绩不属于源 benchmark 成绩。
