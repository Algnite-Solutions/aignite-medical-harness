# AMA Dataset

AMA 运行器只接受 `ama-dataset`。数据集描述固定的信息释放顺序，不决定模型如何交互。一个 Episode 包含有序 Turn；每轮释放 Evidence，最终记录一个 Decision。数组顺序具有权威性；`available_at` 只有真实时间才填写，多轮数据无需伪造日期。

必需文件是 `dataset.json`（仅 `schema`、`name`、`splits`）和 `episodes.jsonl`。可选的 `targets.jsonl` 与 `eval.json` 仅在评分时读取；`provenance.jsonl` 保存本地源标识和记录定位，推理与评分均不读取；`instructions.txt` 可保存导入器生成的任务说明，运行时需显式传 `--instruction-file`。

模型可见 Episode 使用 `{"id":"case-1","turns":[{"id":"t1","observation":"病史已到达","evidence":[{"id":"hpi","type":"note","text":"腹痛"}]}]}`。Evidence 只有 `id`、可选 `type`、可选 `text`、可选相对 `file`；`text` 或 `file` 至少有一个。缺失字段直接省略。Episode、Turn、Evidence 的 ID 在各自范围内唯一；若有真实 `available_at`，已知时间须单调。

Decision 统一为 `{"turn_id":"t1","answer":...,"citations":[]}`。`answer` 由任务定义，`null` 表示弃答且不能附引用。默认 `direct_decision` 直接展示当轮新证据；`tool_agent` 可先调用 `list_evidence`、`read_evidence`，且只能引用已读证据。工具调用只在轨迹里，不属于 Decision。

```bash
ama run datasets/thyroid_demo --model scripted --protocol tool_agent \
  --instruction-file datasets/thyroid_demo/instructions.txt
```

运行清单记录协议、完整任务说明、模型配置和模型可见输入哈希。详见 [English specification](ama-dataset.md)。
