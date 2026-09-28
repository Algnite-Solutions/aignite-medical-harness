# AMA 数据格式

数据描述“按什么顺序给模型看什么”。数组顺序就是释放顺序；只有真实可用时间才填 `available_at`。

| 文件 | 内容 | 何时读取 |
|---|---|---|
| `dataset.json` | schema、名称、split | run、chat 和 eval |
| `episodes.jsonl` | 每行一个样本及其轮次、材料 | run、chat 和 eval |
| `instructions.txt` | 全局任务说明 | run 和 chat |
| `targets.jsonl` | 每轮参考答案 | 仅评测 |
| `eval.json` | 评分器和评分规则 | 仅评测 |
| `provenance.jsonl` | 原始来源定位 | 运行器不读取 |

`dataset.json`：
```json
{"schema":"ama-dataset","name":"example","splits":{"test":["image-1"]}}
```

`episodes.jsonl` 一行：
```json
{"id":"image-1","turns":[{"id":"t1","evidence":[{"id":"scan","type":"image","file":"artifacts/scan.jpg"}]}]}
```

Evidence 的 `text` 与安全相对路径 `file` 至少有一个。`type`、`observation` 和真实时间 `available_at` 都可省略。缺值直接省略，不写 null 或空字符串。JPEG、PNG、GIF、WebP 以图像消息发送；其他文件按 UTF-8 文本读取，不支持的二进制格式需先转换。全局任务要求写在 instructions.txt，不在 observation 里重复。

Decision：
```json
{"turn_id":"t1","answer":{"caption":"A chest image.","cuis":[]},"citations":["scan"],"reasoning_summary":"可见的解剖结构支持该描述。"}
```

`answer` 是任务自定义 JSON，null 表示弃答且不能附引用。引用只能指向截至当前轮已展示的 Evidence。

ROCOv2 的 `targets.jsonl` 一行：
```json
{"id":"image-1","turns":{"t1":{"answer":{"caption":"A chest image.","cuis":[]},"required_evidence":["scan"]}}}
```

`eval.json` 为 `{"scorer":"rocov2"}`。没有评分配置时，只记录完成情况。

run 与 chat 共用 Agent。runner 负责最终输出格式，数据集说明只描述任务及 answer 字段。
run 要求 2–4 句基于证据的 reasoning_summary；chat 保持自由交流。

输出协议 v2 保留原始回复，并在 Decision 旁记录 output_validation。完整 Decision 优先于
重复的简短答案，允许 JSON 前后有解释。冲突结论和明确错误的 turn ID 会导致提取失败。
仅含答案的 JSON 可以恢复，但缺少的引用与摘要仍标记为 missing。摘要只取显式字段或
标有 Reasoning summary 的段落，不把任意分析文字当作摘要。不会增加模型调用。

工具通过 ToolResult.evidence_ids 显式声明证据；只有成功调用才注册，并记录 evidence_release
事件。工具正文或影像目录中仅提到的 ID 不会自动注册。未知引用原样保留并标记无效，
不抹去已恢复的结论。解析后另记 decision 事件；exchange 仍是原始交互记录。

ama eval 将任务得分与 aggregate.output_quality 分开：答案可恢复率、严格格式合规率、
有效非空引用率、摘要存在率，分母均为预期轮次数。严格合规检查四字段 JSON 及类型，引用合法性
单独检查。这些指标不代表医学依据正确或摘要质量。旧版本记录仍可读取，新指标显示 null。

--instruction-file 可替换任务说明。manifest 记录实际提示、哈希、模型配置和协议版本。
只有 eval 读取参考答案；失败与缺失回答仍计入评测分母。

代码阅读与可执行例子见 [最小 Agent 导读](minimal-agent.zh-CN.md)。
