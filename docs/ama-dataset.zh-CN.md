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
{"turn_id":"t1","answer":{"caption":"A chest image.","cuis":[]},"citations":["scan"]}
```

`answer` 是任务自定义 JSON，null 表示弃答且不能附引用。引用只能指向截至当前轮已展示的 Evidence。

ROCOv2 的 `targets.jsonl` 一行：
```json
{"id":"image-1","turns":{"t1":{"answer":{"caption":"A chest image.","cuis":[]},"required_evidence":["scan"]}}}
```

`eval.json` 为 `{"scorer":"rocov2"}`。没有评分配置时，只记录完成情况。

run 与 chat 共用 Agent。run 在 Agent 外解析 Decision，chat 允许自由追问且不可评分。
`--tools file.py` 可选加载显式 TOOLS 列表。任务说明默认读取 instructions.txt，
`--instruction-file ...` 可替换；记录保存实际提示、指令哈希、模型配置、预期轮次和可见数据哈希。
数据检查自动执行，参考答案与规则仅在 eval 时打开。失败与缺失回答仍保留在评测分母中。

代码阅读与可执行例子见 [最小 Agent 导读](minimal-agent.zh-CN.md)。
