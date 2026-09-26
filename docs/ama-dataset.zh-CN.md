# AMA 数据格式

数据描述“按什么顺序给模型看什么”。数组顺序就是释放顺序；只有真实可用时间才填 `available_at`。

| 文件 | 内容 | 何时读取 |
|---|---|---|
| `dataset.json` | schema、名称、split | 运行和评测 |
| `episodes.jsonl` | 每行一个样本及其轮次、材料 | 运行和评测 |
| `targets.jsonl` | 每轮参考答案 | 仅评测 |
| `eval.json` | 评分器和评分规则 | 仅评测 |
| `provenance.jsonl` | 原始来源定位 | 运行器不读取 |

`dataset.json`：
```json
{"schema":"ama-dataset","name":"example","splits":{"test":["image-1"]}}
```

`episodes.jsonl` 一行：
```json
{"id":"image-1","turns":[{"id":"t1","observation":"Describe this image.","evidence":[{"id":"scan","type":"image","file":"artifacts/scan.jpg"}]}]}
```

Evidence 的 `text` 与安全相对路径 `file` 至少有一个。`type`、`observation` 和真实时间 `available_at` 都可省略。缺值直接省略，不写 null 或空字符串。图像以图像消息发送给模型。

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

运行器只提供直接决策；工具协议通过 `InteractionProtocol` 扩展。额外任务说明用 `ama run --instruction-file ...` 传入，实验记录保存完整文本、哈希、模型配置和数据哈希。

实验人员阅读材料见 [中文指南](rocov2-lab-guide.zh-CN.md)；它不进入模型提示。
