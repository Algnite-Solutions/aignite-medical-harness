# AMA — 实验室最小评测器

中文 | [English](README.md)

目前只保留 **ROCOv2 看图描述**：给模型一张影像，得到英文描述和医学概念编号，再与参考答案比较。

`Episode（一个样本）→ Turn（一轮）→ Evidence（图像）→ Decision（模型答案）`

ROCOv2 每个样本只有一轮。核心循环仍能按顺序处理多轮；模型回答不会改变下一轮材料。

## 先离线跑通

```bash
python3 -m pip install -e ".[dev]"
ama validate datasets/rocov2_demo
ama run datasets/rocov2_demo --model scripted
```

最后一行会打印运行目录和对应的 `ama eval` 命令。执行它后打开该目录的 `report.md`：图像、模型原文、参考原文和中文指标说明都在里面。`scripted` 只播放三个预写答案，不访问云端，也不代表模型能力。

看不懂医学英文时，先读 [ROCOv2 中文阅读指南](docs/rocov2-lab-guide.zh-CN.md)：包含三个样本参考描述的中英对照、术语和分数解释。

## 再使用视觉模型

模型配置见 `ama.json`，密钥写在本地 `.env`（见 `.env.example`）。

```bash
ama run datasets/rocov2_demo --model glm-vision \
  --instruction-file datasets/rocov2_demo/instructions.txt
```

模型每轮直接收到图像并返回一个 JSON：
```json
{"turn_id":"t1","answer":{"caption":"A chest image.","cuis":[]},"citations":["image-ID"]}
```
`image-ID` 换成该样本的真实 Evidence ID。`answer: null` 表示弃答。

## 代码从哪里读

按下面顺序看即可：

| 文件 | 负责什么 |
|---|---|
| `data.py` | 定义四个数据对象、读取和检查数据 |
| `agent.py: run_episode` | 依次展示每轮材料、请求模型、记录答案 |
| `protocols.py: DirectDecisionProtocol` | 组装提示、发送图像、解析和检查答案 |
| `model.py` | 把消息发给模型，取回原始响应 |
| `scorer.py: score_rocov2` | 比较答案中的描述词和概念编号 |
| `cli.py` / `recorder.py` | 命令入口、报告和实验记录 |

CLI 目前只运行直接决策。工具使用留在 `InteractionProtocol` 接口：以后实现自己的协议对象并传给 `run_episode(protocol=...)`，由它提供工具列表并处理响应。当前没有内置工具代理或工具模式开关。

## 数据

唯一内置示例是 `datasets/rocov2_demo`。导入完整数据或子集：

```bash
ama import rocov2 --source /path/to/ROCOv2 --out datasets/rocov2 --split test --limit 100
```

模型只读取 `dataset.json`、`episodes.jsonl` 和其中引用的图像。`targets.jsonl` 和 `eval.json` 在评测时读取。`provenance.jsonl` 保存来源。中文阅读指南供实验人员使用。

完整字段见 [数据格式](docs/ama-dataset.zh-CN.md)。运行记录保存在 `runs/`，测试使用 `pytest -q`。
