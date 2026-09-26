# AMA — 一个最小化的临床Agent

中文 | [English](README.md)

Agent 只做三件事：保存历史、请求模型、执行可选工具。它不认识 Episode，也不负责评分。
实验层才决定“什么时候把数据集的下一条 observation 给它”。

## 安装

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
# 参照 .env.example，把密钥写入本地 .env。
ama run datasets/rocov2_demo --model qwen36
ama eval runs/<运行目录>
ama chat datasets/rocov2_demo --episode ROCOv2_2023_test_000001 --model qwen36
```

模型在 `ama.json` 注册；测试假模型只放在 tests，不是运行选项。
注册成功不代表服务一定支持图像或工具，二者混用还需小样本检查。

本次小样本兼容性检查：GLM 返回合法 Decision，但缺少 caption（按空预测评分）；
Qwen 纯文本工具及追问通过，但本次图像输出未满足 Decision 格式，工具返回图像被接口拒绝。
`--model glm-vision` 已跑通运行与评测流程，不代表任务答案完整。这不是模型能力排名。

- `run`：自动依次释放 observation，每个样本用一个新 Agent，每轮返回 JSON Decision。
- `chat`：自动释放第一轮；普通输入是追问，`/next` 才推进一轮，`/quit` 保存退出。
- 最后一轮之后仍可追问；chat 可自然语言回答，是探索记录，**不能评测**。

两者自动检查可见数据，默认使用数据集的 `instructions.txt`。
`--instruction-file 文件` 替换任务说明；`--tools examples/tools.py` 显式启用工具。
默认没有工具，没有摘要或历史裁剪，没有自动重试或 JSON 纠错对话。
`--max-calls 8` 是每条输入的模型调用上限；`--timeout 60` 是单次请求超时秒数。

## 阅读顺序

按“模型请求 → 多轮历史 → 工具循环 → 数据与 CLI”阅读
[最小 Agent 导读](docs/minimal-agent.zh-CN.md)，里面有可直接执行的例子与对应测试。

| 位置 | 唯一职责 |
|---|---|
| `model.py` | 配置别名、发送前编码图像、发一次请求 |
| `agent.py` | 同一个 history 上追加消息，循环至最终回复 |
| `tools.py` | 显式工具定义、Python 函数绑定、返回值 |
| `runner.py` | 释放材料、检查 Decision、run/chat 流程 |
| `cli.py` | 命令入口与独立 eval |
| `data.py` / `scorer.py` | 数据对象与检查 / 评分定义 |
| `recorder.py` | 原始消息、决策和实验配置记录 |

工具文件是可信 Python，不是沙箱。不要让工具读取参考答案或未来轮次。
没有历史协议层、暂停菜单或图像/工具角色改写；服务不支持时明确报错。

## 数据和记录

以ROCOv2为例。每张图像是一例、一轮；也只能多轮对话数据。

```bash
ama import rocov2 --source /path/to/ROCOv2 --out datasets/rocov2 --split test --limit 100
ama run datasets/rocov2 --model qwen36 --split test
```

也可用可重复的 `--episode ID` 替代 `--split`，`--runs-root 目录` 指定输出父目录。

运行目录保存实际提示、模型配置、数据哈希、预期轮次、逐条消息来源、原始回复、
工具调用 ID/结果和终止原因。图像只记录路径，不把 base64 写进日志。
普通追问和数据集材料在日志里分别为 human / dataset，发给模型时都是 user。

`chat` chat模式下没有关于decision格式的prompt指引，作为交互和数据集观察用。

`run` 的 JSON 无效时保留原文、记失败，不追加纠错对话。
API 错误或调用上限终止当前样本，继续下一例；Ctrl-C 保存后停止整个批次。
失败和缺失回答不会从评分分母中消失。部分失败时 CLI 返回非零状态，仍可独立 eval。

`eval` 才读取参考答案和评分配置，只生成 `metrics.json`；
评测前检查可见数据是否已改变。历史实验记录保留，但需要重新运行才能交给新评测器。
详见 [字段说明](docs/ama-dataset.zh-CN.md)。

模型原文保留在 `decisions.jsonl`，参考原文在数据集的 `targets.jsonl`，按样本和轮次 ID 对照。
词重合与 CUI 重合不是临床正确率。
