# 一起读最小 Agent

阅读顺序固定为：模型请求 → 多轮历史 → 工具循环 → 数据与 CLI。
先用确定性测试看清行为，再花少量 API 费用验证接口。假模型仅在 tests。

## 1. 模型请求：model.py

`Model.from_config("qwen36")` 读取 ama.json 的别名，密钥来自指定环境变量。
CLI 会先加载本地 .env；独立 Python 使用时先 export 环境变量。
新增模型只需加一个配置对象，不需要写注册类：

```json
{
  "base_url": "https://your-provider/v1",
  "model": "provider-exact-model-id",
  "api_key_env": "YOUR_KEY",
  "temperature": 0
}
```

`complete(history, tools=None)` 只发一次请求，返回原始 assistant 消息。
`image(path)` 在历史里保存路径；发送请求时才编码成图像内容。
没有工具时不发 tools 字段。HTTP/网络错误直接抛出，不自动重试。

```bash
python3 -m pytest tests/test_model_client.py -q
```

可以一起改的地方：温度、输出 token 上限、请求超时。不要为了接口兼容偷偷改变消息角色。

## 2. 多轮历史：agent.py

公共接口是：

```python
from ama.agent import Agent
from ama.model import Model, image

agent = Agent(Model.from_config("qwen36"), system="Describe images and answer follow-up questions.")
first = agent.chat(["Describe this image.", image("datasets/rocov2_demo/artifacts/ROCOv2_2023_test_000001.jpg")])
second = agent.chat("Explain your previous answer in Chinese.")
print(first, second)
print(agent.history)
```

这两次 chat 使用同一个 history。创建另一个 Agent 才开始新历史。
Agent 不知道它收到的是数据集材料还是人的追问，也不解析 Decision。
代码不做摘要、截断或消息合并，超出模型上下文限制时会得到服务错误。

对应测试（不联网）：

```bash
python3 -m pytest tests/test_agent.py -k history -q
```

## 3. 工具循环：tools.py 与 Agent.chat

读 examples/tools.py：一个 Tool 显式写名称、说明、参数 JSON schema 和 Python 函数。
JSON schema 发送给模型；本地检查 JSON 对象和函数参数绑定，业务类型/范围由工具函数检查。
只有 TOOLS 列表里的函数可被模型调用。工具 Python 本身有当前进程权限，不是安全沙箱。

一次输入的循环是：

1. 发出完整历史，取回 assistant 消息。
2. 没有 tool_calls：返回最终文字。
3. 有 tool_calls：按顺序调用函数，逐个保留调用 ID，追加 tool 消息。
4. 所有 tool 消息完成后，再附带标注来源调用 ID 的图像消息；回到第 1 步。

`ToolResult(text="读取到图像", images=["scan.jpg"])` 是工具返回图像的方式。
图像附在 user 消息里，但日志 source=tool，不伪装成人的新追问。
工具异常、参数错误会变成工具结果，让模型在同一调用预算内修正。
API 错误则停止；不能伪装成成功的工具响应。

```bash
python3 -m pytest tests/test_agent.py -q
ama chat datasets/rocov2_demo --episode ROCOv2_2023_test_000001 --model qwen36 --tools examples/tools.py
# 第一轮结束后可以输入：Please use add to calculate 17 + 25.
```

每条输入最多请求模型 8 次（包含工具后的再次请求），并非最多执行 8 个工具。
达到上限时保留已发生的工具记录并停止此 Agent。

## 4. 数据与 CLI：runner.py

`execute` 读取并检查可见数据，再由 `render` 把本轮 observation/evidence 变为消息。
任务说明来自 instructions.txt，run 额外附加 Decision JSON 要求，chat 额外说明自然语言探索。
两者最终都调用 `exchange → agent.chat`。工具可选，主循环没有协议切换。

`run` 中每轮结束才在 Agent 外调用 `parse_decision`，检查 JSON、轮次和引用。
失败原文保留，不再要求模型修复。已有材料可继续引用，未来材料不可引用。
API 失败跳过本样本剩余轮次；下一个样本用新的 Agent。

`chat` 的普通输入不改变数据集轮次，只有 /next 改变轮次；不解析 Decision。
ROCOv2 本身只有一轮，所以 /next 会提示无后续材料。两轮行为由测试中的小数据集验证：

```bash
python3 -m pytest tests/test_cli.py -q
python3 -m pytest tests/test_rocov2.py -k import_run_eval -q
```

## 如何审阅一次实验

- manifest.json：模式、模型配置、数据集、预算、工具路径、预期轮次和运行状态。
- messages.json：以 episode ID 为 key 的有序消息数组，包含实际 system 提示及完整对话。
- diagnostics.jsonl：工具定义、调用统计、证据释放及错误；历史运行仍使用旧版 events.jsonl。
- decisions.jsonl：只在 run 中存在，保留解析结果、验证、轮次统计和 message_range；未执行轮次也有记录。
- metrics.json：只在独立 eval 后生成；逐例结果与汇总分数均保存在这里，不再生成报告文件。

数据集输入与人工追问都以 user 角色发给模型。message_range 是零起始、右端不含的消息索引范围。
eval 按预期轮次遍历，即使某一条 Decision 记录缺失，也按缺失回答处理。
不再计算或核对 SHA；仍检查病例和预期轮次一致性。数据内容变化不再自动检测。
targets.jsonl、eval.json 只在 eval 阶段打开，provenance.jsonl 始终不参与推理与评分。

建议学生每次只改一个问题：先写一个小测试证明预期，再改相应函数；
用 tests/test_agent.py 理解工具，用 tests/test_cli.py 理解实验边界。
