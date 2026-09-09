# medical-harness v0.1（MVP）

一个**最小医学 Agent 研究仪器**：Agent 在受控的患者时间线环境里，用 5 个显式工具读取证据、维护状态并提交有引用的结构化答案；每一步被校验、计入预算、全量落盘、规则判分。

**定位（重要）**：harness 是测量仪器，合成数据只是校准液。延迟记录、修订、冲突这些"挑战"不预设进代码——等真实数据（MIMIC 等）进来后，用 `profile` 从数据里观测出来，再决定研究什么。v0 刻意不接任何外部基准（MedAgentBench）、任何真实模型 API、任何数据库。

## 快速开始（完全离线）

```bash
# 依赖：Python 3.11+、pydantic>=2.7、pytest。二选一：
pip install -e ".[dev]"          # 正式安装
export PYTHONPATH=src           # 或者免安装直接跑

pytest -q                                                        # 47+1 个测试（离线，API 测试自动跳过）
python -m medical_harness.cli run --config configs/mock.json    # 离线 scripted 运行
python -m medical_harness.cli evaluate --run-dir runs/<目录>    # 金标准评分
python -m medical_harness.cli profile --data-dir data/sim/inputs  # 数据结构观测
```

`run --verbose` 会打印逐步轨迹（动作 -> 结果），像 `codex exec` 的观感。

## 接真实模型（OpenAI 兼容 / GLM）

客户端是标准库实现的 OpenAI Chat Completions + 显式 function tools，任何兼容端点都能接。GLM 实测（2026-09-08）：端点 `https://open.bigmodel.cn/api/paas/v4`，模型 ID 为 **`glm-5.3-flash`**（注意：`glm5.3-flash` 不存在；模型 ID 只写在 config，不硬编码进代码）。

```bash
cp .env.example .env      # 填入你自己的 GLM_API_KEY（.env 已被 gitignore）
python -m medical_harness.cli run --config configs/glm.json
python -m medical_harness.cli evaluate --run-dir runs/<glm 目录>
```

密钥纪律：key 只从环境变量/`.env` 读取，只出现在请求头；代码、配置、日志、run 产物中均不落盘（有测试和 grep 验证）。真实 API 测试是显式 opt-in，默认 `pytest` 永远不联网：

```bash
set -a; source .env; set +a
RUN_API_TESTS=1 pytest -q tests/test_api_integration.py -m api
```

usage（token 数）逐题记录并聚合进 metrics；**费用不计算**（未配置单价，如实记 unknown，不编造）。glm-5.3-flash 是思考型模型（回复含 reasoning_tokens），工具调用兼容良好。

### GLM 校准运行观测（合成数据，样本=6 题，非基准结论）

两次全量运行均为 6/6 completed：field_correct 4/5、correct_abstain 1/1（延迟记录题正确弃答）、violations 0。可复现的错误信号有两类，都属于任务设计层面的观测而非代码缺陷：
1. **自由文本格式方差**：药物名回答与 gold 字符串不等（如"阿司匹林 100mg 每日一次（aspirin 100mg daily）" vs "aspirin 100mg daily"）——正式任务需在题目侧规定 value 格式或受控词表；
2. **过度引用**：更正后的问题上，模型倾向同时引用被替代的原始报告与更正件，被 stale_refs 指标稳定捕获——"当前值"类问题的引用语义需要更明确的题目说明。

评分器已做的唯一容差：数值 vs 带单位字符串（"1.2 mg/dL" == 1.2）；除此之外保持严格匹配，不做语义打分。

## 设计不变量（都有测试钉死）

| 不变量 | 机制 |
|---|---|
| 时间可见性 | 仅 `recorded_time <= as_of` 的证据可见；`event_time` 再早也不豁免（延迟记录测试） |
| 患者隔离 | `patient_id`/`as_of` 是环境构造参数，不在任何工具签名里；跨患者读取/引用被拒 |
| 状态版本化 | `expected_version` 乐观并发；整批校验原子提交；旧版本全保留；版本冲突拒绝 |
| 冲突不裁决 | runner 永不把"最新即真"写进状态；矛盾 claim 并存，判分交给 gold |
| 预算终止 | max_steps / max_model_calls / deadline / 有限重试；终因枚举 completed/budget_exceeded/model_error/tool_error |
| gold 隔离 | gold 只被评测器读取，从不进入模型上下文/工具列表/run 阶段产物 |
| 可见错误 | 非法动作与工具校验失败回给模型并计入预算；内部错误才算 tool_error；无静默裁剪 |

安全细节：越界访问（未来/他人证据）在事件里打 `violation` 标签供指标统计，但给模型的错误信息是统一的"not found"——不泄露不可见证据的存在。

## 合成校准数据（data/sim/）

| case | 校准目标 |
|---|---|
| case_gamma | 无冲突新增、跨时间点状态版本化 |
| case_alpha | 延迟记录（应弃答）、明确更正（引旧值=过期引用） |
| case_beta | 跨患者同名字段隔离 |

gold 独立存放于 `data/sim/gold/`（只被 `evaluation.py` 读取）。scripts（`data/sim/scripts/`）是 ScriptedModel 的预写动作序列，含正确路径；测试里另有**故意答错**变体证明评分器有分辨力。

## 目录

```
src/medical_harness/
  schemas.py       # 数据对象 + Action 判别联合 + 配置（Pydantic, extra=forbid）
  environment.py   # 时间线环境：5 工具 + 可见性 + 隔离 + 版本化状态
  agent.py         # 循环：上下文->动作->校验->执行->记录；预算与终因
  model.py         # ModelClient 协议 + ScriptedModel（离线默认）
  runner.py        # run 目录产物：events/state_snapshots/answers/metrics/report
  evaluation.py    # 规则判分（分子/分母显式，0 分母=N/A，失败任务留在分母）
  profile.py       # 数据结构观测（滞后分布/重复/模态）——"从数据看挑战"的通路
  cli.py           # run / evaluate / profile
data/sim/{inputs,gold,scripts}/   configs/mock.json   tests/
```

每次 run 生成唯一目录 `runs/<UTC时间戳>-<name>-<rand>/`：`resolved_config.json`（含输入哈希与代码源哈希）、`events.jsonl`（逐步动作/观测/错误/模型上下文/usage）、`state_snapshots.jsonl`、`answers.jsonl`、`metrics.json`、`report.md`。

## 诚实性口径

- **backend 标记**：scripted 满分只证明运行器与评分器工作；真实模型运行标注具体模型名，且合成校准集样本极小，不构成基准结论或临床验证；报告页眉强制标注。
- `valid_refs` 只验证引用属于 gold 允许集合，不声称语义忠实度。
- scripted 后端无 token usage，如实记 `unknown`；真实后端记录实际 token 数；费用一律不计算（无单价配置）。
- 回放=读取已保存产物；相同配置不保证真实模型输出可复现（temperature=0 下思考型模型仍可能有方差）。

## 边界与下一步

已实现：离线 scripted 全链路 + OpenAI 兼容客户端（GLM 实测）。未实现（刻意推迟）：MedAgentBench 适配器、MIMIC 数据适配、语义检索/摘要记忆、费用计算、compare 命令。

路线：真实数据一进来 → `profile` 观测实际现象（滞后/冲突/重复分布）→ 数据驱地决定任务与金标准 → 再比较 full_history vs versioned_state 两策略。

架构参考了 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 的 Model/Agent/Environment 分离与极简循环（未复制代码；commit 记录因本机网络受限待实现阶段补上，不阻塞离线开发）。
