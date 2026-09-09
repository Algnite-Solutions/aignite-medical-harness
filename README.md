# medical-harness v0.2（multi-turn clinical trajectory）

一个**最小医学 Agent 研究仪器**，两层工作负载共用同一套环境与评测内核：

1. **Question 模式**（v0.1）：逐时间点开放证据，Agent 读取证据、维护状态、提交带引用的结构化答案；
2. **Trajectory 模式**（v0.2）：一位患者一个 episode，真实观察按诊疗时间逐轮到达（world observation），Agent 每轮更新诊断状态并选择下一临床动作，显式 JSON workflow 验证状态迁移与动作。**teacher-forced replay**：动作被评分，但不改变已记录的患者未来——报告一律称 replay/proposed-action correctness，不称闭环模拟。

**定位**：harness 是测量仪器，合成数据只是校准液；挑战从数据里观测而非预设。v0.2 刻意不接数据库/UI/多 Agent/向量库。架构参考 mini-swe-agent 的 Model/Agent/Environment 分离（未复制代码）。

## 快速开始（完全离线）

```bash
pip install -e ".[dev]"   # 或 export PYTHONPATH=src
pytest -q                                          # 84+1 个测试（默认离线，API 测试跳过）

# Question 模式
python -m medical_harness.cli run --config configs/mock.json
python -m medical_harness.cli evaluate --run-dir runs/<目录>
python -m medical_harness.cli profile --data-dir data/sim/inputs

# Trajectory 模式（teacher-forced episode）
python -m medical_harness.cli validate-episode --episode-dir episodes/dev/thyroid_001
python -m medical_harness.cli run-trajectory --config configs/trajectory_mock.json
python -m medical_harness.cli evaluate-trajectory --run-dir runs/<目录>
python -m medical_harness.cli run-episode --episode-dir episodes/dev/thyroid_001 --config configs/glm.json --interactive
```

## Episode Folder（自包含病例目录）

`episodes/<组>/<病例>/`：`episode.json`（患者+轮次+每轮释放证据）、`evidence.jsonl`（Evidence，含 evaluator tags / 可选 artifact 路径，loader 阻止 `../` 逃逸）、`workflow.json`（显式状态机；`when` 为证据 tag 前置条件）、evaluator-only `gold.json`、可选 `assets/`。研究者复制目录改 JSON 即可出题，`validate-episode` 离线检查 schema/时间单调/ID/引用/路径安全。

Interactive（`run-episode --interactive`）是**操作员监督的在线运行**：每轮 `run` 批准一次、实时看公开 tool calls、`note` 记批注、`quit`/Ctrl-C 落 `operator_stopped` 终因并保留完整 run 目录；operator 等待时间单独计量、不吃 episode deadline。与 batch 走**同一个** `run_episode()`（有测试保证两者决策一致）。

## 逐轮评分（规则判分，分子/分母显式）

`state_field_accuracy`（workflow_state/分期/假设标签）、`state_transition_validity`、`next_action_accuracy`（gold 允许集合，多个正确动作都算对）、`guideline_violation`（禁用动作/非法迁移）、`evidence_grounding`（引用必须来自已成功 read 的证据）、`abstention_accuracy`、`trajectory_completion`、cumulative trajectory success、calls/tokens/latency。失败轮保留在分母；某轮答错不阻断后续真实轮（teacher-forced）。

## Gate 0/1 测量修复（v0.2 已落地）

- 每次 API 调用的 usage 累计（不再只留最后一次）；retry 受 `max_model_calls`/deadline 约束；
- 引用前置约束：**可见但未 read 的证据不可引用**（提交与状态更新都强制）；
- 引用指标拆分：`citation_coverage`（无引用答案不逃出分母）/`citation_correctness`/`forbidden_citations`；数值-字符串正规化为显式 scorer policy（版本记录在 metrics）；
- run manifest：git commit+dirty、依赖版本、全部输入/episode 文件 SHA-256、experiment card（playbook §8）；事件带 run_id/call_id/单次 latency 与 usage；
- 敏感字段策略：`trace.save_model_context/save_evidence_content` 关闭时以摘要替代全文（真实患者数据前默认应关闭）；
- `full_history` 更名 `stateless_retrieval`（其真实行为是新会话+全量可检索+无持久状态，旧对照结论作废）。

## MedAgentBench 适配（两层，均已实跑）

- **离线层**：`profile-medagentbench --patient-limit 5` 审计官方 300 任务/98 患者（按 MRN 聚合、任务数排名、context 时间戳抽取、无 MRN 任务清单）。
- **FHIR 层**（2026-09-09 实跑，官方 Docker `jyxsu6/medagentbench` @ localhost:8080）：`fhir_patient_snapshot()` 聚合真实资源；纳入规则 = ≥3 有日期资源 + ≥2 类临床资源。**实测结果：98 名患者均为 Patient+Condition 单一临床类型（每人均 19 条 Condition）**——按规则全部如实排除出正式 dev set，不强行拼接。
- **管线演示 episode**（exploratory，不进正式比较）：`build_episode_folder()` 将 S2703270（真实肺癌 Condition 时间线，2023-10-10→11-13）生成 4 轮 episode（`episodes/medagentbench/S2703270/`，无 gold、宽松占位 workflow）。GLM 端到端 4/4 轮 completed：从 ICD-10 确认肺癌诊断、跨轮维护假设、随新 Condition 增量更新鉴别诊断，引用全部来自已读证据；evaluate 如实报"未评分（no gold.json）"。
- 边界：官方 agentbench 调度框架与官方 scorer 未接入；MedAgentBench-derived replay 一律单独报告，**不得**称为原始基准成绩。启动官方环境：`docker run -d -p 8080:8080 jyxsu6/medagentbench`。

## 接真实模型（GLM）

```bash
cp .env.example .env      # 填 GLM_API_KEY（.env 已 gitignore；key 不进代码/日志/产物）
python -m medical_harness.cli run --config configs/glm.json           # question 模式
python -m medical_harness.cli run-episode --episode-dir episodes/dev/thyroid_001 --config configs/glm.json
```

实测（2026-09-09，`glm-5.3-flash`，思考型模型，function calling 兼容）：
- Question 校准（合成 6 题）：两次均 6/6 completed，field_correct 4/5，延迟记录题正确弃答，越界 0；
- Episode 基线（thyroid_001 三轮）：3/3 轮 completed，**next_action 3/3**，workflow 迁移 0 违规，state_field 2/4；
- 观察到的可测失败模式（Gate 4 候选）：① workflow_state 语义——模型把"当前节点"理解为"动作出发点"而非流程已到达的阶段（t2 报 initial_imaging）；② 假设标签自由文本与 gold 受控词表不匹配；③ 探索性工具调用多（每轮 4–6 次），预算需按后端校准。这些是任务设计/研究问题素材，不是 harness 缺陷。
- 注意：`glm5.3-flash` 不存在，正确 ID 是 `glm-5.3-flash`（只写在 config，不进代码）。

## 目录

```
src/medical_harness/
  schemas.py       # 数据对象 + Action 联合 + workflow/trajectory/episode gold + 配置
  environment.py   # TimelineEnvironment(question) + EpisodeEnvironment(按轮释放) + 先读后引
  agent.py         # question 循环（预算/终因/usage 累计/per-call 事件）
  trajectory.py    # episode folder 加载与校验 + run_episode + 逐轮评分 + 批量编排
  workflow.py      # 显式 workflow 加载器 + 确定性校验器 + 公开指导摘要
  model.py         # ModelClient 协议 + ScriptedModel + OpenAI 兼容客户端(question/episode 工具集)
  runner.py / evaluation.py / profile.py / utils.py / cli.py
  adapters/medagentbench.py   # 离线审计层 + FHIR 层（可选外部依赖）
episodes/dev/thyroid_001/     # 三轮合成轨迹 episode（可直接 --interactive 在线逐轮跑）
data/sim/{inputs,gold,scripts}/                # question 模式校准数据
data/trajectory_synthetic/scripts/             # episode scripted 动作
configs/{mock,glm,trajectory_mock}.json
```

## 设计不变量（都有测试钉死，84 项）

时间可见性（recorded_time ≤ as_of，episode 为按轮释放）；患者隔离；先读后引；状态版本化与原子提交；冲突不自动裁决；预算终因枚举（completed/budget_exceeded/model_error/tool_error/operator_stopped）；gold/workflow 内部（when-tags、rule ids、期望状态与动作）永不进入模型上下文；scripted 满分只证明仪器工作。

## 边界与下一步

已实现：question + trajectory 双模式、workflow 校验器、episode folder、interactive、GLM 客户端、MedAgentBench 离线审计。未实现（刻意推迟）：MedAgentBench 3–5 患者 FHIR trajectory 生成（等 Docker 环境）、规则引擎/动作安全等干预（等 baseline failure review）、MIMIC 纵向队列、compare 命令、多模态 artifact 评分。

## 诚实性口径

- backend 强制标注；scripted 满分 ≠ 真实模型效果；合成小样本 ≠ 基准结论；replay ≠ 闭环模拟。
- `citation_correctness` 只验证引用属于 gold 允许集合，不声称语义忠实度。
- usage 如实记录（scripted 记 unknown）；费用不计算（无单价配置）。
- 相同配置不保证真实模型输出可复现；区分"回放已保存产物"与"重新调用模型"。
