# ama — aigne-medical-agent

mini-swe-agent 式的最小研究运行器。核心只理解四个对象：

```text
Episode → Turn → Evidence → Decision
```

患者观察逐轮交给模型（teacher-forced 回放）；模型用三个工具（`list_evidence` / `read_evidence` / `submit_decision`）读证据并提交决策；运行器保存轨迹与预算/usage；数据集自己的 scorer 负责评分。核心不理解 MedAgentBench、MIMIC、FHIR、肿瘤 workflow 或具体指南——那些都属于 importer（离线转换）和 scorer（evaluator 侧）。

## 五分钟从安装到第一个 scored run

```bash
pip install -e ".[dev]"                # 1. 依赖：Python 3.11+ / pydantic / pytest
pytest -q                              # 2. 33 个离线测试
ama validate datasets/thyroid_demo     # 3. 校验数据集（离线）
ama run datasets/thyroid_demo --model scripted --runs-root runs   # 4. 离线运行
ama eval $(ls -dt runs/* | head -1)    # 5. 评分（此处才读 targets.jsonl）
```

真实模型（配置在 `ama.json`，key 走环境变量 `.env`）：

```bash
ama run datasets/thyroid_demo --episode thyroid_001 --model glm
ama run datasets/thyroid_demo --episode thyroid_001 --model glm --interactive
```

Interactive 提示符 `(ama) thyroid_001 t1/3 >`，命令：`run / show / trace / next / note <text> / quit`；与 batch 走**同一个** `run_episode()`（有测试保证决策一致；quit/Ctrl-C 落 `operator_stopped` 并保留完整 run）。

## 唯一数据标准：AMA Dataset v0（详见 docs/ama-dataset-v0.md）

```text
datasets/<name>/
  dataset.json     # 元信息、splits、scorer 名
  episodes.jsonl   # 一行一个 episode（单轮任务 = 只有一个 turn 的 episode）
  targets.jsonl    # evaluator-only；run 永不打开（有测试把 targets 移走重跑证明）
  policy.json      # public（可进 prompt）/ hidden（只给 scorer）
  assets/          # 可选文件；artifact 相对路径，validate 阻止逃逸
```

接入新数据集 = 新 importer，不改核心：`ama import medagentbench --source ... --out datasets/mb`（已实跑 300 任务 + FHIR 患者多轮轨迹；$everything 分页跟随全部 next 链并报告截断——修复了原型把第一页当全量的 bug）。旧 v0.2 Episode Folder 也有 `ama import episode-folder`。

## 工程能力（v0.2 原型保留项）

每次调用 usage/latency/call_id 累计；预算（episode 总调用/每轮调用/deadline/重试都受限）与终因枚举；**先 read 后 citation**；abstain 一致性校验；git commit+dirty、数据 hash、依赖版本进 manifest；trace 脱敏（`save_model_context/save_evidence_text`）；某轮失败不阻断后续真实轮（teacher-forced）；某轮 model_error 只记该轮。

## 内置 scorer（scorer.py，注册制扩展）

- `exact_v0`：state 精确匹配（数值-带单位字符串容差）/ answers 成员 / allowed_actions / required_evidence / expect_abstain
- `workflow_v0`：workflow_state、hypotheses、迁移合法性（policy.hidden.transitions + 证据 metadata.tags 前置条件）、allowed/forbidden actions、required evidence、逐轮 cumulative success
- `unscored`：completion / calls / tokens / latency

失败轮保留在分母；分母为 0 记 N/A；scripted 满分只证明运行器与评分器工作。

## 目录

```text
src/ama/
  data.py        # schema、loader、validator（AMA Dataset v0）
  model.py       # ScriptedModel + OpenAI-compatible（3 工具）；ama.json / ~/.config/ama/models.json
  agent.py       # 唯一 run_episode()：预算/usage/trace/interactive 都是它的横切能力
  scorer.py      # scorer protocol + exact_v0 / workflow_v0 / unscored
  recorder.py    # run 目录、events/decisions JSONL、manifest、脱敏
  cli.py         # ama validate/inspect/run/eval/import + interactive shell
  importers/     # medagentbench（含 FHIR 分页修复）、episode-folder
datasets/        # thyroid_demo（3轮 workflow）、qa_mini（单轮 exact）、medagentbench（实跑导入）
docs/ama-dataset-v0.md
```

旧 v0.2 原型保留在 git tag `v0.2-prototype`（双 runner、医学专用 schema、workflow 核心校验器等已按 AMA 边界删除；本分支不向后兼容旧 CLI）。

## 诚实性口径

scripted 满分 ≠ 真实模型效果；MedAgentBench-derived replay 分数单独报告，不得称为原始基准成绩；usage 如实记录（scripted 记 unknown）、费用不计算；相同配置不保证真实模型输出可复现。
