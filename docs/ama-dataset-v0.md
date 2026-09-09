# AMA Dataset v0

一个数据集 = 一个目录。核心 runner 只理解 `Episode → Turn → Evidence → Decision`，不理解 FHIR / MedAgentBench / MIMIC / 肿瘤 workflow / 指南。

```text
datasets/<name>/
  dataset.json     # 必需：元信息、splits、scorer 名
  episodes.jsonl   # 必需：一行一个完整 episode（模型可见）
  targets.jsonl    # 可选：evaluator-only，runner 永不打开
  policy.json      # 可选：public/hidden 两段，public 可进 prompt
  assets/          # 可选：图像/PDF 等；evidence.artifact 相对路径
```

## Episode / Turn / Evidence

- **Episode**：`episode_id`、`subject_id`、`turns[]`、`metadata{}`。
- **Turn**：`turn_id`、`time`（ISO 8601，**单轮 episode 可为 null**；多轮必须非空且单调不减）、`message`（world observation，进入 user 消息）、`evidence[]`（本轮新增证据；此前各轮证据保持可见）。
- **Evidence**：`evidence_id`、`kind`、`text`、`artifact`（数据集目录内相对路径，禁止绝对路径与 `..` 逃逸；存在则由 validate 检查文件存在）、`source`（provenance 定位）、`metadata{}`（原始字段全部放这里；runner 不依据 metadata 分支）。

原始字段一律放 `metadata`。可见性语义：turn N 可见 = 前 N 轮 evidence 之和（按轮释放，无 recorded/event 双时间；需要延迟记录语义的数据集在 importer 阶段重排进轮次）。

## Decision（模型每轮恰好提交一次）

```json
{
  "turn_id": "t1",
  "state": {},                                   // 数据集自定义 JSON
  "action": {"name": "order_biopsy", "arguments": {}},   // 可为 null
  "citations": ["us-001"],                       // 必须来自本轮前已成功 read_evidence 的证据
  "abstain": false,                              // true 时不得有 action/citations/state 内容
  "note": ""
}
```

`state` 与 `action.arguments` 的内部结构由数据集（的 scorer）定义；外壳稳定，核心不扩 class。

## targets.jsonl（evaluator-only）

一行一个 episode：`{"episode_id": ..., "turns": {"t1": {...}}}`。**Runner 不打开、不 hash 进 run manifest**；`ama eval` 才读取并记录其 hash。Turn target 内容由 scorer 解释（见下）。

## policy.json

```json
{"public": {...}, "hidden": {...}}
```

Loader 强制拆分：`public` 可注入 system prompt（例如面向临床的流程指导）；`hidden` 只给 scorer（例如迁移条件、when-tags）。run 事件中只出现 public。

## Scorer 约定（内置三个）

- **exact_v0**：target turn 支持 `state`（与 decision.state 精确相等，数值带单位字符串容差）、`answers`（decision.state.answer ∈ 列表）、`allowed_actions`、`required_evidence`（⊆ citations）。
- **workflow_v0**：target turn 支持 `state`（字段级匹配，如 `{"workflow_state": "...", "hypotheses": {label: status}}`）、`allowed_actions`、`forbidden_actions`、`required_evidence`、`expect_abstain`；`policy.hidden.transitions` 提供 `from/to/when(证据 metadata.tags)/allowed_actions` 的确定性迁移校验（校验对象是已提交的 decisions，不进 loop）。
- **unscored**：只报 completion / calls / tokens / latency / termination。

所有 scorer 输出逐 turn 分子分母 + 汇总；无 target 的 episode 记 `unscored (no target)`；失败轮保留在分母。

## 新数据集接入 = 新 importer（离线转换）+（必要时）新 scorer

`ama import <kind> --source ... --out datasets/<name>` 输出合法目录与 adapter report（纳入/排除、丢失字段、时间代理、分页截断、源 hash）。不允许为接入数据集修改 `run_episode()`。
