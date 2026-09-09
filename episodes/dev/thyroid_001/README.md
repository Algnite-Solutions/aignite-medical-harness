# thyroid_001

合成校准 episode（非真实病例）：68 岁女性甲状腺结节的三轮 teacher-forced 轨迹
（超声 → 增强 CT → 穿刺病理）。用于校准 episode 循环、workflow 校验器与逐轮评分。

- `episode.json`：患者 dev-p001，3 个 turn，每轮释放一条证据
- `evidence.jsonl`：3 条本患者证据 + 1 条其他患者（dev-p999）诱饵，用于患者隔离
- `workflow.json`：oncology_diagnosis_v0（显式状态机；`when` 为证据 tag 前置条件）
- `gold.json`：evaluator-only，逐轮允许动作集合与期望状态；**不得进入模型上下文**

限制：合成数据只验证机制，不代表任何真实诊疗或模型结论。
