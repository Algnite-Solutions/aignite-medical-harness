# thyroid_demo——数据卡

中文 | [English](DATASET_CARD.md)

## 来源与许可

这是为 AMA 创建的合成 toy 数据集。许可：用于仓库测试的合成示例数据。

## 研究用途

用于演示可评分的多轮工作流推理：随着超声、增强影像和病理证据到达，模型持续更新甲状腺肿瘤状态。

## 构造方法

一个合成患者构成一个 Episode，三份带时间的报告按时间顺序分三个 Turn 释放。仅供 evaluator 使用的迁移标签保存在 `eval.json`。

## 规模

- Episode：1
- Turn：3
- Evidence：3

## 评分与 targets

数据集包含一条 target 记录，使用 `workflow` 评估 answer 字段、迁移、建议下一步和引用。

## Artifacts

不包含外部 artifact；所有 Evidence 都有模型可读文本。

## 限制与报告口径

这是 toy 工作流，不是临床 benchmark。Scripted 模型结果只用于验证 runner 和 scorer，不得作为医疗能力证据。
