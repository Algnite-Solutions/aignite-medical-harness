# medagentbench——数据卡

中文 | [English](DATASET_CARD.md)

## 来源与许可

这些样例来自 MedAgentBench 及其 FHIR 环境的导入过程。重新分发前请查阅上游 MedAgentBench 的许可和访问条件。

## 研究用途

仓库内子集是一个精简的多轮样例，用于基于 FHIR 患者记录的纵向上下文和记忆研究。

## 构造方法

导入时选择三个患者，将带时间的 FHIR Resource 排序并按日期分组；每个患者最近四个有记录的日期构成四个 teacher-forced Turn。较早 Turn 的 Evidence 持续可见。

## 规模

- Episode：3
- Turn：12
- Evidence：25

## 评分与 targets

该子集使用 `unscored`，不包含 `targets.jsonl`，只报告完成情况和运行指标。

## Artifacts

不包含外部 artifact。每条 FHIR 派生 Evidence 都有简短文本表示和来源定位。

## 限制与报告口径

该序列是派生的 teacher-forced replay，不是原始交互式 FHIR 任务协议，也没有逐轮临床 gold 标签。结果不得作为原始 MedAgentBench benchmark 成绩报告。
