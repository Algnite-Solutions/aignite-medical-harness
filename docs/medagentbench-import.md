# MedAgentBench 本地数据导入

本阶段只准备数据。医疗工具和 `medagentbench` scorer 尚未实现，不能把生成数据集的成功当作完成 benchmark 接入；不要直接用 `ama eval` 评测它。

Importer 不调用模型、不运行原 grader，也不修改 FHIR。运行期间需要 FHIR 服务；保存快照后可以完全离线重复转换。使用现有 AMA Python 环境，无需新增第三方依赖。

生成数据放在 `datasets/medagentbench/`；该目录及下面示例的 `datasets/medagentbench_subset/` 已加入 Git 忽略规则。Importer 仍支持通过 `--out` 指定其他位置。

## 从 FHIR 导入

准备本地 MedAgentBench 仓库中的 `test_data_v2.json`、`funcs_v1.json`、`refsol.py` 和 `utils.py`。FHIR 服务必须包含原 benchmark 数据；仅启动一个空 FHIR 服务不够。

```bash
PYTHONPATH=src python -m ama.importers.medagentbench \
  --source ../MedAgentBench \
  --fhir-base http://localhost:8080/fhir/ \
  --out datasets/medagentbench
```

可增加 `--id task1_1`（可重复）或 `--limit 10` 选择任务。选择结果保持原始任务顺序；患者和观察资源仍全量导出，以保证患者检索不只包含选中任务的答案。`--timeout` 默认为每次请求 60 秒。

如果使用已有 Docker 镜像，可在独立端口启动新容器，并将 `--fhir-base` 改为实际端口：

```bash
docker run -d --name ama-medagentbench-import \
  -p 127.0.0.1:8081:8080 jyxsu6/medagentbench:latest
```

等待 FHIR 应用启动完成再导入。Importer 全量遍历 Patient 与 Observation 的分页；失败会清理临时输出，不发布半成品。已有输出目录拒绝覆盖，重新导入时选择新目录。

## 从快照离线导入

```bash
PYTHONPATH=src python -m ama.importers.medagentbench \
  --source ../MedAgentBench \
  --snapshot datasets/medagentbench/snapshot \
  --out datasets/medagentbench_subset \
  --id task1_1 --id task2_8
```

`--snapshot` 和显式 `--fhir-base` 互斥。快照目录必须有 `metadata.json`、`Patient.jsonl`、`Observation.jsonl`，且数量和 SHA-256 校验一致。复制后的快照再次验证；转换不执行网络请求。

## 输出文件与隔离

| 文件 | 内容 |
|---|---|
| `dataset.json` | 数据集名称、test 和 smoke splits |
| `episodes.jsonl` | 公开任务 instruction 和非空 context，一个任务一个 t1 |
| `instructions.txt` | 答案字段约定：`answer: {"answers": [...]}` |
| `targets.jsonl` | 原始 case_data，包括仅供评分的 sol、eval_MRN |
| `eval.json` | 未来使用的 medagentbench scorer 名称；目前未注册 |
| `patients.jsonl` | 患者 ID、原始标识符和姓名结构、可选出生日期与性别 |
| `cases.jsonl` | patient_id 和该患者的全部 observations |
| `snapshot/` | 完整原始 Patient、Observation 资源和快照校验信息 |
| `evaluator/` | 未执行的原 refsol.py、utils.py 和原函数说明副本 |
| `import_report.json` | 源文件摘要、资源数、分页数、选中任务与导入状态 |
| `DATASET_CARD*.md` | 中英文数据集说明 |

运行侧不应读取 targets 或 evaluator。患者 ID 来自 FHIR Patient.id，不是任务 ID，也不是 MIMIC 住院 ID。MRN 保存在 identifiers，不能假设所有服务器的 MRN 与 Patient.id 相同。

`test` 包含本次选中的任务，`smoke` 是选中任务与原 seed-42 固定 30 例样本的交集。默认完整 v2 数据应有 300 个任务、每类 30 个、smoke 每类 3 个。

观察记录包含 source_id、evidence_id、原始 code、可选 label/value，以及存在时的 unit、observed_at、参考范围等。所有 effective[x] 时间结构保留；单位不换算，时间不裁剪，同一项目多次测量不合并。没有数值不等于数值为零；缺失字段不虚构。其他源字段可由 source_id 在原始快照中追溯。

Importer 不计算年龄、最近检验、平均值、替代治疗剂量等任务答案。以后模型通过工具读取病例，grader 通过同一快照核验。

## 验证

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from ama.data import validate_dataset; print(validate_dataset(Path("datasets/medagentbench")))'
python -m pytest -q tests/test_medagentbench.py
```

结构验证应输出 `[]`。Importer 自身也会在发布输出目录前调用结构验证。此检查不代表临床工具或评分器已经可用。
