# MIMIC-CXR 导入与小样本运行

本次交付支持 MIMIC-CXR-JPG 2.1.0 的 study 级、单轮报告生成：输入为当前
study 的全部图片及已有视角，输出为 findings / impression。已完成 standalone
raw 准备、AMA importer、合成测试及三例真实模型运行。评分器仍为 `unscored`，
尚无报告质量分数或公开基线数值复现。

- [任务协议与限制](mimic-cxr-protocol.md)
- [验证记录](mimic-cxr-ac-validation.zh-CN.md)
- [聚合源数据审计](mimic-cxr-audit-2026-10-01.json)

## 1. 安装与源数据

在仓库根目录执行，Python 3.11+：

```bash
python3 -m pip install -e '.[dev,mimic-cxr]'
python3 -m pytest -q
```

源目录须包含 `files/` 下的 JPG、`mimic-cxr-2.0.0-split.csv.gz`、
`mimic-cxr-2.0.0-metadata.csv.gz` 和 `mimic-cxr-reports.zip`。
完整文本报告来自 MIMIC-CXR；JPG 发布包本身主要提供图片、元数据和结构化标签。
本准备脚本要求 ZIP 成员路径以 `files/pXX/p<subject_id>/s<study_id>.txt` 结尾。
请在取得相应数据访问权限后准备本地材料。

下面的 `/Volumes/data/incoming/mimic-cxr-jpg-2.1.0` 是本次运行使用的挂载路径，
其他机器需替换 `--source`。输出目录必须尚不存在；重跑时换新名称。

## 2. incoming → standalone raw

```bash
python3 scripts/prepare_mimic_cxr_raw.py \
  --source /Volumes/data/incoming/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/raw_validate_demo \
  --split validate --limit 3
```

脚本关联患者、study、图片和报告，统计报告章节提取失败的原因，然后只复制选中的
图片和报告。raw 包含 `manifest.jsonl`、`images/`、`reports/`、`selection_report.json`。
它不依赖 incoming 的链接，复制后可独立使用。需要固定病例时，重复传入
`--study-id <ID>`，且 `--limit` 不得小于指定 ID 的数量。

## 3. raw → AMA

```bash
ama import mimic-cxr \
  --source local_data/mimic_cxr/raw_validate_demo \
  --out local_data/mimic_cxr/processed_validate_demo \
  --split validate

python3 - <<'PY'
from pathlib import Path
from ama.data import validate_dataset
errors = validate_dataset(Path('local_data/mimic_cxr/processed_validate_demo'))
print(errors)
assert not errors
PY
```

导入器在临时目录生成并验证数据，成功后才发布最终目录：

- `episodes.jsonl`：当前图片及视角，供模型读取。
- `targets.jsonl`：从报告提取的 findings / impression，仅供 evaluator 读取。
- `artifacts/`：实际复制的 JPG。
- `dataset.json`、`instructions.txt`、`eval.json`：数据划分、默认指令、评分设置。
- `provenance.jsonl`、`import_report.json`：来源、hash、选择与计数记录。
- `DATASET_CARD.md`、`DATASET_CARD.zh-CN.md`：自动生成的数据卡。

`prepare_raw` 负责源材料关联与复制；`import_mimic_cxr` 只读 raw 并生成 AMA；
运行器只读取可见数据，`ama eval` 才读取 targets。图片损坏、hash 不符、错误关联、
选中材料缺失或已有输出会报错。自动选择时未解析出必需报告章节的病例会被记录并排除。

## 4. 小样本模型运行

按 `.env.example` 配置密钥，并在个人 `ama.json` 中注册可用的视觉模型别名。
下列 `YOUR_VISION_MODEL_ALIAS` 须替换为自己的别名。API 运行会发送所选图片。

```bash
ama run local_data/mimic_cxr/processed_validate_demo \
  --model YOUR_VISION_MODEL_ALIAS \
  --split validate \
  --instruction-file examples/mimic_cxr_instructions.txt \
  --runs-root local_data/mimic_cxr/runs \
  --max-calls 1 --timeout 120
```

`examples/mimic_cxr_instructions.txt` 是第三轮实际使用的 v3 提示词，明确 Decision
外层结构及字符串内换行规则。它通过 `--instruction-file` 替换导入器的简短默认指令；
复现本次提示词设置时不要省略该参数。提示词完全相同不保证随机生成结果相同。

将运行输出中的目录传给 evaluator：

```bash
ama eval local_data/mimic_cxr/runs/<运行目录>
```

当前 `aggregate` 为 `null` 是预期行为：`unscored` 只报告完成情况，没有报告质量评分。
本次三个 study 已用于提示词调试，不能当成独立评估集。正式指标、独立样本及公开基线
对齐属于后续工作。

## 数据与提交边界

真实 JPG、报告、逐例来源和模型 trace 保留在 Git 忽略的 `local_data/`。仓库提交
代码、合成测试、通用提示词、文档和聚合审计，不提供真实病例 demo。
`ama.json` 中的个人模型配置与 `.env` 密钥分别由使用者管理。

审计脚本还需要 CheXpert、NegBio 和 2.1.0 人工标签 CSV。它只统计全量 CSV、
ZIP 成员名及每个 split 前 20 张图的存在性；不能证明全量图片可解码：

```bash
python3 scripts/audit_mimic_cxr.py \
  --source /Volumes/data/incoming/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/audit_rerun.json
```
