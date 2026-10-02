# MIMIC-CXR——数据卡

## 来源与用途

[MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) 的当前
study 胸片报告生成任务。完整报告来自 MIMIC-CXR，需另行准备。
遵守 PhysioNet 受控数据访问条款；本目录仅提供数据卡和提示词，未发布真实图片或报告，
也不是可直接传给 `ama run` 的 demo 数据集。

## 构造与答案隔离

每个 study 对应一个单轮 Episode，同次检查全部图片按 dicom_id 排序成为 Evidence，
仅附已有 ViewPosition。findings / impression 仅保存到 evaluator-only targets。
保留官方 train / validate / test 划分，不输入既往图片、报告或疾病标签。

`explicit-sections-v1` 只接受明确的 findings / impression 标题，要求两部分非空，
不使用官方 parser 的别名或末段回退规则。未提取到章节不代表临床内容确实缺失。
自动选择先取 study_id 数值顺序中首个合格单图、多图 study，再按数值顺序补足；
数值顺序不表示时间先后。显式选择可重复传入 `--study-id`，limit 不得小于 ID 数量。

## 已验证规模与限制

本地验证子集：3 个 validate study、3 个 Turn、7 张 JPG、3 条 targets。
它们是提示词调试样本，不是独立测试集。离线测试 83 项通过；GLM 三轮输出格式通过数
依次为 2/3、2/3、3/3，不能据此推断医学正确率或长期稳定性。

Scorer 为 `unscored`，eval 的 aggregate 为 null。报告质量指标和公开基线数值对齐
尚未完成；当前输入和严格章节规则不保证与公开基线一致。

## 使用方法

在仓库根目录执行，Python 3.11+。安装并测试：

```bash
python3 -m pip install -e '.[dev,mimic-cxr]'
python3 -m pytest -q
```

源目录需有 `files/`、split/metadata CSV.gz，以及 `mimic-cxr-reports.zip`。
ZIP 报告成员路径应以 `files/pXX/p<subject_id>/s<study_id>.txt` 结尾。
替换下面的源路径；输出目录必须尚不存在。

```bash
python3 scripts/prepare_mimic_cxr_raw.py \
  --source /path/to/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/raw_validate_demo --split validate --limit 3

ama import mimic-cxr \
  --source local_data/mimic_cxr/raw_validate_demo \
  --out local_data/mimic_cxr/processed_validate_demo --split validate
```

raw 实际复制选中图片和报告，可独立导入。importer 在临时目录执行结构验证后才发布，
并生成中英文数据卡、来源记录和导入计数。图片解码失败、hash 不符、错误关联及已有
输出会报错；自动选择时缺报告章节的病例会记录排除原因。

配置 `ama.json` 中的视觉模型及 `.env` 密钥后运行（替换模型别名）：

```bash
ama run local_data/mimic_cxr/processed_validate_demo \
  --model YOUR_VISION_MODEL_ALIAS --split validate \
  --instruction-file datasets/mimic_cxr/instructions.txt \
  --runs-root local_data/mimic_cxr/runs --max-calls 1 --timeout 120

ama eval local_data/mimic_cxr/runs/<运行目录>
```

本目录提示词为第三轮实际使用的 v3，通过 `--instruction-file` 替换 importer 默认指令。
真实图片、报告和逐例运行记录应保留在忽略的 `local_data/`，不随代码提交。
