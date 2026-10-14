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

上述 raw importer 的 scorer 仍为 `unscored`。带报告质量评分的 ReXrank 子集使用下文
单独的准备入口；两条路线的参考报告构造不同，不能仅覆盖 scorer 名称混用。

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

本目录提示词现只说明任务和 answer 内部字段，外层 Decision 由 runner 统一规定。
上述三轮历史格式结果不代表新提示词已经完成模型验证。
真实图片、报告和逐例运行记录应保留在忽略的 `local_data/`，不随代码提交。

## ReXrank 100 例与 RadGraph 评分

从 [ReXrank 固定版本](https://github.com/rajpurkarlab/ReXrank/tree/0a18a41e580992ac7cd32a8a9463009654f0d390)
的 `data/mimic-cxr/ReXRank_MIMICCXR_test.json` 抽样。先按 study_id 数值排序，再以 seed=42
无放回选 100 例，输出再排序；共 164 张图。保留官方 `report` 作为 evaluator-only reference，
不经 raw importer 的 section parser。官方缺失 impression 的 NaN 转为空字符串，仅影响
辅助 answer 字段，不改 reference。输入为当前 study 全部图片和视角，不含 context/history。

```bash
python3 -m pip install -e '.[dev,mimic-cxr,cxr-radgraph]'
export AMA_DATA_PATH=local_data
export REXRANK_JSON=/path/to/ReXRank_MIMICCXR_test.json
export IMAGE_ROOT=/path/to/mimic-cxr-jpg-2.1.0
bash scripts/run_mimic_cxr.sh freeze
bash scripts/run_mimic_cxr.sh import
bash scripts/run_mimic_cxr.sh setup
bash scripts/run_mimic_cxr.sh validate
```

`IMAGE_ROOT` 须包含 `files/` 与官方 split CSV。完整可运行数据位于
`$AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42/dataset/`，相邻 `cohort/` 保存
固定清单。可覆盖 `BENCHMARK_DIR`；准备后 run/eval 不再读取源 JSON 或 NAS 图片。
`freeze/import` 拒绝覆盖已有输出。

```bash
bash scripts/run_mimic_cxr.sh --help
bash scripts/run_mimic_cxr.sh run YOUR_VISION_MODEL_ALIAS
bash scripts/run_mimic_cxr.sh eval runs/mimic-cxr-rexrank-fixed100-seed42/RUN_ID
# 也可直接使用原有 CLI：
ama run "$AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42/dataset" \
  --model YOUR_VISION_MODEL_ALIAS --split test --max-calls 1 --timeout 300
ama eval runs/RUN_ID
```

脚本复用 AMA runner，不另定义外层格式。当前 run 入口没有批量续跑/重试能力。
运行记录默认放本地 `runs/mimic-cxr-rexrank-fixed100-seed42`，不进入 processed 数据集。

`mimic_cxr_radgraph_v1` 按 [ReXrank-metric 的定义](https://github.com/rajpurkarlab/ReXrank-metric/tree/d6be67831ef933dc91f01892303a40ebc6cc9549)
计算实体集合 F1、关系集合 F1，再取平均，最后逐 study 宏平均。无有效报告的病例记 0，
保留在期望例数分母；有效报告均分、失败数、输出格式质量分开报告。抽取器故障直接报错。
使用固定权重的 `radgraph==0.1.18`、`model_type=radgraph`，本地 CPU 离线推理；setup 的下载
可设置 `HF_ENDPOINT`，下载仍校验固定 hash。自定义缓存时设置 `AMA_RADGRAPH_CACHE`。

这是 image-only 的 100 例 onboarding 子集；同一 annotations 上的官方评分函数对账通过，
新版包与 legacy 抽取环境等价性未验证。分数不是完整官方榜单成绩，也不是临床诊断准确率。
这里只提交代码、数据卡、任务指令和 synthetic 测试；真实数据、逐例结果、计划和审计文件留本地。
