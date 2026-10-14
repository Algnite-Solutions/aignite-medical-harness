# MIMIC-CXR ReXrank–RadGraph onboarding v1

## 协议与验证范围

这是 ReXrank MIMIC-CXR 的 100-study onboarding 子集，输入为当前 study 全部 JPG 和官方视角；不提供 context 或历史影像。结果不能直接与完整 ReXrank 榜单比较。

ReXrank 源 revision：`0a18a41e580992ac7cd32a8a9463009654f0d390`。
文件：`data/mimic-cxr/ReXRank_MIMICCXR_test.json`。
SHA256：`39fe05a446cebd7af4507681cc423b04409eaf8d205ad40b6a043168a14059df`。

官方实际 JSON 是 study_id 到 record 的映射。完整清单有 2,347 个 test study、3,858 张图；正文键为 `report`，不是 README 所写的 `reports`。723 条 impression 是 JSON 的非标准 `NaN`。保留全部病例，辅助 answer 字段中的 NaN 转为空字符串，**评分 reference 始终使用官方 report 原文**，不通过现有 section parser 重写。

先将 study_id 按数值排序，再用 Python `random.Random(42).sample(..., 100)` 抽样，结果再次按数值排序。固定结果为 100 个 study、164 张图，排除 0 例。实际 ID 与图片 hash 留在本地 cohort/provenance，不提交真实病例材料。

一个 study 一轮，全部影像按路径稳定排序。生成英文 `answer.findings` / `answer.impression`，评分 prediction 拼接为 `Findings: ... Impression: ...`。允许一个 section 为空，但两个字段都必须是字符串且至少一个非空；否则记为无效报告。引用格式质量由 harness 单独报告，不作为临床图 F1。

## 评分定义

采用 ReXrank-metric 的 **RadGraph Combined** 计算定义：

- 节点集合：`(tokens, label)`；label 包含阳性/阴性/不确定等抽取标签。
- 边集合：`(源节点, 目标节点, relation_type)`。
- 分别计算节点集合与边集合的 F1，再取二者平均。
- F1 = `2 × 交集数量 / (参考集合数量 + 预测集合数量)`。按官方代码约定，两集合均空时 F1 为 0。
- 最终对所有期望 study 作宏平均。缺失/失败/无效报告记 0，同时报告有效报告均分和失败数。此失败规则是本地 v1 约定。

这与新版 `F1RadGraph(reward_level=...)` 的多个语义奖励变体不同，不直接调用其默认值。

公式来源 revision：`d6be67831ef933dc91f01892303a40ebc6cc9549`，具体文件：
`scripts/CXR-Report-Metric/CXRMetric/radgraph_evaluate_model.py` 与 `run_eval.py`。

推理采用官方 `radgraph==0.1.18` 的 `model_type="radgraph"`，CPU，batch size 1。权重仓库 revision 为 `6646433b3ad83a10f6e141db76d0ece44312b236`。固定 archive、weights、config、tokenizer/config 的校验值；详见代码常量。encoder/config revision 为 `e1354b7a3a09615f6aba48dfad4b7a613eef7062`。

官方 ReXrank legacy evaluator 使用旧依赖环境。本次验证的是**新版官方抽取包产出同一批 annotations 后，AMA 与官方实体/关系评分函数一致**。尚未证明新版与 legacy 抽取程序逐项等价，也没有复现任何公开模型的榜单成绩。metadata 中保留 `legacy_extractor_equivalence: not_verified`。

## 架构、数据流与失败处理

1. `mimic_cxr_rexrank.load_official/select_records/freeze_cohort`：验证固定源文件 hash、全量路径/参考结构，再冻结样本。不按模型效果选病例。
2. `materialize`：核对官方 split CSV 中的 subject/study/dicom/test 关联，从只读 source 复制选定 JPG，解码、核对复制 hash，再生成 AMA dataset。targets/provenance 不进入推理输入。staging 验证成功才发布，禁止覆盖既有输出。
3. 原有 runner/Decision 无需改动。新数据集的 `eval.json` 选择 `mimic_cxr_radgraph_v1`。
4. `cxr_radgraph.Extractor`：按需加载可选依赖，先验证固定模型资源；通过离线 HF snapshot 让官方 wrapper 读取固定 tokenizer/config，再执行实体与关系抽取。初始化将当前进程的 HF cache/offline 环境指向已验证缓存，评测宜独立进程运行。
5. `score_mimic_cxr_radgraph`：读取 evaluator-only reference 与 Decision，批量抽取，计算逐例及聚合分数。评分基础设施失败直接报错，不能当作模型得 0 分。
6. `verify_cxr_radgraph.py`：验证官方源码 hash，只加载已核验的两个纯计算/解析函数，避免安装完整 legacy 评测栈。保存 annotations 和对账结果供离线审查。

其他关键失败：源/权重/hash 改变、路径逃逸、重复图、split 不匹配、缺图/损坏图、cohort 被改动、重复 Decision、缺失 authoritative reference 均明确失败；不静默换病例，不回退到词重合评分。

## 重跑准备

从项目根目录执行，Python 3.11+；当前实际验证环境为 macOS arm64、Python 3.13.12、torch 2.14.1、transformers 4.51.3。

```bash
.venv/bin/python -m pip install -e '.[dev,mimic-cxr,cxr-radgraph]'
mkdir -p local_data/mimic_cxr/rexrank_v1
curl -L --fail 'https://api.github.com/repos/rajpurkarlab/ReXrank/contents/data/mimic-cxr/ReXRank_MIMICCXR_test.json?ref=0a18a41e580992ac7cd32a8a9463009654f0d390' \
  -H 'Accept: application/vnd.github.raw+json' \
  -o local_data/mimic_cxr/rexrank_v1/ReXRank_MIMICCXR_test.json
.venv/bin/python scripts/prepare_mimic_cxr_rexrank.py freeze \
  --source local_data/mimic_cxr/rexrank_v1/ReXRank_MIMICCXR_test.json \
  --out local_data/mimic_cxr/rexrank_v1/frozen_100
.venv/bin/python scripts/prepare_mimic_cxr_rexrank.py materialize \
  --source local_data/mimic_cxr/rexrank_v1/ReXRank_MIMICCXR_test.json \
  --cohort local_data/mimic_cxr/rexrank_v1/frozen_100/cohort.json \
  --image-root /Volumes/nas-data/incoming/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/rexrank_v1/dataset_verified
.venv/bin/python scripts/setup_cxr_radgraph.py
```

`image-root` 必须包含 `files/` 和官方 split CSV。当前本地没有 NAS 挂载；以上 materialize 的真实验证在实验室 SSH 主机完成，数据包取回本地。输出若已存在应使用新的目录；不删除已冻结实验来重跑。

网络不能连接 Hugging Face 时，可用 `HF_ENDPOINT=https://hf-mirror.com` 选择镜像。下载文件仍必须通过固定官方 revision/hash 核验。setup 支持 `--archive <预下载的 radgraph.tar.gz>`；默认缓存为 `local_data/radgraph`，自定义缓存须同步设置 `AMA_RADGRAPH_CACHE`。

对账命令：

```bash
curl -L --fail 'https://api.github.com/repos/rajpurkarlab/ReXrank-metric/contents/scripts/CXR-Report-Metric/CXRMetric/radgraph_evaluate_model.py?ref=d6be67831ef933dc91f01892303a40ebc6cc9549' \
  -H 'Accept: application/vnd.github.raw+json' \
  -o local_data/mimic_cxr/rexrank_v1/radgraph_evaluate_model.py
.venv/bin/python scripts/verify_cxr_radgraph.py \
  --pairs local_data/mimic_cxr/rexrank_v1/real_validate_pairs.jsonl \
  --official-code local_data/mimic_cxr/rexrank_v1/radgraph_evaluate_model.py \
  --out local_data/mimic_cxr/rexrank_v1/verification
.venv/bin/python -m pytest -q
```

pairs 为本地 JSONL，每行为 `id/reference/prediction`。现有三例 pairs 来自旧 validate 数据参考和已保存的 GLM 输出；reference 按固定字段拼接，仅用于验证评分接线，与正式 ReXrank 100 例无重叠。重跑对账使用新的输出目录。

后续真正生成报告后使用 `ama eval <新 run 目录>`；旧 importer 的 `unscored` 数据没有 authoritative `report`，不能仅通过 `--scorer` 覆盖就假定获得 ReXrank 评测。

## 2026-10-15 实际验证

- 远端 NAS：100 study、164 JPG 均完成复制、实际解码、复制 hash 核验；官方 test CSV 关联检查通过；dataset validator 通过。
- 冻结 cohort 在本地与远端按相同源/seed 生成，并对账其文件 hash。
- RadGraph 权重及 tokenizer/config 校验通过，CPU 抽取实际运行成功。
- 三例真实报告对账最大绝对差 `5.551115123125783e-17`，容差 `1e-12`。
- 实际 scorer adapter 与上述三例逐例分数再对账。
- 离线测试：156 passed。测试使用 synthetic 数据；无真实病例 fixture 入库。
- 没有新增模型 API 调用，没有完成 100 例模型性能实验，没有执行 commit/push。

本地材料：`local_data/mimic_cxr/rexrank_v1/`（源清单、cohort、dataset_verified、真实 pairs、verification、adapter-validation.json、环境记录），均由 Git 忽略。远端 task 输出在 `/tmp/ama-cxr-step12/`，不是持久存储交付位置；源 NAS 未修改。

## 官方依据

- [ReXrank 固定版本](https://github.com/rajpurkarlab/ReXrank/tree/0a18a41e580992ac7cd32a8a9463009654f0d390)
- [ReXrank-metric 固定版本](https://github.com/rajpurkarlab/ReXrank-metric/tree/d6be67831ef933dc91f01892303a40ebc6cc9549)
- [RadGraph 官方包](https://github.com/Stanford-AIMI/radgraph)
- [官方权重仓库固定版本](https://huggingface.co/StanfordAIMI/RRG_scorers/tree/6646433b3ad83a10f6e141db76d0ece44312b236)
