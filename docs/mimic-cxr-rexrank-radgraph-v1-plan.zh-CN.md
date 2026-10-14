# MIMIC-CXR ReXrank–RadGraph onboarding v1 计划

本文件是实施计划，不代表评测已经完成。目标是尽快交付一套能重复运行、能解释数据流、具有真实报告质量分数的 AMA 任务。

## 当前分支的起点

检查分支：`feature/mimic-cxr-importer`。

- 已有 standalone raw 准备、CXR importer、CLI 接线、图片解码与 hash 检查、答案隔离和相关测试。
- 一个 study 对应一个 episode、一轮；当前 study 全部 JPG 按 dicom_id 排序；答案为英文 findings/impression。
- `eval.json` 仍使用 `unscored`，没有 RadGraph scorer。
- 当前选样按 study_id 排序，并优先覆盖单图/多图；`--limit 100` 不能代表随机抽样。
- 本地已有 validate 三例材料，可用于调试；不能据此声称完成 100 例评测。
- 已有 Symptom2Disease 批量脚本，可参考其冻结样本、断点续跑、失败记录和对比检查，但不能直接套用分类评分。

## v1 的边界

先完成图像到报告生成与离线评分。复用 AMA 的 runner、Decision、日志与 evaluator 接口；不扩展表征检索、历史影像、训练或额外 LLM 裁判。

建议名称：`mimic_cxr_rexrank_radgraph_100_v1`。只有实际使用 ReXrank MIMIC-CXR 官方清单与参考报告后才采用这一名称；否则使用 `mimic_cxr_report_radgraph_100_v1`，并明确为派生任务。

100 例属于 onboarding 子集，不直接与完整 ReXrank 榜单分数比较。

## 阶段 1：冻结协议与样本

1. 固定 ReXrank 仓库 revision，核实 MIMIC-CXR 测试清单的获取方式、study/image 对应关系、参考报告和上下文字段。
2. 对照现有 importer 的严格 section parser。若使用 ReXrank，应以其参考文本为准，不用重新解析结果悄悄替换。检查缺失 findings/impression 是否会改变入选集合；仅在有明确规则时排除并记录。
3. 推荐 v1 输入仅当前 study 全部影像和视角；不使用历史材料。若省略官方提供的 context，明确记录为 image-only 输入变体。
4. 在可用、符合冻结规则的测试 study 中用 seed=42 无放回抽取 100 例；先按稳定 ID 排序再抽样，保存实际 ID、来源 revision、排除数量、图像数量与样本清单 hash。
5. 保留现有 validate 三例作为开发集。正式 test 100 例不用于反复修改 prompt 或挑选样本。
6. 模型输出继续使用 AMA Decision，`answer` 包含 `findings` 和 `impression`；报告正文要求英文，以匹配参考报告和评分器。

验收：数据集通过 validator，推理加载不读取 targets，100 个 study 的影像与参考逐一可追溯。若本地材料不足，不静默用其他 split 或病例替换。

## 阶段 2：先验证 RadGraph，再消耗批量 API

RadGraph 将生成报告与参考报告分别转为医学实体及关系，再计算图结构重合的 F1。它包含学习得到的抽取模型，不是词重合，也不是让被测 LLM 自评。

1. 首先在可用机器上验证官方 RadGraph 包、权重、设备和依赖能运行。可用 CPU 时先小批运行，实际计时后再决定评分机器，不预先承诺运行时长。
2. 核对 ReXrank 官方 metric 实现，固定 package/code revision、抽取模型及权重 revision、reward_level 和输入文本处理；不能把不同 RadGraph 变体统称为同一分数。
3. 默认规划为 findings/impression 按固定格式拼接后评分；若官方选定协议不同，则遵循该协议并写入文档。主指标明确具体 F1 变体及逐 study 宏平均方式。
4. 新增任务专用 scorer，并在现有 registry 注册；重依赖按需加载，未安装时明确报错，不影响其他任务。
5. 在同一批 prediction/reference 上对照官方调用与 AMA adapter 的逐例分数，保存版本和数值差异。合成例覆盖原文相同、空输出、否定词变化、字段缺失；这些用例用于检查行为，不能宣称指标已证明所有临床错误。
6. 缺失、调用失败或无法恢复的报告在 v1 全样本主指标中记 0，保持分母 100；同时单列失败数及有效报告均分。此为本地 v1 规则，不声称是官方榜单规则。
7. 评分器加载或抽取失败属于评测基础设施错误，必须报错并修复，不能算作模型 0 分。

验收：真实小批报告能产生逐例临床图 F1，官方调用与 adapter 对账通过；缓存绑定文本 hash 和评分器版本，修改输入后不会复用旧结果。

## 阶段 3：小批端到端与固定参数

1. 先选一个已确认支持当前 endpoint 多图输入的视觉模型，在 validate 三例上跑通生成与评分。
2. 检查图片是否真实发送、全部视角是否到达、无答案泄漏、输出可恢复，以及 token 上限是否截断报告。
3. 冻结统一任务指令、temperature、输出上限、超时与重试策略。记录准确模型 ID、endpoint、时间和 provider 返回的模型信息；不能仅凭模型族名称判断支持视觉。
4. 严格 JSON 合规率、可恢复报告率与 RadGraph 分数分开。不得用另一个 LLM 修复答案后充当原模型输出。
5. 只对明确定义的传输错误重试并记录；已有模型回答不得因内容不好而重试。核对原始请求图像数量与本地 episode。

验收：一个模型三例完整跑通，能从保存的输出重新离线评分，无须再次调用模型。

## 阶段 4：100 例正式运行与交付

先完成一个模型的 100 例结果，再扩展到第二个及更多模型；模型数量不应阻塞第一个闭环。

- 批量脚本参考 Symptom2Disease 的续跑与结果对齐检查，任务专用处理报告字段和 RadGraph。
- 所有模型共享相同 sample hash、图像输入协议和 prompt。无法支持多图的 endpoint 不静默改成单图加入同一表。
- 对比表包括：准确模型 ID、期望/完成/可评分数量、失败分类、严格格式合规率、全样本 RadGraph 宏平均、有效报告均分、耗时和可获得的 token 用量。
- 本地保留逐例 prediction/reference、评分、原始运行日志和 provenance；仓库提交代码、协议、复跑说明、聚合结果和 synthetic 测试材料。
- 阅读少量高低分及模型分歧样本，记录遗漏、否定、部位、设备描述等错误；明确标记人工观察，不把 RadGraph 分数等同于完整临床正确性。

## 最小代码改动

- 现有 raw/importer：增加冻结清单支持或薄的 ReXrank adapter，复用拷贝、验证、答案隔离和 staging。
- 任务专用评分模块：包装官方 RadGraph，注册到 `src/ama/scorer.py`。
- `pyproject.toml`：按验证结果增加可选评分依赖，不强制所有使用者安装。
- 任务专用 benchmark 脚本：prepare/run/eval/compare，复用现有能力，避免重写核心 runner。
- 新增有意义的协议/评分/续跑测试；运行相关测试和完整离线测试，实际记录结果。

## 完成标准与公开基线

最小端到端完成：冻结数据集 + 一个真实视觉 API 模型完成 100 例 + RadGraph 逐例评分与聚合 + 失败统计 + 可复跑文档与测试。

若 onboarding 另有“复现公开基线数值”要求，还需同一子集、输入协议、权重和生成参数的官方模型/官方预测对账。官方 scorer 一致只证明评分接线正确，不能替代模型复现。多个 API 模型间比较是新实验，不自动构成公开基线复现。

## 执行优先级

先排除两个关键阻塞：ReXrank 清单/参考能否取得并与本地影像关联；RadGraph 在可用环境是否能实际运行。然后三例闭环，再冻结 100 例，最后逐模型扩展。实际耗时主要由这两项环境验证和 API 吞吐决定。

## 官方依据

- ReXrank 数据字段与任务：https://github.com/rajpurkarlab/ReXrank
- ReXrank metric 实现（具体版本和调用方式待实施核对）：https://github.com/rajpurkarlab/ReXrank-metric
- RadGraph 官方包与 F1 调用：https://github.com/Stanford-AIMI/radgraph
