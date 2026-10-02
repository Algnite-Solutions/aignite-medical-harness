# MIMIC-CXR 首轮可执行协议与基线核对

状态：阶段 A–C 固定为 onboarding 派生任务 v1；公开基线与评分口径尚未确定。
离线转换及三例真实模型流程已验证，详见验证记录。以下固定协议描述当前实现；
后半部分为尚未采用的公开基线候选，不代表本次已完成数值复现。

## 首轮固定协议（v1）

- 输入：当前 study 的全部 JPG；仅已知 ViewPosition 写入各图片 Evidence.text，缺失省略。
- Episode ID 为 `mimic-cxr-s<study_id>`；唯一一轮 `t1`；study_id 数值升序，图片 dicom_id 字典序。
- Evidence ID 为 `image-<dicom_id>`；file 是包内安全相对路径，实际复制，不使用源目录链接。
- 答案仅为 targets 中的 findings/impression；报告正文、标签、既往材料不进入 episodes 或 instructions。
- 采集 StudyDate/StudyTime 仅写 provenance，不表示报告 available_at。
- 官方 train/validate/test split 保留，不随机重分。首轮 validate 三例固定选择最早合格单图、最早合格多图，再按 study_id 补齐并排序；不按模型效果选择。
- 主样本必须 findings、impression 都非空。缺 section/报告记录排除来源与原因；选中缺图、坏图失败，显式 ID 缺材料报错，不换病例。
- 引用只指向已展示影像；不要求引用全部图片，不将引用覆盖率解释成临床正确性。
- eval 使用 unscored：只验证转换结构，无报告质量分数，尚未满足最终 onboarding 数值验收。

### Section parser 来源核对与差异

已检查 [MIT-LCP 官方 parser](https://github.com/MIT-LCP/mimic-cxr/blob/31901464793d0bb23835092f05bd1aba6af7baff/txt/section_parser.py)
及 [MIT 许可证](https://github.com/MIT-LCP/mimic-cxr/blob/31901464793d0bb23835092f05bd1aba6af7baff/LICENSE)，
固定参考 revision `31901464793d0bb23835092f05bd1aba6af7baff`。本轮独立实现 `explicit-sections-v1`，没有复制官方代码。
官方实现包含大写标题匹配、别名归一化、last_paragraph 以及病例特定规则。本轮仅接受行首明确的 findings/impression
标题（大小写不敏感，标题可另起一行，冒号可省略）；其他带冒号标题终止当前 section。
重复同名 section 按原顺序拼接，压缩空白；不接受别名，不应用病例特例，不以全文或末段填补缺失。
这是可审计的首轮选择规则，不等价于官方 parser 或论文测试集。合成测试覆盖边界。

### 输出与故障规则

raw 包含 manifest.jsonl、images/、reports/、selection_report.json；报告及个案来源仅在 Git 忽略的 local_data。
流式读取 gzip CSV，严格检查关联和重复 ID；ZIP 仅索引定位报告，不全量解压；全 split 解析报告以记录资格计数，
只复制选中的图片。图片实际 load 解码且复制前后 SHA256 对账。Pillow 是 mimic-cxr 可选依赖。
raw 和 AMA 输出均在同父目录 staging 中构建，失败清理临时目录；已有输出拒绝覆盖。
AMA 在 staging 显式调用完整 validate_dataset，成功后才发布；运行器不读取 targets。


## 1. 候选任务

- 一个 Episode 对应一个当前 study，一个 Turn。
- 同一 study 的多张 JPG 放进同一轮，各自有独立 Evidence ID。
- 当前 study 的 findings/impression 作为参考答案，只供 evaluator 读取。
- CheXpert/NegBio 标签不作为可见输入。
- 输出候选结构：`answer = {"findings": "...", "impression": "..."}`。
- 开发和交互检查先使用 validate 子集；正式评测使用与基线一致的 test 子集。
- 当前影像输入和允许的临床文本仍需最终确认。

## 2. CXRMate-2 代码中已确认的事实

| 项目 | 公开实现行为 | 对 AMA 的影响 |
|---|---|---|
| 样本单位 | prepare_mimic_cxr_jpg.py 按 study_id 聚合 dicom_id 和视角 | study 分组与当前候选方案相容 |
| 影像收集 | 准备脚本读取聚合后的全部 dicom_id 对应 JPG | 不能只复制第一张图；processor 是否另行筛选或排序待核对 |
| split | 按官方 split CSV 关联 study | 不能自行随机拆分后比较官方成绩 |
| 报告 | 调用 create_section_files 提取 section，再压缩空白 | section parser 的具体规则和修订版本待核对 |
| 参考缺失 | sft_public.yaml 使用 findings_and_impression_strategy: and；stages 中排除缺任一 section 的 study | 评分病例数不应直接假定为完整 3,269-study test |
| 历史材料 | 公开配置 history: 1；dataset.py 添加最近一个 prior study 的影像与 findings/impression（若存在） | current-images-only 实验输入与该配置不同，不能直接声称数值复现 |
| 输出 | 解码为 findings 和 impression 两部分 | 两部分应分别保留 |
| 评测 | stages 为 findings 和 impression 分别初始化指标，包括 CheXbert、RadGraph-XL、GREEN 等 | 不应自行拼接 section 或替换为 ROCO token overlap |
| 推理入口 | README 示例使用 Transformers 自定义模型/processor 和 CUDA | AMA 当前 Chat Completions 客户端不能假定能直接调用该 checkpoint |

出处：

- https://github.com/aehrc/cxrmate-2/blob/main/prepare_datasets/prepare_mimic_cxr_jpg.py
- https://github.com/aehrc/cxrmate-2/blob/main/config/sft_public.yaml
- https://github.com/aehrc/cxrmate-2/blob/main/dataset.py
- https://github.com/aehrc/cxrmate-2/blob/main/stages_cxrmate2.py
- https://github.com/aehrc/cxrmate-2#download-model

以上是公开代码行为，不证明论文的每个表格、公开 checkpoint 或作者提供的预测
都使用了完全相同的运行配置。必须进一步对照论文、实际模型配置和预测 study 列表。

## 3. 实施前必须解决的问题

1. GLM 视觉 API 已用于流程验证；复现公开 checkpoint 所需算力仍待确认。
2. 基线输入：复现 CXRMate-2 的 history=1，还是选择另一个 current-only 基线？
3. 是否输入 indication/history/comparison/technique：待核对 processor 与调用参数。
4. 样本选择：选定官方 section parser 后，对账实际 test study 集合与排除原因。
5. 指标：确定论文表格对应的 section、指标变体、权重、平均方式和失败处理。
6. 模型：固定 checkpoint/revision、图像处理与排序、生成配置和推理适配方式。

评分验证分两次：同一预测/参考验证 evaluator 一致性；同模型同配置验证端到端
推理结果。使用作者公开预测只能验证第一项，不代表已跑通 AMA 模型实验。

## 4. 本检查点完成标准

当前 importer 与三例流程验证已经完成。后续需选定可执行的基线协议；
上述未知项不阻止交付 importer，但阻止宣称数值复现或最终 onboarding 完成。
