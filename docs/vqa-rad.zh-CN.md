# VQA-RAD：AutoMedBench 到 AMA 的迁移

[完整技术说明及固定版本官方代码链接](vqa-rad.md)

本任务只迁移数据与 VQA 任务，采用 AMA 运行及评分接口；没有复刻 S1–S5、Docker
或完整自主研究流程，没有重复实现 PathVQA，也没有修改 MIMIC-CXR 分支。

## 版本与官方语义

- 数据：`flaviagiammarino/vqa-rad`，固定 revision
  `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`；该版本 HF 数据卡标记 CC0-1.0。
  适配的官方任务代码提交许可为仅限非商业研究，与 main 的 MIT 许可不同；
  原文保存在 `docs/licenses/AutoMedBench-LICENSE.txt`。
- 官方代码：AutoMedBench `eval_vqa` 分支提交
  `5a19ff72491df7f8c0279438fcfade9364f9a8b7`。
- 已读任务配置、`stage_hf_datasets.py`、task loader、runner、答案规范化及评分代码。
  `stage_data.py` 处理的是 MedXpertQA，不把它的 seed/划分误用到 VQA-RAD。
- 官方 VQA-RAD staging 读取 test，默认全部 451 行；限量时 seed 默认 42，先抽样
  行号再排序。官方加载未指定 HF revision，因此本迁移选择固定版本，不声称还原了
  历史排行榜运行使用的源文件。
- 官方运行可进一步按显式 ID、smoke/calibration ID 文件或 `sample_limit` 选病例；
  staging 全量不意味着每次官方运行都评了全部 test。本迁移的 smoke 单独定义。
- 参考答案只去首尾空白。官方题型由原答案去首尾空白并转小写后是否恰为 yes/no
  决定，不是这份源数据中另外提供的题型标注。

## 架构和数据流

1. `scripts/download_vqa_rad.py` 下载固定版本至新目录，对照上游 LFS SHA-256 或
   Git blob 校验源文件，不覆盖已有目录。
2. `src/ama/importers/vqa_rad.py` 核对本地 manifest、划分和文件校验值，批量读取
   Parquet。每条图像—问题—答案记录生成一个单轮 Episode。
3. `episodes.jsonl` 只含图像和问题两条 Evidence；答案与由答案推导的题型仅写入
   `targets.jsonl`。图像去元数据后保存为 PNG，并按源图像字节哈希去重。
4. 沿用 AMA run loader 与 runner：运行时不读取 targets、eval、provenance 或
   import report；生成 decisions 后，eval 才加载目标与 scorer。
5. `src/ama/importers/vqa_rad_scoring.py` 提供固定规范化和确定性评分；注册不改
   runner 架构，也不要求其他 importer 安装 PyArrow/Pillow。
6. `scripts/verify_vqa_rad.py` 独立回读源数据，逐条核对问题、参考答案、ID、划分、
   顺序和图像像素，验证所有生成文件哈希，并检查 oracle=1、空预测=0。

模型回答使用 AMA 外层 Decision，任务 answer 为 `{"text": "简短英文答案"}`。
应用层 loader 的答案隔离并非操作系统沙箱；后续不要给被评测模型能读取 targets
的任意文件系统工具。

## 评分规则与差异

Scorer 为 `vqa_rad_automedbench_v1`。规范化顺序与固定官方代码一致：转小写、
删除 ASCII 标点、合并空白；整句匹配 yes/no 同义词；否则逐 token 转数字词和
医学缩写。完整转换表在评分模块中。不额外删除冠词、不做 Unicode 折叠、词干化、
模糊匹配或医学语义判断。比如 `IVC` 展开为 `inferior vena cava`，`two` 变为 `2`，
`left-sided` 变为 `leftsided`，`the lung` 保留冠词。

- 规范化后的参考答案为 yes/no：要求规范化后的整句精确匹配；`yes, a lesion`
  不能因部分 token 重合获得该题分数。
- 其他题：取 `(exact match + token F1)/2`，F1 使用 token 多重集。
- 逐题宏平均，保留缺失、空回答、错误类型、额外 answer 字段和失败轮次的分母，
  此类预测计零。缺失或错误 target、重复决策和未知 episode/turn 则报错。
- 使用 0–1 rate 对象，不做中间四舍五入；空组为 null。
- `by_question_type` 按 staging 的原始 yes/no 定义分组；`yes_no_accuracy` 按
  官方 scorer 的规范化定义分组。比如原答案 `positive` 属于前者的 open，却属于
  后者的 yes/no 分母；这是有意保留的两种官方定义。

迁移差异：完整转换保留 train/test，而官方 VQA-RAD staging 仅 test；ID 用源
split+行号而非抽样后序号；题型保留在 evaluator 一侧；图像用无损 PNG 而非
JPEG quality=92；输出用 AMA text 对象及已有输出/引用质量检查，不搬运官方
raw output placeholder 检测。只实现确定性 heuristic，不实现官方可选 LLM judge
或其主分数，也不搬运流程评分、模型选择和训练。分数不宣称等同官方排行榜或
临床正确率。

## 本地复现与存储

环境、下载、全量/子集导入和验证命令见[英文说明](vqa-rad.md#reproduce-locally)。
需要安装 `.[dev,vqa-rad]`；迁移验收测试显式排除 `api`；随后另行授权的真实模型实验见文末链接。

- 全量：train 后接 test，各 split 内按 shard 文件名、Parquet 源行顺序。
- smoke：独立目录，仅从 test 用
  `sorted(random.Random(42).sample(range(N), min(8, N)))` 选择 8 条；train 为空。
- 本地 report 保存具体行号清单、源文件哈希、选择规则、seed、顺序和环境版本；
  provenance 保存逐例源映射，不仅记录 seed。
- 同一源数据、输出目录 basename 和依赖环境下可重建相同生成文件；生成数据卡
  包含目录名，report 包含运行时版本，因此跨环境的目录整体哈希可能不同。
- 源 snapshot 整体哈希按文件名→SHA-256 的 JSON map 规范化计算，不含下载时间。
  `checksums.json` 同样以规范化 map 记录全部生成文件（除自身）的哈希。
- 全量目录应显式选 `test` 评测留出 QA；源 train/test 共享图像，不是患者独立或
  图像独立划分。smoke 只检查管线，不用于宣称性能。
- 真实图像、问答、病例/源行 ID、逐例 provenance、运行及环境记录均留在本地
  忽略的 `data/` 下；demo 目录只有说明文档。Git 仅保留代码、说明和整体校验摘要。

已实际执行的验证见[验证记录](vqa-rad-validation.md)。FakeModel 端到端测试和
真实数据 oracle sanity 检查只能证明管线工作，不能代替模型性能实验。
本任务未修改 NAS、挂载、远端环境、SSH 授权或 tmux。

固定 test100 的三模型评测协议及运行命令见[实验说明](vqa-rad-benchmark.md)。

已完成同一自选 test100 队列的 GLM / Qwen / DeepSeek 真实 API 评测，
[实际结果与限制](vqa-rad-benchmark-results.md)单独报告，不等同官方完整评测。
