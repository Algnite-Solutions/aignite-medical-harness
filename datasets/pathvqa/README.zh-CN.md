# PathVQA AMA 数据卡

本导入器使用固定 revision 的 PathVQA Parquet 快照。13 个数据分片的名称、大小和 SHA-256 由仓库内的 [来源锁定清单](source.lock.json)约束，不能只修改下载目录里的 manifest 来改变预期校验值。每条原始记录对应一个单轮 Episode，图片作为 Evidence；参考答案仅写入 `targets.jsonl`，AMA 运行时不读取它。保留原始 train、validation 和 test 划分，完整数据为 32,632 例、4,288 张去重图片。

从 test 固定选取的 500 例，以及其中的 [100 例基线](pr100.v2.lock.json)是本地 AMA 适配集合，不是经过验证的 AutoMedBench 官方 cohort。100 例含二元题 49 例、开放题 51 例；v2 锁定 ID 顺序、题目、图像内容、参考答案、任务说明和实际 system prompt。真实图片、答案、完整 AMA 数据和逐题日志留在 NAS 的 `incoming/Automedbench/PathVQA`。PR 审核通过后，由人工核验并发布到 `/mnt/nas/data/processed`。

`pathvqa_v1` 将缺失、失败和无效答案计入分母，分别报告完成率、有效答案率、二元题 yes/no 准确率、开放题规范化 exact match 与逐题 token F1 的平均值；F1 使用词频交集计算。AMA 外层严格 JSON 格式合规率单独统计。词面重合不能证明临床正确性。开启 thinking 的原始回复会保留；AMA 能恢复答案时仍可评分，外层格式是否严格合规则按原始回复计算。

三种请求配置见 [models.v2.json](models.v2.json)：Qwen3.6-27B 开启 thinking、关闭 thinking，以及 GLM-5.3-Flash 默认设置。正式评测统一使用温度 0、8192 输出 token 上限、300 秒超时、每题最多一次模型调用、不使用工具；每种配置先做独立单图冒烟测试。仅网络故障和 HTTP 408、429、5xx 最多尝试 3 次；已有回复不会重生成。运行和复算命令见 [英文数据卡](README.md#reproduce-the-v2-baseline)。输出上限一致不代表不同服务耗费的计算量相同。

统一 [v2 汇总](baseline.v2.json)包含两个模型、三种配置：

| 配置 | 可恢复完成数 | 有效答案 | 二元题正确数 | 开放题 EM | 开放题 token F1 | 严格格式 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen thinking 关闭 | 99/100 | 98/100 | 30/49 | 0/51 | 0.0318 | 21% |
| Qwen thinking 开启 | 88/100 | 87/100 | 31/49 | 0/51 | 0.0634 | 0% |
| GLM 默认设置 | 96/100 | 94/100 | 36/49 | 1/51 | 0.1009 | 95% |

Qwen 关闭/开启 thinking 分别有 1/12 例终态 `invalid_decision`，以及 3/1 次网络重试；GLM 有 4 例终态 HTTP 400，无重试。有效答案指规范化后非空的字符串，二元题合法选项还需严格为 `yes` 或 `no`。所有失败均保留在对应分母中。三次运行的代码哈希存在差异，汇总中保留了这些版本信息。

[旧基线汇总](baseline.json)使用了不同超时与输出上限，仅供历史参考，不与统一 v2 结果混合。
