# rocov2_demo——数据卡

## 来源与许可

来源：ROCOv2 radiology (https://github.com/sctg-development/ROCOv2-radiology)

许可：ROCOv2: CC BY-NC-SA 4.0; each image retains its source attribution

## 研究用途

用于单轮放射影像描述与 UMLS 概念预测研究。

## 构造方法

每张选定图像转换为一个 Episode、一个 Turn 和一条图像 Evidence；caption 与 CUI 仅保存在 evaluator target 中。

## 规模

- Episode：3
- Turn：3
- Evidence：3

## 评分与 targets

Scorer：`rocov2_v0`。Target 记录数：3。

## Artifacts

JPEG 图像复制到 artifacts/；逐图 PMC 链接与署名记录在 Episode metadata 和 import_report.json 中。

## 限制与报告口径

该派生子集的结果不是官方 ROCOv2 benchmark 成绩；caption 仅有单一参考，token 重合度不能代表临床正确性。

## 收录图像来源

| 图像 | PMCID | 署名 | 来源 |
|---|---|---|---|
| ROCOv2_2023_test_000001 | PMC8762516 | CC BY-NC Al Mulhim et al. (2022) | [论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8762516) |
| ROCOv2_2023_test_000009 | PMC9198419 | CC BY-NC Trowbridge et al. (2022) | [论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9198419) |
| ROCOv2_2023_test_000015 | PMC8931810 | CC BY Hishikawa et al. (2022) | [论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8931810) |
