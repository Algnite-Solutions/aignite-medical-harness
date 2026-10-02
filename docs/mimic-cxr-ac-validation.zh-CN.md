# MIMIC-CXR 导入与交互验证记录

2026-10-02；分支 `feature/mimic-cxr-importer`。下方离线验收记录 A–C 阶段结果；
后续模型运行单列记录。真实材料和逐例 trace 仅在忽略的 local_data。

## 实际结果

- 新增合成测试：`33 passed`；完整测试：`83 passed`。
- 实际执行 raw 准备脚本和 AMA import 的 `--help`。
- 实际 raw：`local_data/mimic_cxr/raw_validate_3/`。
- 实际 AMA：`local_data/mimic_cxr/processed_validate_3/`。
- 最终代码另行仅用 raw 重导入 `processed_validate_3_standalone/`，validator 返回 `[]`；episodes/targets/instructions/eval 与首次输出逐字节一致。
- 全量 split CSV 377,110 个影像 ID；官方 validate 1,808 个 study。
- explicit-sections-v1 合格 932；排除 876：缺 findings 578，缺 impression 206，两者都缺 92。
- ID 过滤排除 0；资格过滤后因 limit 未选 929；选中 3 study（1 单图、2 多图），7 JPG。
- 全部选中图像源/复制 SHA256 对账、Pillow load 解码通过；报告 ZIP CRC 和复制 hash 通过。
- AMA 为 3 Episodes、3 Turns、7 条影像 Evidence、3 条 targets；视角缺失 0。
- 完整 `validate_dataset` 返回 `[]`；离线重解析 targets、图片顺序/hash/解码、答案隔离检查通过。
- 首次 raw + processed 总文件字节数 25,219,719（约 24.1 MiB），含本地报告与排除来源记录。
- `git diff --check` 通过；`git check-ignore` 确认真实 manifest 和 targets 被忽略。

选择先取最早合格单图、最早合格多图，再按 study_id 数值顺序补齐，最终排序；
固定个案 ID 存在本地 `selection_report.json`，未按模型效果选样本。
上述排除数仅针对本轮严格 parser，不是官方 parser/公开基线的样本数。

## 函数、数据流与失败处理

`mimic_cxr_raw.csv_rows` 流式读 gzip CSV；`prepare_raw` 建立 dicom→subject/study/split 关联，
严格检查重复和错误关联，读取 ZIP 目录并检查目标 split 报告。它解析资格，不复制全量影像。
只复制选中图片与报告到 standalone raw；采集时间、相对源定位和 hash 保存在本地 manifest。
选中材料错误会终止，不以其他病例替代；自动选择的缺报告/section 保留排除来源和计数。
显式 ID 未知、跨 split、重复或缺材料报错；raw 的 limit 小于显式 ID 数量也报错。

`mimic_cxr.parse_sections` 是独立实现的严格 section parser：仅明确标题，大小写不敏感，
支持换行标题、重复 section 顺序拼接、压缩空白；未知冒号标题终止当前 section。
不使用官方别名、病例特例或全文/末段回退。参考官方 revision 和许可证核对见协议。

`import_mimic_cxr` 只读 raw；以 study 为 Episode，以 dicom_id 排序全部图片；
可见字段仅 image ID、包内图片路径和已有视角。报告仅转换为 targets，时间和来源仅进 provenance。
`safe_path` 拒绝绝对路径、父目录逃逸、符号链接、缺失文件；`decode_image` 实际解码 JPEG；
`digest` 流式 SHA256；损坏或 hash 不符直接失败。

`staging` 在输出同一文件系统的临时目录构建，异常时自动清理，已有输出拒绝覆盖。
AMA 产物在临时目录显式执行完整 `validate_dataset`，返回错误则不发布；通过才 rename 到最终目录。
完整 validator 是结构校验，不是临床/语义验证，因此另外验证图片解码、hash 和 section 条件。
核心运行器、模型、agent 和 recorder 未修改。CLI 仅增加类型分发与 `--study-id`；ROCOv2 默认 test 保持。

## 可重跑命令

下列命令中的输出使用新目录，因为已有目录会拒绝覆盖。再次运行时改为另一个新名称；不要删除既有结果。
依赖安装只在新环境需要：`.venv/bin/python -m pip install -e '.[dev,mimic-cxr]'`。

```bash
.venv/bin/python scripts/prepare_mimic_cxr_raw.py --help
.venv/bin/ama import mimic-cxr --help
.venv/bin/python -m pytest tests/test_mimic_cxr.py -q
.venv/bin/python -m pytest -q

.venv/bin/python scripts/prepare_mimic_cxr_raw.py \
  --source /Volumes/data/incoming/mimic-cxr-jpg-2.1.0 \
  --out local_data/mimic_cxr/raw_validate_3_rerun \
  --split validate --limit 3

.venv/bin/ama import mimic-cxr \
  --source local_data/mimic_cxr/raw_validate_3_rerun \
  --out local_data/mimic_cxr/processed_validate_3_rerun --split validate

.venv/bin/python - <<'PY'
from pathlib import Path
from ama.data import validate_dataset
errors = validate_dataset(Path('local_data/mimic_cxr/processed_validate_3_rerun'))
print(errors)
assert not errors
PY
```

重跑固定个案（避免复制 ID 到公开文档）：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from ama.importers.mimic_cxr_raw import prepare_raw
saved = json.loads(Path('local_data/mimic_cxr/raw_validate_3/selection_report.json').read_text())
report = prepare_raw(Path('/Volumes/data/incoming/mimic-cxr-jpg-2.1.0'),
                     Path('local_data/mimic_cxr/raw_validate_3_fixed_rerun'),
                     split='validate', limit=3, study_ids=saved['selected_study_ids'])
print({k: report[k] for k in ('n_selected', 'n_images')})
PY
```

固定 ID 重跑的资格统计只覆盖所选 ID，与第一次对整个 validate 的资格统计范围不同。
standalone 复验无需 SMB，只需原 raw 和一个新输出目录：

```bash
.venv/bin/ama import mimic-cxr \
  --source local_data/mimic_cxr/raw_validate_3 \
  --out local_data/mimic_cxr/processed_validate_3_standalone_rerun --split validate
```

## 离线阶段边界

上述离线检查证明数据转换和结构有效性，不是报告质量评估。

## 后续真实模型运行

同一组 3 个 validate study（共 7 张 JPG）调用 `glm-5.3-flash`，temperature=1，
每例 max-calls=1，timeout=120 秒。运行通过个人配置的 GLM Coding endpoint；
密钥不在仓库中。每轮保留原始失败，没有为获得成功结果静默重试。

| 提示词 | 格式通过 | 失败原因 | API 返回的总 token 数 |
|---|---:|---|---:|
| 导入器默认指令 | 2/3 | 一例缺少 Decision 外层结构 | 57,768 |
| v2：明确 Decision 外层结构 | 2/3 | 一例 JSON 字符串含未转义换行 | 59,271 |
| v3：增加单段落与换行要求 | 3/3 | 本轮未发生 | 58,782 |

v3 的原文保存为 `examples/mimic_cxr_instructions.txt`；该文件与成功运行 manifest
中的 instruction 逐字一致。运行命令见使用说明，必须显式指定 `--instruction-file`。
这三轮结果是提示词调试记录，不能证明长期稳定性，也不能合并挑选成一次无失败实验。
`ama eval` 仍使用 `unscored`，aggregate 为 null；不存在报告质量或临床正确率结论。

## 提交前检查

2026-10-02：完整离线测试 83 项通过，`git diff --check` 通过。
测试数据全部合成，真实图片、targets 和 events 均由 `local_data/` 规则忽略。
开发依赖显式包含 Pillow，以覆盖新增测试的图像生成与解码。

正式报告指标、独立评估样本和公开基线数值复现仍待完成。
