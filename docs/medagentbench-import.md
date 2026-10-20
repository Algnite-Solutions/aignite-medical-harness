# MedAgentBench 本地数据导入

本地 importer、医疗查询工具、模拟提交工具和 `medagentbench` scorer 已实现，均位于 `src/ama/importers/medagentbench.py`。可以通过 `ama run` 运行任务，再通过 `ama eval` 对结构化答案和实际模拟提交记录评分。成绩应标注为 **AMA 对 MedAgentBench 的适配成绩**，简化工具接口与官方 FHIR 接口不同。

Importer 不调用模型、不运行原 grader，也不修改 FHIR。从服务器导入时需要 FHIR 服务；已有快照可完全离线重复转换。运行已转换的数据和评分都不需要 FHIR 或 Docker。使用现有 AMA Python 环境，无需新增第三方依赖。

生成数据放在 `datasets/medagentbench/`；该目录及下面示例的 `datasets/medagentbench_subset/` 已加入 Git 忽略规则。Importer 仍支持通过 `--out` 指定其他位置。

## 使用组内 NAS 数据复现

原始项目：[MedAgentBench 官方仓库](https://github.com/stanfordmlgroup/MedAgentBench)。任务文件在原仓库 `data/medagentbench/`，患者与观察资源由官方 FHIR Docker 镜像提供。组内共享的是经过本 importer 转换的 **AMA 适配版**。

完整 processed 数据位于 NAS 的 `data/processed/MedAgentBench/`。macOS 当前挂载路径如下；其他机器将 `AMA_DATA_PATH` 改为自己的 NAS 数据根目录：

```bash
export AMA_DATA_PATH=/Volumes/10.214.105.104/data
```

其中包含任务、患者、检验数据、评分参考、原始快照及原 grader 副本。已有这份 processed 数据时，不需要下载原仓库、启动 Docker 或重新导入。

### 环境和 API 配置

先激活已有的 AMA Python 环境（Python >=3.11，依赖见 `pyproject.toml`）。脚本默认使用当前 `python`，也可用 `PYTHON_BIN=/path/to/python` 指定现有解释器，无需为本脚本重装环境。

模型别名从 `ama.json` 读取；可用 `CONFIG=/path/to/config.json` 指定个人配置。每个模型配置的 `api_key_env` 字段给出密钥环境变量名称，将对应密钥写入项目根目录 `.env`，或事先 export。CLI 会读取 `.env`；不要把密钥提交 Git。

例如当前 `deepseek-v4-flash` 的 `api_key_env` 为 `MEDAI_API_KEY`，可在本地 `.env` 中填写：

```dotenv
MEDAI_API_KEY=替换为自己的密钥
```

### 一键验证、运行和评分

在项目根目录执行：

```bash
bash scripts/run_medagentbench.sh --help
bash scripts/run_medagentbench.sh validate
bash scripts/run_medagentbench.sh all deepseek-v4-flash
```

`validate` 校验 300 个任务与评分参考、固定 smoke、患者和观察计数、快照哈希，并读取 processed 事实。`all` 先校验已有数据，再运行、评分；它不会重新导入、重新抽样或写入 NAS。实际模型请求只发生在 `run`/`all`。

默认 temperature=0、max_tokens=2048、每例最多 8 次模型调用、请求超时 120 秒。可用 `TEMPERATURE`、`MAX_TOKENS`、`MAX_CALLS`、`TIMEOUT` 覆盖；其他模型参数（包括 thinking 设置）沿用配置。脚本没有自动重试、续跑或请求限速；失败、不提交、未执行题仍计入分母。

默认 split 为固定 `smoke`，需要完整 300 例时显式设置 `SPLIT=test`。可顺序传入多个模型别名；每次调用建立新的 batch，不覆盖旧实验。

### 固定 30 例参考集

直接复用 importer 的 `SMOKE_IDS`，来自原 MedAgentBench `configs/samples/` 的 seed-42 固定样本，十类任务各 3 例。这是 test 的子集，无需复制患者和观察文件。验证会拒绝被替换或重新排列的 smoke：

| 任务 | 固定任务 ID |
|---|---|
| task1 | task1_1, task1_4, task1_21 |
| task2 | task2_8, task2_9, task2_24 |
| task3 | task3_5, task3_8, task3_24 |
| task4 | task4_4, task4_22, task4_24 |
| task5 | task5_3, task5_18, task5_29 |
| task6 | task6_2, task6_14, task6_19 |
| task7 | task7_1, task7_3, task7_7 |
| task8 | task8_8, task8_17, task8_20 |
| task9 | task9_1, task9_7, task9_18 |
| task10 | task10_18, task10_21, task10_23 |

### 输出和重新评分

默认输出到 `runs/medagentbench-smoke/<batch>/model-<序号>/<run-id>/`，可用 `RUNS_DIR` 覆盖输出根目录。脚本自动加载 `src/ama/importers/medagentbench.py` 的工具工厂。

- batch 下的 `validation.json` 保存数据校验值、任务 ID 和资源数量；`code-version.json` 保存代码 commit、工作区修改状态及运行时源代码和脚本哈希。
- 每个 run 下保留 `manifest.json`、`decisions.jsonl`、`messages.json` 和 `diagnostics.jsonl`；`all`/`eval` 额外生成 `metrics.json` 和 `summary.json`。
- `summary.json` 报告总成功率、各任务成功率、完成率、API 错误数和终止状态。默认分母为固定 30 题，每类分母为 3。
- 不能只保留最后分数：MedAgentBench scorer 必须读取工具日志来核验模拟提交。

将下面路径替换为运行输出的实际 run 目录：

```bash
# 离线重新评分，不需要 API key，不请求模型
bash scripts/run_medagentbench.sh eval runs/medagentbench-smoke/<batch>/model-1/<run-id>

# 也可分开运行和评分
bash scripts/run_medagentbench.sh run deepseek-v4-flash
```

重新评分仍需访问 `manifest.json` 中记录的 `dataset_dir`。跨机器迁移时，在实验记录的工作副本里将该字段改为新机器的数据挂载路径，并保留原始记录。

### 数据与复现验证

NAS 数据已通过 AMA 格式校验，包含 300 个任务、695 个患者和 563,426 条观察记录；原始快照校验通过。固定 smoke 为 30 例，十类任务各 3 例。

已通过启动脚本在 NAS 数据上完成一次 30 例评测，并确认保存日志可离线重新评分，结果一致：

| 模型 | 任务成功率 | 完成率 |
|---|---:|---:|
| Qwen3.6-27B（DashScope，关闭 thinking） | 13/30（43.3%） | 29/30（96.7%） |

参数为 temperature=0、max_tokens=2048、max_calls=8、timeout=120 秒。所有 30 题计入分母。上述为 **AMA 适配成绩**。

分任务成功数（各 3 例）：task1–task10 依次为 0、0、2、3、2、2、1、0、1、2。

本次使用个人配置 `CONFIG=runs/medagentbench-qwen.local.json`，模型别名为 `qwen36-nothink`；该本地配置将 `base_url` 设为 `https://dashscope.aliyuncs.com/compatible-mode/v1`、`model` 设为 `qwen3.6-27b`、`api_key_env` 设为 `QWEN_API_KEY`，并设置 `chat_template_kwargs: {"enable_thinking": false}`。复现时在自己的配置中填写这些字段，密钥仍放在 `.env` 或环境变量中。

### 重新导入（可选）

已有 NAS 数据时跳过本步骤。需要重新导入时，指定原仓库、已有完整快照和**尚不存在的新输出目录**：

```bash
SOURCE_DIR=/path/to/MedAgentBench \
SNAPSHOT_DIR="$AMA_DATA_PATH/processed/MedAgentBench/snapshot" \
DATASET_DIR=/path/to/new-medagentbench \
bash scripts/run_medagentbench.sh import
```

也可用 `FHIR_BASE` 代替 `SNAPSHOT_DIR` 从服务器导入；两者互斥。输出目录已存在时拒绝覆盖。命令说明可随时用 `--help` 查看。

## 从 FHIR 导入

准备本地 MedAgentBench 仓库中的 `test_data_v2.json`、`funcs_v1.json`、`refsol.py` 和 `utils.py`。FHIR 服务必须包含原 benchmark 数据；仅启动一个空 FHIR 服务不够。

```bash
PYTHONPATH=src python -m ama.importers.medagentbench \
  --source ../MedAgentBench \
  --fhir-base http://localhost:8080/fhir/ \
  --out datasets/medagentbench
```

可增加 `--id task1_1`（可重复）或 `--limit 10` 选择任务。选择结果保持原始任务顺序；患者和观察资源仍全量导出，以保证患者检索不只包含选中任务的答案。`--timeout` 默认为每次请求 60 秒。

如果使用已有 Docker 镜像，可在独立端口启动新容器，并将 `--fhir-base` 改为实际端口：

```bash
docker run -d --name ama-medagentbench-import \
  -p 127.0.0.1:8081:8080 jyxsu6/medagentbench:latest
```

等待 FHIR 应用启动完成再导入。Importer 全量遍历 Patient 与 Observation 的分页；失败会清理临时输出，不发布半成品。已有输出目录拒绝覆盖，重新导入时选择新目录。

## 从快照离线导入

```bash
PYTHONPATH=src python -m ama.importers.medagentbench \
  --source ../MedAgentBench \
  --snapshot datasets/medagentbench/snapshot \
  --out datasets/medagentbench_subset \
  --id task1_1 --id task2_8
```

`--snapshot` 和显式 `--fhir-base` 互斥。快照目录必须有 `metadata.json`、`Patient.jsonl`、`Observation.jsonl`，且数量和 SHA-256 校验一致。复制后的快照再次验证；转换不执行网络请求。

## 输出文件与隔离

| 文件 | 内容 |
|---|---|
| `dataset.json` | 数据集名称、test 和 smoke splits |
| `episodes.jsonl` | 公开任务 instruction 和非空 context，一个任务一个 t1 |
| `instructions.txt` | AMA 结构化答案与模拟提交约定，不再要求 FINISH 答案列表 |
| `targets.jsonl` | 原始 case_data，包括仅供评分的 sol、eval_MRN |
| `eval.json` | 已注册的 medagentbench scorer 名称 |
| `patients.jsonl` | 患者 ID、原始标识符和姓名结构、可选出生日期与性别 |
| `cases.jsonl` | patient_id 和该患者的全部 observations |
| `snapshot/` | 完整原始 Patient、Observation 资源和快照校验信息 |
| `evaluator/` | 未执行的原 refsol.py、utils.py 和原函数说明副本 |
| `import_report.json` | 源文件摘要、资源数、分页数、选中任务与导入状态 |
| `DATASET_CARD*.md` | 中英文数据集说明 |

运行侧不应读取 targets 或 evaluator。患者 ID 来自 FHIR Patient.id，不是任务 ID，也不是 MIMIC 住院 ID。MRN 保存在 identifiers，不能假设所有服务器的 MRN 与 Patient.id 相同。

`test` 包含本次选中的任务，`smoke` 是选中任务与原 seed-42 固定 30 例样本的交集。默认完整 v2 数据应有 300 个任务、每类 30 个、smoke 每类 3 个。

观察记录包含 source_id、evidence_id、原始 code、可选 label/value，以及存在时的 unit、observed_at、参考范围等。所有 effective[x] 时间结构保留；单位不换算，时间不裁剪，同一项目多次测量不合并。没有数值不等于数值为零；缺失字段不虚构。其他源字段可由 source_id 在原始快照中追溯。

Importer 不计算年龄、最近检验、平均值、替代治疗剂量等任务答案。模型通过工具读取病例；只有评测时 scorer 才计算参考值并核验操作。

## 通过现有 `--tools` 加载

所有新工具及 `make_tools(dataset_dir, episode_id)` 都位于 `src/ama/importers/medagentbench.py`。Importer 命令仍可照常使用；AMA 加载同一个文件时只调用工具工厂，不会再次导入数据。通用工具加载器和 `pyproject.toml` 无需修改，也无需新增依赖。评分入口的少量修改见下面“评分”。

使用已配置的模型，例如：

```bash
PYTHONPATH=src python -m ama.cli run datasets/medagentbench \
  --model qwen36 --episode task3_1 \
  --tools src/ama/importers/medagentbench.py
```

也可用 `ama chat` 交互检查，或将 `--episode task3_1` 换成 `--split smoke` 运行样本集。运行需要配置好模型的 API；工具和评分器自身完全离线，不需要 Docker 或 FHIR 服务。`chat` 属于探索，不产生基准分数。

| 工具 | 输入和用途 |
|---|---|
| `patient_search` | 至少提供 `identifier`（MRN）、`given`、`family`、`birth_date` 之一；返回所有匹配患者的来源事实，不计算年龄 |
| `patient_information` | 提供检索得到的 `patient_id`，读取出生日期、姓名、标识符等 |
| `laboratory_results` | `patient_id`，可选 `code`、`after`、`before`；读取原始观察值，不计算最近值、均值或推荐剂量 |
| `submit_order` | `order` 对象，`order_type` 为 `medication`、`test` 或 `referral`，提交模拟医嘱 |
| `record_observation` | `observation` 对象，提交血压等模拟观察记录 |

姓名匹配忽略大小写，对 given 名字单词和 family 姓氏做精确匹配；同时给出的姓名条件须匹配同一姓名条目。MRN 和出生日期精确匹配，多条件取交集。搜索不到返回空列表，多条匹配全部保留，不替模型选择患者。

两个列表查询支持 `offset` 和 `limit`，每页最多 200 条，返回 `total` 和 `next_offset`；`next_offset` 为 null 才表示没有后续页。检验默认每页 100 条，患者检索默认 50 条。检验代码匹配原始 `coding.code`（如 MG、GLU、K、A1C），保持来源顺序，不默认按时间倒序。时间筛选边界包含等于，时间须带时区；无 `observed_at` 的记录不参加时间筛选。原单位、精度、参考范围、缺失值和时间字段保留，只有当前页返回的事实获得可引用证据 ID。

### 模拟医嘱的结构

例如补钾和复查检验是 **两次** `submit_order` 调用。下面展示输入格式，不代表任何具体患者应该接受此治疗：

```json
{
  "order": {
    "order_type": "medication",
    "patient_id": "患者检索返回的ID",
    "code": {"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"},
    "authored_at": "2023-11-13T10:15:00+00:00",
    "status": "active",
    "intent": "order",
    "dose": {"value": 20, "unit": "mEq"},
    "route": "oral"
  }
}
```

用药可另填 `rate: {"value": ..., "unit": ...}`，保留原任务的量及单位语义（如镁补充任务的小时数）。检查/会诊用 `order_type: "test"`/`"referral"`，仍须填写患者、编码、开立时间、status、intent，并填写 `priority`；可填 `occurrence_at`（预约/执行时间）和 `note`（如 SBAR 会诊文字）。检查/会诊没有用药的 dose、route、rate 字段。

血压使用独立工具，输入如下：

```json
{
  "observation": {
    "patient_id": "患者检索返回的ID",
    "code": "BP",
    "value": "118/77",
    "unit": "mmHg",
    "observed_at": "2023-11-13T10:15:00+00:00",
    "category": "vital-signs",
    "status": "final"
  }
}
```

这些字段都由模型填写。工具不补齐正确编码、剂量、患者或任务时间；格式合法但临床内容错误的提交也会保留，供 scorer 判断。缺必填字段、错误类型、无时区时间、未知字段等格式错误会返回工具错误，不产生成功回执。重复调用算多次提交，不自动合并。

回执包含 `operation`、`status: "accepted"`、`simulation: true`、`episode_id`、`submission_id` 和实际提交对象。**accepted 仅表示格式合格，不表示医嘱正确或已经执行。** AMA 原有记录器将工具请求和回执存入 `runs/<run-id>/messages.json`，每个 Episode 单独记录；提交顺序和失败调用都可检查。回执 ID 可引用，但它不是新的检验结果。患者文件、检验记录及快照均不被修改；重新运行从新的提交序号开始。

最终回答使用 AMA 的 `answer` 对象：患者检索回答 `{"mrn": "实际MRN"}`（从 identifiers 读取，不假设等于 patient_id），年龄回答 `{"age": 43}`，检验值回答 `{"value": 93}`，操作可回答 `{"completed": true}`；需要时添加 `observed_at`。找不到患者时将任务要求的文字放入 mrn，缺检验值时将任务要求的哨兵值放入 value。写“建议补钾”不会产生医嘱，必须调用提交工具。

## 评分

在运行命令输出的目录上执行（将路径换成你的实际目录）：

```bash
PYTHONPATH=src python -m ama.cli eval runs/<run-id>
```

`score_medagentbench(dataset, decisions, histories=...)` 直接读取 AMA 的答案对象和会话日志，不把答案转成原 benchmark 的 FINISH 列表，也不导入/运行 `refsol.py`。使用离线患者和检验事实计算参考值；不调用模型或服务器。

`cli.py` 统一读取 `messages.json`，将内容通过可选的 `histories` 参数传给 registry 中的 scorer。因此 scorer 不需要运行目录，也没有 `if scorer == "medagentbench"` 分支。原有 scorer 仅增加这个可选参数并忽略它，原评分规则不变；缺少会话文件的旧 ROCO/MIMIC 运行仍可评测。MedAgentBench 必须有日志，缺文件时明确报错，不将“没有日志”当作“没有开医嘱”。

### 十类任务的规则

| 任务 | 答案与操作检查 |
|---|---|
| task1 | `mrn` 与原 sol 一致；找不到时为 `"Patient not found"`；禁止尝试提交 |
| task2 | `age` 按原 grader 固定日期 2023-11-13 计算；禁止尝试提交 |
| task3 | 恰好一次成功血压提交；核对患者、BP、118/77、mmHg、时间、类别和状态 |
| task4 | `value` 为 24 小时窗口内最近镁值，无记录为 -1；禁止尝试提交 |
| task5 | 镁低时恰好一次正确 IV 补镁，核对 NDC、剂量、小时数、时间、状态等；无记录或 >1.9 时禁止尝试提交 |
| task6 | `value` 为窗口内血糖均值，无记录为 -1；绝对误差 <0.1；禁止尝试提交 |
| task7 | `value` 为最近血糖，无记录为 -1；禁止尝试提交 |
| task8 | 恰好一次正确骨科会诊，核对 SNOMED、stat、开立时间及完整 SBAR 原文 |
| task9 | 钾 <3.5 时恰好两次成功提交：先口服补钾，再次日 8 点复查；核对 NDC/LOINC、剂量（误差 ≤0.1）、单位、时间等；无记录或 ≥3.5 时禁止尝试提交 |
| task10 | 有记录时 `value` 和 `observed_at` 匹配；无记录为 -1；无记录或早于 2022-11-13T10:15:00+00:00 时恰好一次正确 HbA1C 检查，否则禁止尝试提交 |

沿用原 grader 的宽松答案规则：task3、task8 不判断最终答案的临床内容；task5、task9、task10 可仅返回 `{"completed": true}` 或 `{}`。如果提供 value，则按原规则核对；task10 有测量记录且提供 value 时也要提供对应 observed_at。task5 无镁记录时原 grader 不核对答案值。所有任务仍须有成功完成的 AMA Decision 和非 null 答案对象；缺答案、API 错误、未执行和弃答计失败，仍进入分母。输出格式与引用质量由通用 `output_quality` 单独报告，不混入任务规则。

保留原 grader 的时间细节：24 小时查询只有下界，没有额外的“不得晚于当前时间”上界；最近时间相同取来源顺序中第一条；HbA1C 恰好满一年不要求重开。部分开立/预约时间按原字符串条件判断，不自行改成更宽松的时区等价比较。单位不转换，参考值使用原 quantity 数值。评分将 MRN 解析为实际 Patient.id 后核对提交患者，适配本地工具的患者身份约定。

### 日志与结果

评分只认可 assistant 的实际工具调用及其对应 tool 成功回执，按 call ID、顺序、Episode、提交序号和载荷配对；不能用最终答案或普通文字伪造提交。禁止提交的分支连格式错误的尝试也不允许；需要提交的分支按成功提交数判断，允许失败后修正，但不能多开或颠倒医嘱。缺某个 Episode 日志或回执不匹配，该任务失败并报告 `history_error`。源数据缺少可核验的参考患者、数值或时间时明确报错，避免把不完整评测数据当作模型失败。

`runs/<run-id>/metrics.json` 保存：

- `task_success`：答案和操作都正确的任务比例，包含失败/未执行任务。
- `answer_accuracy`、`action_accuracy`：分别定位答案错误和操作错误。
- `completion`、`answer_object_format`、`valid_history`：完成、答案对象和日志状态。
- `per_task`：各任务类型的成功数、总数及比例。
- `per_episode`：每例的答案/操作判断、提交尝试数、成功提交数、日志错误和参考值。参考值只在评测产物中出现。

与原 grader 的对照测试仅在测试中做格式适配，以验证规则，不影响运行或评分接口。这些确定性测试的通过率不是模型实验成绩。

## 验证

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from ama.data import validate_dataset; print(validate_dataset(Path("datasets/medagentbench")))'
python -m pytest -q tests/test_medagentbench.py
python -m pytest -q tests/test_medagentbench_scoring.py
```

结构验证应输出 `[]`。Importer 自身也会在发布输出目录前调用结构验证。测试覆盖公开数据隔离、检索/分页/时区、提交校验、原数据不变、十类评分、边界、失败分母、日志配对及 `ama eval`。有相邻 MedAgentBench 仓库时，还会离线执行原 `refsol.py` 的对照测试；不需要 FHIR 服务或新增依赖。
