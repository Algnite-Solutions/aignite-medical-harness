# AMA — minimal lab evaluation runner

[中文](README.zh-CN.md) | English

AMA currently includes one task: **ROCOv2 image captioning and concept prediction**.

`Episode → Turn → Evidence → Decision`: one image is one episode with one turn. The replay loop also supports ordered multi-turn data.

## Try it offline

```bash
python3 -m pip install -e ".[dev]"
ama validate datasets/rocov2_demo
ama run datasets/rocov2_demo --model scripted
```

Run the `ama eval` command printed at completion, then open the generated `report.md`. It shows images, model answers, references, and Chinese explanations of the scores. The scripted model plays three canned answers; it is a pipeline check.

The [Chinese lab guide](docs/rocov2-lab-guide.zh-CN.md) explains the task, metrics, medical terms, and all three demo reference captions.

## Use a vision model

Configure aliases in `ama.json` and keys in local `.env` using `.env.example`.

```bash
ama run datasets/rocov2_demo --model glm-vision \
  --instruction-file datasets/rocov2_demo/instructions.txt
```

Each turn directly presents the image and accepts a JSON Decision with `turn_id`, task-defined `answer`, and `citations`. For ROCOv2, the answer contains `caption` and `cuis`. A null answer means abstention.

## Read the code

Start with `data.py` (objects), `agent.py: run_episode` (replay), `protocols.py: DirectDecisionProtocol` (prompt and parsing), `model.py` (transport), and `scorer.py: score_rocov2` (scoring). `cli.py` and `recorder.py` handle commands and run artifacts.

Tool use has an extension boundary, `InteractionProtocol`, and the model transport accepts optional tool definitions. Pass a custom protocol object to `run_episode(protocol=...)` to extend the loop. There is no built-in tool agent or CLI protocol selector.

## Data

```bash
ama import rocov2 --source /path/to/ROCOv2 --out datasets/rocov2 --split test --limit 100
```

Inference reads the visible dataset and referenced images. Evaluation reads targets and scoring rules. Source provenance and the Chinese reading guide are separate from model input. Caption token overlap and concept-ID overlap measure agreement with references, not clinical correctness.

See the [dataset format](docs/ama-dataset.md) and the [ROCOv2 data card](datasets/rocov2_demo/DATASET_CARD.md). Runs are saved under `runs/`; run tests with `pytest -q`.
