# AMA dataset format

Array order defines when evidence is released. An optional `available_at` must be a real timestamp.

| File | Contents | Read by |
|---|---|---|
| dataset.json | schema, name, splits | run, chat and eval |
| episodes.jsonl | one Episode per line | run, chat and eval |
| instructions.txt | global task instructions | run and chat |
| targets.jsonl | per-turn reference answers | eval only |
| eval.json | scorer and optional rules | eval only |
| provenance.jsonl | source locators | neither run nor eval |

`dataset.json`:
```json
{"schema":"ama-dataset","name":"example","splits":{"test":["image-1"]}}
```

One line of `episodes.jsonl`:
```json
{"id":"image-1","turns":[{"id":"t1","evidence":[{"id":"scan","type":"image","file":"artifacts/scan.jpg"}]}]}
```

Evidence needs at least one of `text` or a safe relative `file`. Type, observation and true availability time are optional; omit absent values. JPEG, PNG, GIF and WebP evidence is sent as image content. Other files are read as UTF-8 text; unsupported binary formats must be converted first. Global task requirements belong in instructions.txt, not repeated in each observation.

Episode IDs are unique across a dataset; Turn and Evidence IDs are unique within an Episode. Known timestamps must be monotonic.

Decision:
```json
{"turn_id":"t1","answer":{"caption":"A chest image.","cuis":[]},"citations":["scan"]}
```

The task defines the answer. Null means abstention and requires no citations. Citations may reference only evidence released so far.

A ROCOv2 target line:
```json
{"id":"image-1","turns":{"t1":{"answer":{"caption":"A chest image.","cuis":[]},"required_evidence":["scan"]}}}
```

Use `{"scorer":"rocov2"}` in `eval.json`. Without an evaluation configuration, the runner reports completion only.

Both run and chat use the same Agent. Run parses a Decision outside the Agent; chat allows natural-language follow-ups and cannot be scored. `--tools file.py` optionally loads an explicit TOOLS list. Instructions default to instructions.txt and can be replaced with `--instruction-file`; manifests record actual prompts, their instruction hash, model configuration and visible dataset hashes. Data checks run automatically. Only eval opens reference answers and rules. Failed and missing turns remain in evaluation denominators.

The [Chinese lab guide](rocov2-lab-guide.zh-CN.md) is for human readers and never enters model input.
