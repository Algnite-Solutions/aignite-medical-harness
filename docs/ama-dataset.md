# AMA dataset format

Array order defines when evidence is released. An optional `available_at` must be a real timestamp.

| File | Contents | Read by |
|---|---|---|
| dataset.json | schema, name, splits | run and eval |
| episodes.jsonl | one Episode per line | run and eval |
| targets.jsonl | per-turn reference answers | eval only |
| eval.json | scorer and optional rules | eval only |
| provenance.jsonl | source locators | neither run nor eval |

`dataset.json`:
```json
{"schema":"ama-dataset","name":"example","splits":{"test":["image-1"]}}
```

One line of `episodes.jsonl`:
```json
{"id":"image-1","turns":[{"id":"t1","observation":"Describe this image.","evidence":[{"id":"scan","type":"image","file":"artifacts/scan.jpg"}]}]}
```

Evidence needs at least one of `text` or a safe relative `file`. Type, observation and true availability time are optional; omit absent values. JPEG, PNG, GIF and WebP evidence is sent as image content. Other files need useful accompanying text when the model must understand them.

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

The built-in interaction directly presents evidence and requests a Decision. Custom tool protocols implement `InteractionProtocol` and are passed to `run_episode`. Optional instructions are supplied with `--instruction-file`; manifests record their text and hash, model configuration and visible dataset hashes.

The [Chinese lab guide](rocov2-lab-guide.zh-CN.md) is for human readers and never enters model input.
