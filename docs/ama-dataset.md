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
{"turn_id":"t1","answer":{"caption":"A chest image.","cuis":[]},"citations":["scan"],"reasoning_summary":"The visible anatomy supports this description."}
```

The task defines the answer. Null means abstention and requires no citations. Citations may reference only evidence released so far.

A ROCOv2 target line:
```json
{"id":"image-1","turns":{"t1":{"answer":{"caption":"A chest image.","cuis":[]},"required_evidence":["scan"]}}}
```

Use `{"scorer":"rocov2"}` in `eval.json`. Without an evaluation configuration, the runner reports completion only.

Both run and chat use the same Agent. The runner owns the final envelope; dataset instructions describe only the task and answer fields. Run requests a concise 2–4 sentence evidence-based `reasoning_summary`. Chat remains exploratory.

Contract version 2 preserves the raw reply and records `output_validation` alongside the normalized Decision. A complete Decision takes precedence over a matching shorter answer, even when surrounded by prose. Conflicting conclusions and explicitly wrong turn IDs fail extraction. An unambiguous bare answer is recoverable, but absent citations and summary remain marked missing. Only an explicit summary field or a labeled “Reasoning summary” block supplies the summary. No additional model calls are made.

Trusted tools declare released IDs through `ToolResult.evidence_ids`; successful releases are recorded in `evidence_release` events and remain available throughout the episode. IDs mentioned only in result text or an imaging catalog are not registered. Unknown citations are retained and reported invalid without erasing a recoverable conclusion. Normalization emits a separate `decision` event after the raw `exchange`.

`ama eval` reports task scores and `aggregate.output_quality`: recoverable-answer rate, strict output compliance, valid nonempty citations, and summary presence. Rates use all expected turns as the denominator. Strict compliance measures the four-field JSON envelope and types; citation validity is checked separately. These metrics do not establish clinical grounding or summary quality. Versionless historical runs remain readable, with new quality metrics set to null.

`--tools file.py` loads trusted tools. Instructions default to instructions.txt and can be replaced with `--instruction-file`; manifests record the actual prompts, hashes, model configuration, and output contract version. Only eval opens reference answers and rules. Failed and missing turns remain in evaluation denominators.

See the [code walkthrough](minimal-agent.zh-CN.md) for runnable examples.
