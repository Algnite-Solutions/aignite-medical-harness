# AMA Dataset

AMA accepts `ama-dataset` only. A dataset describes a fixed release of
observations, not an agent interaction protocol. An Episode contains ordered
Turns; each Turn releases Evidence and receives one Decision. The array order is
authoritative. `available_at` is optional and must never be fabricated to make
a multi-turn Episode valid.

## Files

```text
dataset.json       required: schema, name, splits
episodes.jsonl     required: one model-visible Episode per line
targets.jsonl      optional: evaluator-only per-turn target payloads
eval.json          optional: scorer and evaluator-only global rules
provenance.jsonl   optional: local source identifiers and record locators
instructions.txt  optional generated task instruction; pass explicitly to ama run
```

`ama run` reads only `dataset.json` and `episodes.jsonl`. `ama eval` may read
`targets.jsonl` and `eval.json`; neither path reads `provenance.jsonl`. The
dataset card and `import_report.json` describe licensing, derivation, exclusions,
and temporal limitations. No source identifiers or gold labels belong in
`episodes.jsonl` unless they are genuinely part of model-visible evidence.

## Minimal example

`dataset.json`:

```json
{"schema":"ama-dataset","name":"example","splits":{"all":["case-1"]}}
```

One `episodes.jsonl` line:

```json
{"id":"case-1","turns":[{"id":"t1","observation":"History available","evidence":[{"id":"hpi","type":"note","text":"Abdominal pain"}]},{"id":"t2","observation":"Labs available","evidence":[{"id":"lab","type":"lab","text":"WBC 14"}]}]}
```

`targets.jsonl` can contain `{"id":"case-1","turns":{"t2":{"answer":{"primary_diagnosis":"appendicitis"}}}}`.
The scorer interprets the target payload; the model cannot access it during a
run. `eval.json` can contain `{"scorer":"exact","rules":{}}`.

Evidence has `id`, optional `type`, optional `text`, and optional relative
`file`; at least one of `text` and `file` must be non-empty. Only safe paths
inside the dataset are accepted. JPEG, PNG, GIF, and WebP files are delivered
as image parts. Omit absent fields rather than writing empty or null values.
Evidence IDs are unique within an Episode, as are Turn IDs; Episode IDs are
unique within a dataset. If true `available_at` values are supplied, their
known subsequence must be monotonic.

## Decision and protocols

Every accepted Decision is `{"turn_id":"t1","answer":...,"citations":[]}`.
`answer` is task-defined JSON; `null` means abstention and requires no
citations. In `direct_decision` (the default), newly released Evidence is shown
inline and citations may reference any Evidence visible so far. In `tool_agent`,
the model may call `list_evidence` and `read_evidence`; citations must refer to
Evidence it actually read. Tool calls are trace events, not fields of Decision.

```bash
ama run datasets/thyroid_demo --model scripted --protocol tool_agent \
  --instruction-file datasets/thyroid_demo/instructions.txt
```

The model adapter does not parse Decisions or choose protocols. The run manifest
records the chosen protocol, the complete instruction, the resolved model
configuration, and hashes of the two model-visible input files.
