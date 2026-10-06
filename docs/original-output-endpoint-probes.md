# Probe original AMA submissions and replay tool histories

Use the standard AMA entry points for inference and offline evaluation:

```sh
ama run DATASET --model MODEL --config MODELS --probing-plan PLAN --max-calls 1
ama eval RUN
```

Ordinary clinical runs do not automatically insert probes. A separate frozen
measurement plan binds completed standard AMA HPI and interactive runs. The
clinical histories and Decisions remain unchanged. No new clinical tool executes;
each measurement independently forks an original history through `Agent.chat()`
with at most one model call. Probe responses never enter the observed trajectory.

## Measurement designs

| `measurement_design` | History visible to the probe | Candidate diagnosis |
|---|---|---|
| `post_submission` | Complete original HPI or interactive history, including final assistant explanation | Each view's own original diagnosis |
| `tool_history_replay` | Initial interactive HPI prefix and every complete tool batch, including its image attachments | Same original interactive final diagnosis at every prefix |
| `evidence_prefix` | Historical evidence prefixes, preserving the existing legacy behavior | Existing fixed HPI/final-hindsight targets |

Post-submission copies the entire saved system, assistant, tool IDs/results and
final assistant message, then appends one user measurement request. Sufficient
and True fork separately from the same history; they never condition on each
other's responses.

Tool-history replay runs **after** the original clinical run completes. It keeps
only the initial HPI boundary and complete tool-result batches. Multiple tools in
one assistant response are atomic: no probe is inserted between incomplete tool
results. The corresponding image attachments remain in that batch. Unpaired,
duplicate or incomplete call/result IDs are rejected. No-tool cases have only the
initial prefix; no artificial final step is created. A pre-final reflection that
is not part of a tool batch does not create an extra measurement.

Replay preserves the original system and every message in each prefix. Only its
independent user measurement is appended. Future tool evidence and the final
assistant explanation are excluded. The final diagnosis appears only as the
proposed candidate in the probe question. Thus this is a retrospective curve of
support for the final diagnosis, not the agent's online confidence or stopping
policy. Its last tool prefix is distinct from a post-submission endpoint, which
also includes the final explanation. They must not be silently combined.

If an original diagnosis is absent, the corresponding selected measurements are
recorded in `missing_original_candidates.json`; no HPI fallback or new diagnosis
is generated. Original cases remain in the accuracy denominator. For replay,
missing final diagnoses make every selected measurement for that case missing.

## Configuration and frozen inputs

This is a fragment to include in an otherwise complete original-agent plan:

```json
{
  "schema": "ama-original-agent-probing",
  "measurement_design": "tool_history_replay",
  "probe_kinds": ["sufficient"],
  "label_readout": "first_complete_label",
  "top_logprobs": 20,
  "max_probe_tokens": 32,
  "workers": 1
}
```

`probe_kinds` accepts a non-empty unique list containing `sufficient`, `true`, or
both. Omission preserves both kinds in post-submission/replay and the existing
three-target evidence-prefix behavior. Jobs keep their existing canonical order.
Empty lists, duplicates, unknown names and incorrect types fail before requests.
Unselected kinds do not generate calls or count as missing. Selecting only
Sufficient means two endpoint measurements per ordinary post-submission case,
or one measurement per replay prefix.

The complete plan additionally needs:

| Field | Required content |
|---|---|
| `original_sources.hpi`, `original_sources.interactive` | Completed standard AMA run paths and absolute-path SHA256 maps for each source `manifest.json`, `messages.json`, `decisions.jsonl` |
| `original_sources.*.eval_hashes` | Frozen original metrics/reference hashes for evaluator-only verification |
| `snapshots_file`, `snapshots_sha256` | Frozen JSONL measurement snapshots and its SHA256 |
| `transport` | Shared `lease_path`; optional `request_delay_seconds`, `max_429_attempts`, `max_retry_wait_seconds` |
| `reuse_runs` | Optional existing measurement run paths for exact successful-job reuse |

Both clinical sources must cover the same admissions and use the requested model
and provider. Original sources must be standard AMA Decision runs; the historical
custom Qwen runner is not relabeled as such. The current clinical evaluator is
`mimic_cdm_open`; this is not an arbitrary-benchmark scorer adapter.

Use `ama.original_agent_probing.measurement_snapshots(histories, design)` during
offline input preparation, with `histories` mapping `hpi` and `interactive` to the
source message arrays for one case. Add that case's `episode_id` and a stable
`turn_id` to each returned snapshot. For replay, the source-message boundary can
be used for the turn ID. Construct a probe dataset with exactly matching episode
and turn IDs, containing placeholders rather than gold targets. Freeze the JSONL
and hashes. Input preparation makes no inference; execute only through `ama run`.
Snapshot history, order, boundary and replay metadata are reconstructed from the
original trajectory and verified before inference and again during evaluation.

Public plan names are `ama-original-agent-probing` and
`ama-interactive-probing`. New manifests use `original-agent-probing` or
`interactive-probing` in the existing contract field. Archived names remain
read-only aliases for dispatch, scoring and reuse. Original plan contents and
hashes are never rewritten or normalized. Reusing an unchanged archived plan for
resume is allowed; changing its schema spelling, selection or other contents
requires a new run instead of resuming under a different plan hash.

## Probability, accuracy and artifacts

The probe requests A/B, provider logprobs and the configured output token limit.
Read the first complete A/B token. Sum returned token probability mass mapping to
A and B, then report `p_ab = p_A / (p_A + p_B)`. Keep both provider masses,
`ab_mass`, chosen-token bytes/logprob and returned candidates. Both alternatives
must be available; missing candidates are invalid, never filled with zero.
`top_logprobs=20` is a candidate-token count, not twenty independent samples.

`first_complete_label` allows an explanation after the complete first label but
records trailing-output/format diagnostics separately. JSON/Markdown openings
or the first letter of a diagnosis cannot masquerade as A/B. The default
`single_label` reader remains supported. Missing probabilities, model mismatch
or failed calls stop original-agent inference before the next request.

| Artifact | Content |
|---|---|
| `requests/<hash>.json` | Exact independent measurement input |
| `probe_calls.jsonl` | Job identity, candidate, response, raw provider metadata/logprobs, usage and timing |
| `transport_attempts.jsonl` | Started/completed/rejected/failed attempts, retries and usage |
| `original_submissions/{view}/` | Byte-exact original Decisions and manifest |
| `missing_original_candidates.json` | Missing selected measurements and reasons |
| `metrics.json` | Original clinical scores, historical comparisons and probe probabilities |

Replay rows/features include `source_message_end`, `step`, `batch_size`,
`evidence_types`, `tool_failures`, `duplicate_results`, `diagnostic_impression`
and `source_view`. The candidate/job ID lives with each scored probe under
`per_prefix[].probes`; existing field names remain compatible. They allow offline
trajectory plotting without inventing within-batch timestamps.

`ama eval` reads saved logprobs offline and scores byte-exact archived clinical
submissions through the original scorer. It compares full source-cohort
per-episode scores with historical scores, and separately reports selected-cohort
accuracy. Probe answers never replace clinical Decisions. The measurement run's
`decisions.jsonl` is empty. Evaluation makes no model requests.

Job identity binds the snapshot, candidate, kind, request, model configuration,
original Decision and measurement request. Only exactly matching successful
jobs are reused; a Sufficient-only run can reuse matching Sufficient records from
a dual-probe run. Different measurement histories/prompts cannot cross-reuse.

Single-worker execution, process/shared leases and bounded429 recovery are
preserved. Default pacing is3 seconds; recovery respects Retry-After or bounded
backoff and increases pacing. Unknown network completion/5xx stops before the
next job, without automatic resubmission or treating unknown usage as zero.
There is no server KV-cache telemetry or calibration claim.

## Qwen configuration and validation

The existing `qwen36-35b-a3b-bailian-thinking` alias in `ama.json` preserves
`enable_thinking=true`, `thinking_budget=2048`, `temperature=0` and
`max_tokens=4096`. Model transport validates these parameters and sends them at
the top level for Bailian. Its synthetic tool-round-trip tests remain in the
native suite. See [the preserved Qwen rerun report](qwen35-open100-thinking-rerun.md)
for the historical80/100 interactive result. That custom-runner accuracy result
is not an independent AMA rerollout or the paper's AUC comparison.

Implementation validation uses synthetic fixtures via standard `ama run` and
`ama eval` CLI dispatch with a mocked provider. No paid/patient requests are
required to test batch boundaries, attachments, future-evidence exclusion,
selection, missing candidates, alias compatibility, reuse, resume, scoring or
rate-limit recovery. Existing real experiment records remain unchanged.
