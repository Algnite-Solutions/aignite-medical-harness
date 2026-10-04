# Interactive 低于 HPI：Qwen27／Qwen35 部署診斷

## Material Passport

- Material type: exploratory validation report; ARS experiment-agent, validate workflow, inline execution.
- Verification status: ANALYZED. Synthetic API behavior observed; clinical correction not yet verified.
- Date: 2026-10-04, Asia/Shanghai.
- Inputs: preserved PR13 open100 Qwen35 baseline; actual native API requests/responses from authored laboratory-marker fixtures.
- New scope: 84 synthetic HTTP requests, 0 patient requests. No gold labels or previous model answers in inference payloads.
- Evidence: `analysis.json`, `v1/`, `repeat2_nonthinking/`, `repeat2_thinking/`, `repeat3_nonthinking/`, `repeat3_thinking/`, `medai_named8/`.

## Current conclusion

**53/100 HPI versus 28/100 interactive is a deployment-process anomaly that must be explained before attributing it to medical knowledge.** Of the 100 interactive cases, 89 never called a tool and 58 issued valid but diagnosis-less final answers before any tool call. Parsing a prior tool result cannot account for those 58 cases.

The live comparison establishes a failure of automatic evidence acquisition under the original medical prompt and the tested Bailian non-thinking configurations. Enabling thinking with a bounded budget restores acquisition in synthetic tasks. The client preserves native calls and tool-result IDs. This does not yet certify clinical recovery, nor isolate the server-side template, weights, or decoding mechanism.

## Matched synthetic evidence

Each repeated condition uses the same medical system prompt and five real harness tool schemas. The answer is a newly authored marker available only in a laboratory tool result; it is absent from the initial messages. Markers were AURORA_K57, LANTERN_P93, and ORBITAL_R28. These are three variants of one engineered task, not three independent clinical scenarios.

| Endpoint/model | Thinking | Runs with tool acquisition | Exact marker recovered |
|---|---|---:|---:|
| MEDAI Qwen3.6-27B | off | 3/3 | 3/3 |
| Bailian qwen3.6-27b | off | 0/3 | 0/3 |
| Bailian qwen3.6-27b | on | 3/3 | 3/3 |
| Bailian qwen3.6-35b-a3b | off | 0/3 | 0/3 |
| Bailian qwen3.6-35b-a3b | on | 3/3 | 3/3 |

The MEDAI thinking original-prompt control also succeeded, once. Normal automatic-tool prompts and explicitly named first calls succeeded on both Bailian models with thinking off. Earlier removal of only the final JSON envelope did not repair the medical task. Thus this is task/prompt/configuration-dependent; neither the JSON envelope alone nor a universal absence of tool support is an established explanation.

Across all 84 new requests, native tool-call loss was 0, history ID mismatches were 0, and non-string tool-result requests were 0. All finish reasons were `stop` or `tool_calls`; none was `length`. Returned model IDs matched requested IDs. API names do not attest identical checkpoints or serving templates across providers.

All unsuccessful runs remain in the record. Forcing the initial MEDAI laboratory call reached both a four-call limit and, in a separate repeat, an eight-call limit without a final answer. It is therefore not a validated MEDAI remedy, despite successful MEDAI automatic acquisition. This separate loop behavior is unresolved and must not be omitted.

## Configuration and patch

- Temperature 0; non-thinking `max_tokens=4096` for this comparison, matching the patient baseline rather than the earlier 512-token synthetic probe.
- Bailian thinking: top-level `enable_thinking=true`, `thinking_budget=2048`, `max_tokens=4096`.
- MEDAI thinking switch: `chat_template_kwargs.enable_thinking`; its thinking-only synthetic condition used `max_tokens=8192`.
- Original baseline prompt, cases, scores, and model aliases preserved.
- Experiment branch adds explicit Bailian Qwen27 and thinking aliases, validated `thinking_budget` transport, and offline tool-cycle tests. 118 tests passed.
- The original Qwen35 non-thinking budget runner cannot be reused for thinking-mode accounting; it deliberately rejects reasoning-token usage.

Official documentation confirms the compatible Function Calling contract and the thinking-budget parameter: [Function Calling](https://help.aliyun.com/zh/model-studio/qwen-function-calling), [deep thinking](https://help.aliyun.com/zh/model-studio/deep-thinking/). Forced/named calls are incompatible with Bailian thinking mode, so those tests were performed only with thinking off. The [Qwen27 model card](https://help.aliyun.com/zh/model-studio/qwen3-6-27b) provides the Beijing list prices used for Bailian accounting.

The [official Qwen27 model card](https://huggingface.co/Qwen/Qwen3.6-27B) documents top-level thinking flags for Model Studio and nested chat-template flags for self-hosted compatible services. The captured requests follow these locations; this does not establish identical provider templates or weights.

## Additional prompt factorial and clinical-gap attribution

The prospectively saved `prompt_factorial_v1/prospective_plan.json` crosses plain/medical task instructions with presence/absence of the final envelope, with one extra explicit-final-only applicability control. All 15 conditions and 29 HTTP requests are retained. This is one fixture per condition, not a clinical validation.

| Endpoint | Plain, no envelope | Plain + envelope | Medical, no envelope | Medical + envelope | Medical + explicit final-only envelope |
|---|---|---|---|---|---|
| Bailian27, thinking off | tool + marker | tool + marker | tools + marker | no tool | no tool |
| Bailian35, thinking off | tool + marker | no tool | no tool | no tool | no tool |
| MEDAI27, thinking off | tool + marker | tool loop, no final | tools + marker | tools + marker | no tool |

“Marker” in this table means the unique hidden value appears in the final answer after retrieval. Natural-language no-envelope answers are not penalized for lacking a JSON decision. `interpreted_analysis.json` keeps this endpoint distinct from strict exact-answer/normalization metrics in the raw run summary. The MEDAI plain-plus-envelope loop reached the eight-call cap; it is included rather than dropped. Some looped MEDAI calls requested nonexistent imaging and received a normal handler-error message; this is not evidence of lost or misparsed results.

The factorial rejects a universal JSON-only explanation. Qwen27's medical no-envelope condition works here; Qwen35's does not. Explicitly limiting the envelope to the final phase is not a demonstrated repair on any of these endpoints in this fixture. The supported mechanism remains a model/prompt/serving-configuration interaction.

A new local audit of the preserved clinical baseline (`clinical_gap_local_audit.json`, zero API calls) partitions the 29 HPI-correct/interactive-noncorrect cases: **27 returned no diagnosis before any tool use**, one had an invalid final turn ID, and one gave a noncorrect diagnosis after using tools. The reverse transition occurred in four cases. Thus the observed drop is predominantly lost final diagnoses before evidence acquisition, rather than 29 newly mistaken diagnoses after reading extra evidence.

This also holds descriptively across source groups: HPI versus interactive correct counts are diverticulitis 9 versus 1, cholecystitis 6 versus 6, appendicitis 23 versus 11, and pancreatitis 15 versus 10, each out of 25. There is no group in which the overall degradation is reversed into an interactive advantage. These counts do not identify a causal knowledge deficit.

## Prepared clinical diagnostic — not executed

Select positions 0–2, 25–27, 50–52, and 75–77 from the frozen cohort, without selecting by model outcomes. The source groups are 2 appendicitis, 3 cholecystitis, 3 diverticulitis, and 4 pancreatitis; the positions are not disease blocks. Compare the same 12 cases under four configurations: Bailian35 thinking, Bailian27 non-thinking, Bailian27 thinking, and MEDAI27 non-thinking. Existing Qwen35 HPI/interactive/full-info results on those IDs provide reference observations without new baseline calls.

Primary diagnostic endpoint: proportion acquiring evidence through native tools. Secondary endpoints: presence of a final diagnosis and unchanged PR13 automatic diagnosis score. Keep every attempted case, including failures. Targets are loaded for local scoring only after all inference. These 48 proposed encounters are an exploratory mechanism check; they are not an independent benchmark or AUC estimate.

The API launch was rejected by automatic approval because real case HPI/evidence would be exported to MEDAI and Bailian Qwen27, while the earlier explicit payload authorization named Bailian Qwen35. The broader all-API authorization was not accepted as destination-specific patient-content permission. No patient request was sent, and no workaround was attempted. A precise authorization question is pending.

## Cost and interpretation limits

The new 84 requests comprise 45 Bailian and 39 MEDAI requests. New Bailian list-price estimate: CNY0.2290590; cumulative Bailian accounted estimate including the original experiment and earlier diagnostics: CNY1.8886914. Full-context reservations precede each Bailian request and the CNY35 ceiling remains. MEDAI tariffs are unavailable; actual usage is saved and no fee is imputed. These are price-based estimates, not billing-statement reconciliation.

The evidence supports an operational acquisition failure and a synthetic configuration remedy. It does not demonstrate that clinical accuracy improves, that MEDAI serves identical weights, or that a general medical training deficit exists. No confidence or complete diagnosis-token logprobs were collected; diagnosis accuracy is not the paper's AUC.

## Statistical interpretation check

11/11 fallacy types checked, with individual dispositions in `analysis.json`: Simpson's paradox, ecological fallacy, Berkson selection, collider bias, base-rate neglect, regression to the mean, survivorship, look-elsewhere, forking paths, correlation/causation, and reverse causality. No population inference or significance test is claimed. Engineered fixtures, exploratory prompt changes, endpoint confounding, unequal clinical-subset groups, and retained call-limit failures remain visible.
