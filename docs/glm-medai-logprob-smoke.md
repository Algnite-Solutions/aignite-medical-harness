# GLM and MEDAI logprob capability smoke test

## Material Passport

- Date: 2026-10-05, Asia/Shanghai.
- ARS experiment-agent: bounded synthetic capability check; inline execution.
- Status: ANALYZED; four real API requests completed and standard run artifacts audited offline.
- User request: quick test of GLM and MEDAI logprob output.
- Entry points: `ama run` for inference, then `ama eval` for all four runs. Exact executable commands, working directory and artifact paths are saved in commands.json and summary.json.
- Data: one fictional marker-copy episode, no clinical cases or patient data.

## Result

| Endpoint/model | Requested top_logprobs | Response | Chosen-token records | Byte coverage |
|---|---:|---|---:|---|
| BigModel / glm-5.3-flash | 0 | Normal final answer; no logprobs | 0 | None; answer 257 bytes |
| BigModel / glm-5.3-flash | 3 | Normal final answer; no logprobs | 0 | None; answer 303 bytes |
| MEDAI / Qwen3.6-27B | 0 | Normal final answer; partial logprobs | 1 | 5 of 439 bytes |
| MEDAI / Qwen3.6-27B | 3 | Normal final answer; partial logprobs | 1 | 5 of 439 bytes |

All four requests explicitly set `logprobs=true`. GLM accepted the requests but returned no usable logprob object. This demonstrates absent output for these tested requests, not a universal claim about every GLM model or endpoint.

MEDAI returned one finite chosen-token logprob, whose bytes match the beginning of the reply. With top_logprobs=3, that one position also had three candidate probabilities. It still omitted the rest of the reply, including the marker in the answer. Candidate probabilities for the first position cannot substitute for missing chosen-token probabilities at later positions.

Neither tested deployment supplies complete answer-token probabilities here; these outputs cannot support the planned whole-diagnosis ProbScoreDx. Missing probabilities are not imputed and no sequence probability is reported.

## Configuration and integrity

- Official configured GLM endpoint: https://open.bigmodel.cn/api/paas/v4/chat/completions; requested model glm-5.3-flash, reasoning_effort=low, max_tokens=1024.
- Configured MEDAI endpoint: https://medaimodel.medicalaibase.cn:30000/v1/chat/completions; requested model Qwen3.6-27B, chat_template_kwargs.enable_thinking=false, max_tokens=512.
- Both: temperature=0, synchronous text-only, no tools, max_calls=1, timeout=90 seconds, no automatic retry.
- Returned model IDs match the configured IDs; all finish reasons are stop. There were no response truncations or API errors.
- Actual usage: GLM 683 total tokens; MEDAI 822; combined 1505. GLM/MEDAI tariffs were not verified, so no fee is imputed. No Bailian requests were made.
- MEDAI uses a process-local NO_PROXY exception for its configured host, with normal TLS verification.
- The harness now forwards validated logprobs/top_logprobs flags and records returned logprobs plus original answer content in standard diagnostics.jsonl. Defaults remain unchanged. 129 offline tests passed, including standard CLI tests preserving null and partial logprob responses and confirming ama eval makes no model call.
- summary.json stores byte-coverage checks and artifact hashes. analyze.py reads only saved standard artifacts; it makes no API calls and does not replace scoring.

## Limits and sources

One prompt per parameter setting: this is a quick capability check, not a performance benchmark or AUC reproduction. Only the named GLM model and MEDAI Qwen27 were tested; other MEDAI models, thinking-enabled Qwen, streaming, and tool trajectories were not tested. A response model ID does not attest a checkpoint or server implementation.

The [official Z.AI API reference](https://docs.z.ai/api-reference/llm/chat-completion) documents reasoning_effort=low for GLM-5.3-FLASH; the China endpoint observations above are established by this live test, not inferred from that documentation. Historical MEDAI probes on 2026-10-03 also returned only the first token; this test independently confirms the same limitation with top_logprobs=0 and 3 through the standard AMA CLI.
