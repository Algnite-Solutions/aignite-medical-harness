# Small compatibility check — 2026-09-26

These are request-format checks, not a benchmark or a model ranking. Only public ROCOv2 demo
image 000001 and simple arithmetic were used. Local raw records are retained under
`runs/minimal-agent-smoke/` and are not committed.

| Request | Observed result |
|---|---|
| GLM `glm-4v-flash`, one image, structured run | Valid Decision envelope, but answer omitted caption; separate eval completed and scored the missing caption as empty. |
| Qwen `Qwen3.6-27B`, one image, default thinking | API returned text containing `<think>` plus JSON; rejected as invalid Decision. |
| Same Qwen image, `enable_thinking=false` | API returned prose plus an answer object, not the required Decision; rejected. |
| Qwen, text-only add tool + follow-up | One tool call, final answer and subsequent follow-up completed; full history retained. |
| Qwen, tool returning an image | First request produced a tool call; second request with the tool result/image returned HTTP 400. |

Qwen used the configured medaimodel endpoint; GLM used its configured BigModel endpoint.
The tool/image error proves that this particular request was rejected, not that every deployment
of that model lacks the capability. The two invalid Qwen decisions are sample observations,
not a claim that the model can never follow JSON instructions.

`qwen36` explicitly sets `chat_template_kwargs.enable_thinking=false`. This does not guarantee
JSON compliance. There is no text extraction, repair dialogue, role merging or compatibility
retry. Invalid replies remain visible in records and count as missing predictions in eval.

For an initial image/run example, `--model glm-vision` completed the pipeline, not the full task requirements. Validate your own
endpoint before a larger experiment. Deterministic tests cover the harness independently of
these provider behaviors:

```bash
python3 -m pytest -q
ama run datasets/rocov2_demo --episode ROCOv2_2023_test_000001 --model glm-vision
# Execute the separate ama eval command printed by run.
```
