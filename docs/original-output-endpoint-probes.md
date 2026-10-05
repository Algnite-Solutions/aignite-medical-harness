# Probe after the original AMA output

Use the existing standard CLI with a frozen original-agent probing plan whose
`measurement_design` is `post_submission`:

```sh
ama run DATASET --model MODEL --config MODELS --probing-plan PLAN --max-calls 1
ama eval RUN
```

The plan binds previously completed standard AMA HPI and interactive runs. Each
measurement copies a source run's **entire** saved message sequence, including
its unchanged system message, tool IDs/results and final assistant submission.
It appends one user request for an A/B probability measurement. Sufficient and
True probes are separate forks, so neither conditions on the other's response.
No tools execute and no new clinical diagnosis replaces the archived Decision.
`ama eval` scores the original submissions through `mimic_cdm_open`; it reads
probe logprobs offline and does not execute inference.

This mode measures each view's own submitted diagnosis after completion. It is
distinct from `evidence_prefix` mode, which measures fixed historical candidates
at earlier tool boundaries and adds a measurement exception to the system.
Requests from these designs have different identities and cannot be reused
across modes. Original abstentions remain in accuracy denominators; their
diagnosis-specific probes are missing, never imputed or replaced.

Because the final assistant explanation is retained, this endpoint is a
post-answer self-assessment and may condition on the model's own explanation.
It cannot establish evidence-only sufficiency, clinical calibration, or a causal
benefit from interactive evidence. HPI and interactive endpoints concern their
respective actual diagnoses, not a single fixed candidate across both views.
