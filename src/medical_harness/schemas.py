"""Core data objects, actions, and run configuration.

Design invariants (enforced elsewhere, stated here):
- patient_id / as_of are bound by the environment, never action parameters.
- Visibility = same patient AND recorded_time <= as_of (inclusive).
- Evidence is immutable; revisions are new Evidence rows.
- The runner never auto-resolves contradictions between claims.
- Gold files are read only by the evaluator, never enter the model context.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- data objects

class Evidence(Strict):
    evidence_id: str
    patient_id: str
    event_time: datetime
    recorded_time: datetime
    modality: str
    content: str
    source_locator: str
    data_version: str = "v1"
    # trajectory/episode extensions (v0.2); deterministic workflow validators read
    # explicit tags only — the model under test never assigns evaluator tags
    tags: list[str] = Field(default_factory=list)
    artifact_path: str | None = None
    mime_type: str | None = None


ClaimStatus = Literal["active", "superseded", "contradicted", "uncertain"]


class Claim(Strict):
    claim_id: str
    patient_id: str
    key: str
    value: Any
    evidence_refs: list[str] = Field(default_factory=list)
    valid_from: datetime
    valid_to: datetime | None = None
    status: ClaimStatus = "active"
    supersedes: str | None = None


class StateSnapshot(Strict):
    patient_id: str
    version: int
    as_of: datetime
    claims: list[Claim] = Field(default_factory=list)


class AnswerClaim(Strict):
    key: str
    value: Any
    evidence_refs: list[str] = Field(default_factory=list)


class Answer(Strict):
    question_id: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    abstain: bool = False
    reason: str = ""


class Question(Strict):
    question_id: str
    text: str
    key: str | None = None


class Timepoint(Strict):
    as_of: datetime
    questions: list[Question] = Field(default_factory=list)


class CaseTask(Strict):
    patient_id: str
    timepoints: list[Timepoint] = Field(default_factory=list)


class CaseInput(Strict):
    case_id: str
    task: CaseTask
    evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------- gold (evaluator-only)

class GoldExpectation(Strict):
    abstain: bool = False
    value: Any = None
    key: str | None = None


class GoldAnswer(Strict):
    question_id: str
    expected: GoldExpectation
    allowed_refs: list[str] = Field(default_factory=list)
    stale_refs: list[str] = Field(default_factory=list)
    forbidden_refs: list[str] = Field(default_factory=list)
    note: str = ""


class CaseGold(Strict):
    case_id: str
    answers: list[GoldAnswer] = Field(default_factory=list)


# ---------------------------------------------------------------- trajectory (v0.2)

class TrajectoryTurn(Strict):
    turn_id: str
    as_of: datetime
    phase: str | None = None
    event: str = ""
    world_message: str = ""
    release_evidence_ids: list[str] = Field(default_factory=list)


class Episode(Strict):
    schema_version: str = "0.2"
    episode_id: str
    patient_id: str
    mode: str = "teacher_forced_replay"
    workflow_id: str = ""
    workflow_file: str = "workflow.json"
    evidence_file: str = "evidence.jsonl"
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)  # assembled by the loader


class DiagnosticHypothesis(Strict):
    label: str
    status: Literal["possible", "supported", "confirmed", "ruled_out"]
    evidence_refs: list[str] = Field(default_factory=list)


class ClinicalState(Strict):
    hypotheses: list[DiagnosticHypothesis] = Field(default_factory=list)
    clinical_stage: str | None = None  # disease stage (e.g. TNM); NOT the workflow node
    unresolved_questions: list[str] = Field(default_factory=list)


class ClinicalAction(Strict):
    action_type: str
    target: str | None = None
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class TurnDecision(Strict):
    turn_id: str
    workflow_state: str  # workflow node (e.g. "pathology"); distinct from clinical_stage
    state: ClinicalState = Field(default_factory=ClinicalState)
    next_action: ClinicalAction | None = None
    report: str | None = None
    abstain: bool = False


# ---------------------------------------------------------------- workflow (evaluator-side policy)

class TransitionRule(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")
    when: list[str] = Field(default_factory=list)  # evidence tags required for this transition
    allowed_actions: list[str] = Field(default_factory=list)

    @property
    def rule_id(self) -> str:
        return f"{self.from_state}->{self.to_state}"


class WorkflowConfig(Strict):
    workflow_id: str
    version: str = "0.1"
    states: list[str]
    initial_state: str | None = None  # defaults to states[0]
    transitions: list[TransitionRule] = Field(default_factory=list)

    def start_state(self) -> str:
        return self.initial_state or self.states[0]


class TransitionResult(Strict):
    state_valid: bool
    action_valid: bool
    transition_valid: bool
    violations: list[str] = Field(default_factory=list)
    matched_rule_ids: list[str] = Field(default_factory=list)


class TurnGold(Strict):
    expected_workflow_state: str | None = None
    expected_hypotheses: dict[str, str] | None = None  # label -> status
    expected_stage: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    required_evidence_refs: list[str] = Field(default_factory=list)
    expect_abstain: bool = False


class EpisodeGold(Strict):
    episode_id: str
    turns: dict[str, TurnGold] = Field(default_factory=dict)


# ---------------------------------------------------------------- actions

class ListEvidenceAction(Strict):
    type: Literal["list_evidence"] = "list_evidence"


class ReadEvidenceAction(Strict):
    type: Literal["read_evidence"] = "read_evidence"
    evidence_id: str


class GetStateAction(Strict):
    type: Literal["get_state"] = "get_state"


class ProposedClaim(Strict):
    key: str
    value: Any
    evidence_refs: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None  # defaults to env as_of
    supersedes: str | None = None
    status: Literal["active", "uncertain"] = "active"


class ProposeStateUpdateAction(Strict):
    type: Literal["propose_state_update"] = "propose_state_update"
    claims: list[ProposedClaim] = Field(default_factory=list)
    expected_version: int


class SubmitAnswerAction(Strict):
    type: Literal["submit_answer"] = "submit_answer"
    answer: Answer


class SubmitTurnDecisionAction(Strict):
    type: Literal["submit_turn_decision"] = "submit_turn_decision"
    decision: TurnDecision


Action = Annotated[
    Union[
        ListEvidenceAction,
        ReadEvidenceAction,
        GetStateAction,
        ProposeStateUpdateAction,
        SubmitAnswerAction,
        SubmitTurnDecisionAction,
    ],
    Field(discriminator="type"),
]

ACTION_ADAPTER: TypeAdapter = TypeAdapter(Action)


def parse_action(obj: Any) -> ListEvidenceAction | ReadEvidenceAction | GetStateAction | ProposeStateUpdateAction | SubmitAnswerAction:
    """Validate an action payload (dict or instance) against the Action union."""
    return ACTION_ADAPTER.validate_python(obj)


class ToolResult(Strict):
    ok: bool
    error: str | None = None
    error_kind: Literal["validation", "internal"] | None = None
    # violation is logged in events but NEVER shown to the model (existence of
    # invisible evidence must not be leaked through error messages)
    violation: Literal["future", "other_patient"] | None = None
    data: Any = None

    def public(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "data": self.data}


# ---------------------------------------------------------------- config

class BudgetConfig(Strict):
    max_steps: int = 12
    max_model_calls: int = 12
    per_turn_model_calls: int | None = None  # episode turns only; None = no per-turn cap
    request_timeout_seconds: float = 30.0  # enforced by real API client
    deadline_seconds: float = 300.0
    max_retries: int = 1


class StrategyConfig(Strict):
    # v0.2 rename: the old "full_history" never injected full history into context;
    # its actual behavior (fresh session per question + full visible-evidence
    # retrieval + no persistent state) is now called stateless_retrieval.
    name: str = "versioned_state"
    state_tools_enabled: bool = True  # False => stateless_retrieval: state tools explicitly disabled
    inject_prior_state: bool = True


class ModelConfig(Strict):
    type: str = "scripted"  # scripted | openai_compatible
    script_dir: str | None = None  # scripted only
    base_url: str | None = None  # openai_compatible only, e.g. https://open.bigmodel.cn/api/paas/v4
    model: str | None = None  # openai_compatible only; never hardcoded in code
    api_key_env: str = "GLM_API_KEY"  # secret lives in the environment, never in configs/logs
    temperature: float = 0.0


class TraceConfig(Strict):
    # sensitive-content policy (Gate 0): before real patient data, full evidence
    # content and full model context must not be persisted by default.
    save_model_context: bool = True
    save_evidence_content: bool = True


class RunConfig(Strict):
    name: str = "mock"
    data_dir: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cases: list[str] | None = None  # None => all
    trace: TraceConfig = Field(default_factory=TraceConfig)
    experiment: dict[str, Any] | None = None  # experiment card (playbook §8), stored verbatim in the manifest
