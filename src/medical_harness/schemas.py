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


Action = Annotated[
    Union[
        ListEvidenceAction,
        ReadEvidenceAction,
        GetStateAction,
        ProposeStateUpdateAction,
        SubmitAnswerAction,
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
    request_timeout_seconds: float = 30.0  # enforced by real API client (later phase)
    deadline_seconds: float = 300.0
    max_retries: int = 1


class StrategyConfig(Strict):
    name: str = "versioned_state"
    state_tools_enabled: bool = True  # False => full_history: state tools explicitly disabled
    inject_prior_state: bool = True


class ModelConfig(Strict):
    type: str = "scripted"  # scripted | openai_compatible
    script_dir: str | None = None  # scripted only
    base_url: str | None = None  # openai_compatible only, e.g. https://open.bigmodel.cn/api/paas/v4
    model: str | None = None  # openai_compatible only; never hardcoded in code
    api_key_env: str = "GLM_API_KEY"  # secret lives in the environment, never in configs/logs
    temperature: float = 0.0


class RunConfig(Strict):
    name: str = "mock"
    data_dir: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cases: list[str] | None = None  # None => all
