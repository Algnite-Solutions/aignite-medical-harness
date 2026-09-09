"""Timeline environment: the guarded world the agent acts on.

Binds (patient_id, as_of) at construction; tools cannot switch either.
All tool errors toward the model are generic (no leak of invisible evidence);
out-of-bounds attempts are tagged internally as violations and logged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import (
    Answer,
    CaseInput,
    Claim,
    Evidence,
    GetStateAction,
    ListEvidenceAction,
    ProposeStateUpdateAction,
    ReadEvidenceAction,
    StateSnapshot,
    SubmitAnswerAction,
    ToolResult,
)

_NOT_FOUND = "evidence_id not found among visible evidence (unknown, not yet recorded, or another patient)"


class TimelineEnvironment:
    def __init__(
        self,
        case: CaseInput,
        as_of: datetime,
        allow_state_tools: bool = True,
        initial_state: StateSnapshot | None = None,
    ) -> None:
        self.case = case
        self.patient_id = case.task.patient_id
        self.as_of = as_of
        self.allow_state_tools = allow_state_tools
        self._by_id: dict[str, Evidence] = {e.evidence_id: e for e in case.evidence}
        base = initial_state or StateSnapshot(patient_id=self.patient_id, version=0, as_of=as_of)
        self.snapshots: list[StateSnapshot] = [base]
        self._claim_counter = len(base.claims)
        self.violations: dict[str, int] = {"future": 0, "other_patient": 0}

    # ------------------------------------------------------------ visibility

    def _visible(self) -> list[Evidence]:
        rows = [
            e for e in self._by_id.values()
            if e.patient_id == self.patient_id and e.recorded_time <= self.as_of
        ]
        return sorted(rows, key=lambda e: (e.recorded_time, e.evidence_id))

    def _visible_ids(self) -> set[str]:
        return {e.evidence_id for e in self._visible()}

    def _check_ref(self, evidence_id: str) -> str | None:
        """Return violation tag if the id exists but is out of bounds; None if usable/unknown."""
        e = self._by_id.get(evidence_id)
        if e is None:
            return None
        if e.patient_id != self.patient_id:
            return "other_patient"
        if e.recorded_time > self.as_of:
            return "future"
        return None

    # ------------------------------------------------------------ dispatch

    def execute(self, action: Any) -> ToolResult:
        try:
            if isinstance(action, ListEvidenceAction):
                return self._list_evidence()
            if isinstance(action, ReadEvidenceAction):
                return self._read_evidence(action.evidence_id)
            if isinstance(action, GetStateAction):
                return self._get_state()
            if isinstance(action, ProposeStateUpdateAction):
                return self._propose_state_update(action)
            if isinstance(action, SubmitAnswerAction):
                return self._submit_answer(action.answer)
            return ToolResult(ok=False, error=f"unsupported action: {type(action).__name__}", error_kind="validation")
        except ToolResult as r:  # raised by helpers for early exit
            return r
        except Exception as exc:  # unexpected failures terminate the run (tool_error)
            return ToolResult(ok=False, error=f"internal error: {exc!r}", error_kind="internal")

    # ------------------------------------------------------------ tools

    def _list_evidence(self) -> ToolResult:
        rows = [
            {
                "evidence_id": e.evidence_id,
                "event_time": e.event_time.isoformat(),
                "recorded_time": e.recorded_time.isoformat(),
                "modality": e.modality,
                "source_locator": e.source_locator,
            }
            for e in self._visible()
        ]
        return ToolResult(ok=True, data={"evidence": rows, "count": len(rows)})

    def _read_evidence(self, evidence_id: str) -> ToolResult:
        e = self._by_id.get(evidence_id)
        if e is None:
            return ToolResult(ok=False, error=_NOT_FOUND, error_kind="validation")
        violation = self._check_ref(evidence_id)
        if violation:
            self.violations[violation] += 1
            return ToolResult(ok=False, error=_NOT_FOUND, error_kind="validation", violation=violation)
        return ToolResult(ok=True, data=e.model_dump(mode="json"))

    def _get_state(self) -> ToolResult:
        if not self.allow_state_tools:
            return ToolResult(
                ok=False,
                error="state tools disabled by strategy (full_history does not maintain derived state)",
                error_kind="validation",
            )
        return ToolResult(ok=True, data=self.snapshots[-1].model_dump(mode="json"))

    def _propose_state_update(self, action: ProposeStateUpdateAction) -> ToolResult:
        if not self.allow_state_tools:
            return ToolResult(
                ok=False,
                error="state tools disabled by strategy (full_history does not maintain derived state)",
                error_kind="validation",
            )
        current = self.snapshots[-1]
        if action.expected_version != current.version:
            return ToolResult(
                ok=False,
                error=f"version conflict: expected_version={action.expected_version}, current version={current.version}",
                error_kind="validation",
            )

        # validate the whole batch first; any failure rejects everything (atomic)
        for p in action.claims:
            for ref in p.evidence_refs:
                violation = self._check_ref(ref)
                if self._by_id.get(ref) is None or violation:
                    if violation:
                        self.violations[violation] += 1
                    return ToolResult(
                        ok=False,
                        error=f"claim key '{p.key}' references evidence not visible at current as_of",
                        error_kind="validation",
                    )
            if p.supersedes:
                target = next((c for c in current.claims if c.claim_id == p.supersedes), None)
                if target is None:
                    return ToolResult(ok=False, error=f"supersedes target not found: {p.supersedes}", error_kind="validation")
                if target.key != p.key:
                    return ToolResult(
                        ok=False,
                        error=f"supersedes target '{p.supersedes}' has key '{target.key}', expected '{p.key}'",
                        error_kind="validation",
                    )

        claims = [c.model_copy(deep=True) for c in current.claims]
        for p in action.claims:
            new_claim = Claim(
                claim_id=f"cl-{self.patient_id}-{self._claim_counter + 1:03d}",
                patient_id=self.patient_id,
                key=p.key,
                value=p.value,
                evidence_refs=list(p.evidence_refs),
                valid_from=p.valid_from or self.as_of,
                valid_to=None,
                status=p.status,
                supersedes=p.supersedes,
            )
            self._claim_counter += 1
            if p.supersedes:
                for i, c in enumerate(claims):
                    if c.claim_id == p.supersedes:
                        claims[i] = c.model_copy(update={"status": "superseded", "valid_to": new_claim.valid_from})
                        break
            claims.append(new_claim)

        snapshot = StateSnapshot(
            patient_id=self.patient_id,
            version=current.version + 1,
            as_of=self.as_of,
            claims=claims,
        )
        self.snapshots.append(snapshot)
        return ToolResult(ok=True, data={"version": snapshot.version, "claims": len(snapshot.claims)})

    def _submit_answer(self, answer: Answer) -> ToolResult:
        if answer.abstain and (answer.claims or answer.evidence_refs):
            return ToolResult(ok=False, error="abstain answer must not contain claims or evidence refs", error_kind="validation")
        refs = list(answer.evidence_refs) + [r for c in answer.claims for r in c.evidence_refs]
        for ref in refs:
            violation = self._check_ref(ref)
            if self._by_id.get(ref) is None or violation:
                if violation:
                    self.violations[violation] += 1
                return ToolResult(
                    ok=False,
                    error="answer references evidence not visible at current as_of",
                    error_kind="validation",
                )
        return ToolResult(ok=True, data={"accepted": True})


def summarize_state(snapshot: StateSnapshot) -> str:
    """Compact, model-safe rendering of a state snapshot (used for prior-state injection)."""
    if not snapshot.claims:
        return "(empty)"
    lines = [f"state version {snapshot.version}:"]
    for c in snapshot.claims:
        note = f" (supersedes {c.supersedes})" if c.supersedes else ""
        lines.append(f"- [{c.status}] {c.key} = {c.value!r}{note} refs={c.evidence_refs}")
    return "\n".join(lines)
