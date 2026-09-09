"""Explicit workflow: loader + deterministic validator (evaluator-side policy).

The agent may receive a clinical guidance SUMMARY (states, permitted actions
per transition); it never sees `when` tags, rule ids, gold states, or the
validator's hidden rule matching.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import TransitionResult, WorkflowConfig


def load_workflow(path: Path) -> WorkflowConfig:
    return WorkflowConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_decision(
    workflow: WorkflowConfig,
    current_state: str,
    decision_workflow_state: str,
    action_type: str | None,
    visible_tags: set[str],
) -> TransitionResult:
    violations: list[str] = []
    state_valid = decision_workflow_state in workflow.states
    if not state_valid:
        violations.append(f"unknown_state:{decision_workflow_state}")

    if decision_workflow_state == current_state:
        # staying in the same node: no transition rule needed, but the action
        # must still be available at the reported node
        outgoing = [t for t in workflow.transitions if t.from_state == decision_workflow_state]
        action_valid = True
        stay_violations = list(violations)
        if action_type is not None and outgoing:
            action_valid = any(action_type in t.allowed_actions for t in outgoing)
            if not action_valid:
                stay_violations.append(f"action_not_available_at:{decision_workflow_state}")
        return TransitionResult(
            state_valid=state_valid,
            action_valid=action_valid,
            transition_valid=state_valid,
            violations=stay_violations,
            matched_rule_ids=[],
        )

    candidates = [
        t for t in workflow.transitions
        if t.from_state == current_state and t.to_state == decision_workflow_state
    ]
    if not candidates:
        violations.append(f"no_transition:{current_state}->{decision_workflow_state}")
        return TransitionResult(state_valid=state_valid, action_valid=False, transition_valid=False,
                                violations=violations, matched_rule_ids=[])

    violations = [v for v in violations if not v.startswith("unknown_state")]
    transition_valid = False
    matched: list[str] = []
    for rule in candidates:
        missing = [tag for tag in rule.when if tag not in visible_tags]
        if missing:
            violations.append(f"missing_preconditions:{','.join(missing)}@{rule.rule_id}")
            continue
        transition_valid = True
        matched.append(rule.rule_id)

    # next_action is judged against the actions AVAILABLE AT the reported node
    # (outgoing transitions), not against the incoming transition's rule
    outgoing = [t for t in workflow.transitions if t.from_state == decision_workflow_state]
    if action_type is None or not outgoing:
        action_valid = True
    else:
        action_valid = any(action_type in t.allowed_actions for t in outgoing)
        if not action_valid:
            violations.append(f"action_not_available_at:{decision_workflow_state}")
    return TransitionResult(
        state_valid=state_valid,
        action_valid=action_valid,
        transition_valid=transition_valid and state_valid,
        violations=violations,
        matched_rule_ids=matched,
    )


def guidance_summary(workflow: WorkflowConfig) -> str:
    """Public, clinician-facing rendering: no when-tags, no rule ids, no gold."""
    lines = [f"诊疗流程节点（workflow_state 取值）：{' → '.join(workflow.states)}。"]
    lines.append(f"流程从 {workflow.start_state()} 开始；每轮报告你判断的当前节点，并从该节点可用的动作中选择下一步。")
    for t in workflow.transitions:
        if t.allowed_actions:
            lines.append(f"处于 {t.from_state}（通往 {t.to_state}）时可考虑的动作：{'、'.join(t.allowed_actions)}。")
        else:
            lines.append(f"{t.from_state} 之后可以进入 {t.to_state}。")
    lines.append("（节点迁移的完整前置条件由评测端校验；不要编造列表之外的节点名。workflow_state 是流程节点，clinical_stage 是疾病分期，两者不可混用。）")
    return "\n".join(lines)
