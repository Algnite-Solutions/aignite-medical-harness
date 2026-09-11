"""episode-folder importer: convert v0.2 prototype Episode Folders into AMA Dataset v0."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..dataset_card import write_dataset_cards


def _iso(dt: str | None) -> str | None:
    return dt


def import_episode_folders(source: Path, out: Path, dataset_name: str | None = None,
                           scorer: str = "workflow_v0") -> dict[str, Any]:
    """Each <source>/<episode_id>/ folder (episode.json + evidence.jsonl + workflow.json
    + gold.json) becomes one AMA episode. workflow.json → policy.json (public guidance
    summary built from states/actions; hidden transitions verbatim). gold.json → targets.jsonl."""
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    policies: dict[str, Any] | None = None
    evidence_tags: dict[str, list[str]] = {}
    skipped: list[str] = []

    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        try:
            ep_raw = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
            evidence = [json.loads(l) for l in (folder / ep_raw.get("evidence_file", "evidence.jsonl"))
                        .read_text(encoding="utf-8").splitlines() if l.strip()]
            by_id = {e["evidence_id"]: e for e in evidence}
            workflow = json.loads((folder / ep_raw.get("workflow_file", "workflow.json")).read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append(f"{folder.name}: {exc}")
            continue

        turns = []
        for t in ep_raw["turns"]:
            evs = []
            for eid in t.get("release_evidence_ids", []):
                e = by_id[eid]
                evs.append({
                    "evidence_id": e["evidence_id"],
                    "kind": e.get("modality", "unknown"),
                    "text": e.get("content", ""),
                    "artifact": None,
                    "source": e.get("source_locator", ""),
                    "metadata": {"patient_id": e.get("patient_id")},
                })
                if e.get("tags"):
                    evidence_tags[e["evidence_id"]] = e["tags"]
            turns.append({
                "turn_id": t["turn_id"],
                "time": t["as_of"],
                "message": t.get("world_message") or t.get("event", ""),
                "evidence": evs,
            })
        episodes.append({
            "episode_id": ep_raw["episode_id"],
            "subject_id": ep_raw["patient_id"],
            "metadata": {"source": "episode-folder", "folder": str(folder)},
            "turns": turns,
        })

        gold_path = folder / "gold.json"
        if gold_path.exists():
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            tmap = {}
            for tid, g in gold.get("turns", {}).items():
                state = {}
                if g.get("expected_workflow_state"):
                    state["workflow_state"] = g["expected_workflow_state"]
                if g.get("expected_hypotheses"):
                    state["hypotheses"] = g["expected_hypotheses"]
                if g.get("expected_stage"):
                    state["clinical_stage"] = g["expected_stage"]
                tmap[tid] = {
                    **({"state": state} if state else {}),
                    "allowed_actions": g.get("allowed_actions", []),
                    "forbidden_actions": g.get("forbidden_actions", []),
                    "required_evidence": g.get("required_evidence_refs", []),
                    "expect_abstain": g.get("expect_abstain", False),
                }
            targets.append({"episode_id": ep_raw["episode_id"], "turns": tmap})

        # one shared policy from the first workflow seen
        if policies is None:
            states = workflow.get("states", [])
            lines = [f"诊疗流程节点（decision.state.workflow_state 取值）：{' → '.join(states)}。"]
            for tr in workflow.get("transitions", []):
                if tr.get("allowed_actions"):
                    lines.append(f"处于 {tr['from']}（通往 {tr['to']}）时可考虑的动作：{'、'.join(tr['allowed_actions'])}。")
            lines.append("（迁移的完整前置条件由评测端校验；不要编造列表之外的节点名。）")
            policies = {
                "public": {"guidance": "\n".join(lines)},
                "hidden": {"initial_state": workflow.get("initial_state") or (states[0] if states else None),
                           "evidence_tags": evidence_tags,
                           "transitions": [
                               {"from": t["from"], "to": t["to"], "when": t.get("when", []),
                                "allowed_actions": t.get("allowed_actions", [])}
                               for t in workflow.get("transitions", [])
                           ]},
            }

    name = dataset_name or source.name
    dataset = {
        "schema": "ama-dataset-v0", "name": name, "version": "0.1",
        "description": f"imported from episode folders in {source}",
        "splits": {"all": [e["episode_id"] for e in episodes]},
        "scorer": scorer,
    }
    (out / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in episodes) + "\n", encoding="utf-8")
    if targets:
        (out / "targets.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in targets) + "\n", encoding="utf-8")
    if policies:
        policies["hidden"]["evidence_tags"] = evidence_tags
        (out / "policy.json").write_text(json.dumps(policies, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_cards(
        out,
        name=name,
        source=str(source),
        license_name="",
        purpose_en="Teacher-forced replay of imported single-turn or multi-turn clinical Episodes.",
        purpose_zh="用于导入的单轮或多轮临床 Episode 的 teacher-forced 回放。",
        construction_en="Legacy Episode folders are normalized into ordered Turns; released records become citable Evidence.",
        construction_zh="旧版 Episode folder 被规范化为有序 Turn，按轮释放的记录成为可引用 Evidence。",
        episodes=len(episodes),
        turns=sum(len(e["turns"]) for e in episodes),
        evidence=sum(len(t["evidence"]) for e in episodes for t in e["turns"]),
        scorer=scorer,
        targets=len(targets),
        limitations_en="Review source licensing, temporal assumptions, and generated public guidance before publication.",
        limitations_zh="发布前需要核对源数据许可、时间假设和自动生成的公开指导。",
    )
    return {"n_episodes": len(episodes), "n_targets": len(targets), "skipped": skipped,
            "policy": policies is not None}
