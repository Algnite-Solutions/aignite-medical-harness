"""ReXrank-style entity/relation combined F1, with lazy RadGraph inference.

Formula verified against ReXrank-metric d6be67831ef933dc91f01892303a40ebc6cc9549.
Modern inference package equivalence to the legacy extractor is a separate check.
"""
from __future__ import annotations

import math
import os
import hashlib
from importlib.metadata import version
from pathlib import Path

from .importers.mimic_cxr import digest

SCORER = "mimic_cxr_radgraph_v1"
METRIC_REVISION = "d6be67831ef933dc91f01892303a40ebc6cc9549"
WEIGHTS_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"
WEIGHTS_SHA256 = "34a529d7ffee36ad9196a0ebd84a6c911948a2c5ba9f3d8c1024c4d264719cbc"
CONFIG_SHA256 = "602ecdc439e0292e137b505e28b9e3c96a838e0240a0e1280bbabb5bd1e8bc68"
ENCODER_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
ENCODER_FILES = {"config.json": "f8a8cd37533eee7ab2f72c09890b360824352953",
                 "tokenizer_config.json": "a661b1a138dac6dc5590367402d100765010ffd6",
                 "vocab.txt": "9d595d9c20feef7012f174efaaa5eb621910588e"}
ENCODER_ALIAS = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def git_blob_digest(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def graph_sets(annotation: dict) -> tuple[set, set]:
    entities = annotation["entities"]
    nodes = {key: (value["tokens"], value["label"]) for key, value in entities.items()}
    edges = {(nodes[key], nodes[target], relation)
             for key, value in entities.items() for relation, target in value["relations"]}
    return set(nodes.values()), edges


def set_f1(reference: set, prediction: set) -> float:
    # Official compute_f1 assigns 0 when both sets are empty.
    denominator = len(reference) + len(prediction)
    return 2 * len(reference & prediction) / denominator if denominator else 0.0


def combined_f1(reference: dict, prediction: dict) -> dict:
    ref_entities, ref_relations = graph_sets(reference)
    pred_entities, pred_relations = graph_sets(prediction)
    entity = set_f1(ref_entities, pred_entities)
    relation = set_f1(ref_relations, pred_relations)
    return {"entity_f1": entity, "relation_f1": relation,
            "radgraph_combined": (entity + relation) / 2}


def prediction_report(answer) -> str | None:
    if not isinstance(answer, dict):
        return None
    findings, impression = answer.get("findings"), answer.get("impression")
    if not isinstance(findings, str) or not isinstance(impression, str):
        return None
    if not (findings.strip() or impression.strip()):
        return None
    return f"Findings: {findings} Impression: {impression}"


class Extractor:
    def __init__(self):
        if version("radgraph") != "0.1.18":
            raise RuntimeError("install the pinned ama[cxr-radgraph] dependencies")
        cache = Path(os.environ.get("AMA_RADGRAPH_CACHE", "local_data/radgraph"))
        hub = cache / "encoder-hub"
        snapshot = hub / ("models--" + ENCODER_ALIAS.replace("/", "--")) / "snapshots" / ENCODER_REVISION
        for name, checksum in ENCODER_FILES.items():
            path = snapshot / name
            if not path.is_file() or git_blob_digest(path.read_bytes()) != checksum:
                raise RuntimeError("missing or modified encoder config/tokenizer; run scripts/setup_cxr_radgraph.py")
        # The official wrapper takes a repository name, not a revision. Feed it
        # a verified offline snapshot under its original repository alias.
        os.environ["HF_HUB_CACHE"] = str(hub.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        from radgraph import RadGraph
        weights = cache / "radgraph" / "weights.th"
        config = cache / "radgraph" / "config.json"
        if not weights.is_file() or digest(weights) != WEIGHTS_SHA256:
            raise RuntimeError("missing or modified pinned RadGraph weights; run scripts/setup_cxr_radgraph.py")
        if not config.is_file() or digest(config) != CONFIG_SHA256:
            raise RuntimeError("missing or modified pinned RadGraph model config")
        self.model = RadGraph(model_type="radgraph", cuda=-1, batch_size=1,
                              model_cache_dir=str(cache), tokenizer_cache_dir=str(hub))
        self.metadata = {"package": "radgraph", "package_version": version("radgraph"),
                         "model_type": "radgraph", "device": "cpu",
                         "weights_sha256": digest(weights), "config_sha256": digest(config),
                         "weights_repository_revision": WEIGHTS_REVISION,
                         "encoder_revision": ENCODER_REVISION, "encoder_files": ENCODER_FILES,
                         "transformers_version": version("transformers"),
                         "torch_version": version("torch"),
                         "metric_revision": METRIC_REVISION,
                         "legacy_extractor_equivalence": "not_verified"}

    def __call__(self, texts: list[str]) -> list[dict]:
        result = self.model(texts)
        if set(result) != {str(i) for i in range(len(texts))}:
            raise RuntimeError("RadGraph returned incomplete annotations")
        return [result[str(i)] for i in range(len(texts))]


def score_mimic_cxr_radgraph(dataset, decisions, *, extractor=None) -> dict:
    from .scorer import expected_rows

    pairs = []
    for ep in dataset.episodes:
        if len(ep.turns) != 1:
            raise ValueError("CXR v1 requires one turn per study")
        target = dataset.targets.get(ep.id)
        payload = target.turns.get(ep.turns[0].id) if target else None
        if not payload or not isinstance(payload.get("report"), str) or not payload["report"].strip():
            raise ValueError("CXR RadGraph requires an authoritative nonempty target report")
        rows = decisions.get(ep.id, [])
        if len(rows) > 1 or any(row["turn_id"] != ep.turns[0].id for row in rows):
            raise ValueError("duplicate or unexpected CXR decision")
        row = expected_rows(ep, decisions)[0]
        decision = row.get("decision") or {}
        pairs.append((ep.id, payload["report"], prediction_report(decision.get("answer")), row))
    if not pairs:
        raise ValueError("no CXR studies to score")
    valid = [pair for pair in pairs if pair[2] is not None]
    metadata = {"metric_revision": METRIC_REVISION, "inference": "not_needed_no_valid_reports"}
    annotations = {}
    if valid:
        extractor = extractor or Extractor()
        texts = [text for _, ref, pred, _ in valid for text in (ref, pred)]
        graphs = extractor(texts)
        if len(graphs) != len(texts):
            raise RuntimeError("RadGraph annotation count mismatch")
        metadata = extractor.metadata
        for i, pair in enumerate(valid):
            annotations[pair[0]] = combined_f1(graphs[2 * i], graphs[2 * i + 1])
    per = {}
    for eid, _, pred, row in pairs:
        score = annotations.get(eid, {"entity_f1": 0.0, "relation_f1": 0.0, "radgraph_combined": 0.0})
        if any(not math.isfinite(value) for value in score.values()):
            raise RuntimeError("nonfinite RadGraph score")
        per[eid] = {**score, "report_valid": pred is not None, "termination": row["termination"]}
    total = sum(row["radgraph_combined"] for row in per.values())
    return {"scorer": SCORER, "metric": metadata, "per_episode": per, "aggregate": {
        "expected_studies": len(pairs), "valid_reports": len(valid),
        "failed_or_invalid_reports": len(pairs) - len(valid),
        "radgraph_combined_macro": total / len(pairs),
        "valid_report_radgraph_combined_macro": total / len(valid) if valid else None,
        "failure_policy": "zero; retain every expected study in denominator"}}
