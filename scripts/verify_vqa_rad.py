#!/usr/bin/env python3
"""Compare a local AMA conversion to source rows, pixels and file checksums.

Offline only. Oracle/empty-answer checks exercise the scorer; they do not
measure a model. Output contains counts, versions and overall hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from ama.data import load_dataset, validate_dataset
from ama.importers.vqa_rad import REPO, REVISION
from ama.importers.vqa_rad_scoring import score_vqa_rad


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(source: Path, dataset: Path) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image

    manifest = json.loads((source / "source-manifest.json").read_text())
    require(manifest["repo_id"] == REPO and manifest["revision"] == REVISION, "wrong source version")
    for entry in manifest["files"]:
        path = source / "snapshot" / entry["path"]
        require((source / "snapshot").resolve() in path.resolve().parents, "unsafe source path")
        data = path.read_bytes()
        require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                "source checksum mismatch")
    checksums = json.loads((dataset / "checksums.json").read_text())
    actual = {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dataset.rglob("*")) if p.is_file() and p != dataset / "checksums.json"}
    require(actual == checksums["files"], "generated file checksum mismatch")
    overall = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(overall == checksums["overall_sha256"], "overall checksum mismatch")
    require(not validate_dataset(dataset), "invalid AMA structure")
    ds = load_dataset(dataset, with_targets=True)
    public = load_dataset(dataset)
    require(not public.targets and public.eval_config is None, "run loader exposed evaluator files")
    report = json.loads((dataset / "import_report.json").read_text())
    provenance = [json.loads(line) for line in (dataset / "provenance.jsonl").read_text().splitlines()]
    require(len(provenance) == len(ds.episodes) == len(ds.targets), "episode/target/provenance count mismatch")
    require(len({row["id"] for row in provenance}) == len(provenance), "duplicate provenance")
    require(set(ds.targets) == {ep.id for ep in ds.episodes}, "target membership mismatch")
    require(set(ds.info.splits) == {"train", "test"}, "source splits missing")
    require(set(ds.info.splits["train"]).isdisjoint(ds.info.splits["test"]), "split membership overlaps")
    originals = {}
    for split in ("train", "test"):
        originals[split] = []
        for path in sorted((source / "snapshot/data").glob(f"{split}-*.parquet")):
            originals[split].extend(pq.read_table(path).to_pylist())
        require(len(originals[split]) == report["source_counts"][split], "source count mismatch")
        selection = report["selection"]
        if selection["subset"] == "full":
            indices = list(range(len(originals[split])))
        else:
            require(selection["subset"] == "smoke", "unknown selection")
            indices = sorted(random.Random(selection["seed"]).sample(
                range(len(originals[split])), min(selection["count"], len(originals[split])))) if split == "test" else []
        require(indices == selection["selected_row_indices"][split], "selection cannot be reconstructed")
        require(ds.info.splits[split] == [f"vqarad-{split}-{i:06d}" for i in indices], "split/order mismatch")
    require([row["id"] for row in provenance] == [ep.id for ep in ds.episodes], "episode order mismatch")
    require([ep.id for ep in ds.episodes] == ds.info.splits["train"] + ds.info.splits["test"],
            "episode order does not follow source splits")
    episodes = {ep.id: ep for ep in ds.episodes}
    for row in provenance:
        original = originals[row["split"]][row["row_index"]]
        require(row["id"] == f"vqarad-{row['split']}-{row['row_index']:06d}", "source identity mismatch")
        ep = episodes[row["id"]]
        require(len(ep.turns) == 1 and ep.turns[0].id == "t1", "not a single-turn episode")
        turn = ep.turns[0]
        require(turn.observation is None, "unexpected observation")
        require([ev.id for ev in turn.evidence] == ["image", "question"], "unexpected evidence")
        image, question = turn.evidence
        require(question.text == original["question"] and question.file is None, "question changed")
        require(image.text is None and image.type == "image", "unexpected image text")
        target = ds.targets[ep.id].turns["t1"]
        require(target["answer"] == {"text": original["answer"].strip()}, "reference answer changed")
        kind = "closed_yes_no" if original["answer"].strip().lower() in {"yes", "no"} else "open"
        require(target["question_type"] == kind, "staged question type changed")
        raw = original["image"]["bytes"]
        require(hashlib.sha256(raw).hexdigest() == row["image_sha256"], "source image identity mismatch")
        with Image.open(io.BytesIO(raw)) as a, Image.open(dataset / image.file) as b:
            a.load(); b.load()
            if a.mode not in ("RGB", "L"):
                a = a.convert("RGB")
            require(a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes(),
                    "converted pixels differ from source")
    oracle = {ep.id: [{"turn_id": "t1", "decision": {"answer": ds.targets[ep.id].turns["t1"]["answer"]},
                      "termination": "completed", "model_calls": 0, "usage": None, "duration_ms": 0}]
              for ep in ds.episodes}
    perfect = score_vqa_rad(ds, oracle)["aggregate"]["accuracy_heuristic"]["value"]
    empty = score_vqa_rad(ds, {})["aggregate"]["accuracy_heuristic"]["value"]
    require(perfect == 1 and empty == 0, "oracle/empty scorer sanity check failed")
    return {"offline_validation": "passed", "revision": REVISION,
            "subset": report["selection"]["subset"], "episodes": len(ds.episodes),
            "split_counts": {key: len(value) for key, value in ds.info.splits.items()},
            "artifacts": len(list((dataset / "artifacts").glob("*.png"))),
            "overall_sha256": overall, "oracle_sanity_score": perfect,
            "empty_sanity_score": empty, "model_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.dataset), indent=2))


if __name__ == "__main__":
    main()
