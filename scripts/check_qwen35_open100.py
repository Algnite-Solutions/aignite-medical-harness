"""Offline cohort check: hashes and counts only; never constructs a Model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ama.data import load_dataset

VIEWS = ("hpi", "interactive", "full_info")


def check(cohort: Path, reference: Path) -> dict:
    result = {"status": "verified_offline", "cases": 100, "views": {}, "api_requests": 0}
    order = None
    targets = None
    for view in VIEWS:
        name = "mimic_cdm_open_" + view
        folder, original = cohort / name, reference / name
        dataset = load_dataset(folder)
        ids = [ep.id for ep in dataset.episodes]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f"{view}: expected 100 unique cases")
        if order is not None and ids != order:
            raise ValueError(f"{view}: case order differs")
        order = ids
        target_hash = hashlib.sha256((folder / "targets.jsonl").read_bytes()).hexdigest()
        if targets is not None and targets != target_hash:
            raise ValueError(f"{view}: targets differ")
        targets = target_hash
        hashes = {}
        for filename in ("dataset.json", "episodes.jsonl", "targets.jsonl", "instructions.txt", "eval.json"):
            data = (folder / filename).read_bytes()
            if data != (original / filename).read_bytes():
                raise ValueError(f"{view}: {filename} differs from the PR13 cohort")
            hashes[filename] = hashlib.sha256(data).hexdigest()
        result["views"][view] = {"cases": len(ids), "reference_files_byte_equal": True, "sha256": hashes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("runs/reviewer-open100"))
    parser.add_argument("--reference", type=Path, default=Path("runs/mimic-cdm-open-v1-100-seed42"))
    args = parser.parse_args()
    print(json.dumps(check(args.cohort, args.reference), indent=2))


if __name__ == "__main__":
    main()
