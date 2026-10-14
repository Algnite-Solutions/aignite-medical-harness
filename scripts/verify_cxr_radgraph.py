"""Compare AMA graph F1 with pinned official functions on real report pairs.

Pairs JSONL: {"id": ..., "reference": ..., "prediction": ...} (keep local).
Only verified official function definitions are loaded; heavy legacy imports are omitted.
"""
import argparse
import ast
import json
from pathlib import Path

from ama.cxr_radgraph import Extractor, combined_f1
from ama.importers.mimic_cxr import digest, json_file, staging

OFFICIAL_SHA256 = "b8d685a00c5a41746ea3bd61edd59a3b554cc4c68b2e14faa21b33bd35b2410f"


def verify(pairs_path: Path, official_code: Path, out: Path):
    if digest(official_code) != OFFICIAL_SHA256:
        raise ValueError("official metric source hash mismatch")
    tree = ast.parse(official_code.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"compute_f1", "parse_entity_relation"}]
    if len(functions) != 2:
        raise ValueError("official metric functions missing")
    namespace = {"json": json, "tqdm": lambda value: value}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(official_code), "exec"), namespace)
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if not pairs or len({row["id"] for row in pairs}) != len(pairs):
        raise ValueError("empty or duplicate report pairs")
    if any(not isinstance(row.get(k), str) or not row[k].strip()
           for row in pairs for k in ("reference", "prediction")):
        raise ValueError("verification requires nonempty real report pairs")
    extractor = Extractor()
    texts = [row[k] for row in pairs for k in ("reference", "prediction")]
    with staging(out) as stage:
        graphs = extractor(texts)
        annotation_file = stage / "annotations.json"
        json_file(annotation_file, {str(i): g for i, g in enumerate(graphs)})
        entities, relations = namespace["parse_entity_relation"](annotation_file)
        per = []
        for i, row in enumerate(pairs):
            a, b = str(2 * i), str(2 * i + 1)
            official_entity = namespace["compute_f1"](entities[a], entities[b])
            official_relation = namespace["compute_f1"](relations[a], relations[b])
            ours = combined_f1(graphs[2 * i], graphs[2 * i + 1])
            difference = max(abs(ours["entity_f1"] - official_entity),
                             abs(ours["relation_f1"] - official_relation),
                             abs(ours["radgraph_combined"] - (official_entity + official_relation) / 2))
            per.append({"id": row["id"], **ours, "max_abs_difference": difference})
        maximum = max(row["max_abs_difference"] for row in per)
        if maximum > 1e-12:
            raise RuntimeError("official metric parity failed")
        summary = {"pairs": len(pairs), "max_abs_difference": maximum,
                   "pairs_sha256": digest(pairs_path), "official_code_sha256": digest(official_code),
                   "extractor": extractor.metadata,
                   "scope": "metric functions on shared annotations; legacy inference not reproduced",
                   "per_pair": per}
        json_file(stage / "verification.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.pairs, args.official_code, args.out), indent=2))


if __name__ == "__main__":
    main()
