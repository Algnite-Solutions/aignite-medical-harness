"""Freeze official 100-study cohort or materialize it after mounting the image disk."""
import argparse
import json
from pathlib import Path

from ama.importers.mimic_cxr_rexrank import freeze_cohort, materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "materialize"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_cohort(args.source, args.out)
    else:
        if args.cohort is None or args.image_root is None:
            parser.error("materialize requires --cohort and --image-root")
        result = materialize(args.source, args.cohort, args.image_root, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
