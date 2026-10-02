#!/usr/bin/env python3
"""Prepare a standalone MIMIC-CXR raw package; never calls a model."""
import argparse
import json
import zipfile
from pathlib import Path

from ama.importers.mimic_cxr_raw import prepare_raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validate', 'test'], default='validate')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--study-id', action='append', dest='study_ids')
    args = parser.parse_args(argv)
    try:
        report = prepare_raw(args.source, args.out, args.split, args.limit, args.study_ids)
    except (OSError, ValueError, KeyError, ImportError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    # Individual identifiers and excluded case origins remain in the local report.
    print(json.dumps({k: v for k, v in report.items() if k not in
                      {'selected_study_ids', 'excluded_provenance', 'source_sha256'}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
