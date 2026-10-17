"""CLI adapter for the existing MIMIC-CXR importer."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from .mimic_cxr import import_mimic_cxr, reuse_mimic_cxr


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default="validate",
        choices=["train", "validate", "test"],
        help="source split (default: validate)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--study-id", action="append", dest="ids")
    parser.add_argument("--source-format", choices=["raw", "ama"], default="raw",
                        help="standalone raw package, or an existing frozen AMA CXR dataset")
    parser.add_argument("--report-mode", choices=["findings-impression", "findings-only"],
                        default="findings-impression")


def import_from_cli(args: Namespace) -> dict[str, Any]:
    importer = reuse_mimic_cxr if args.source_format == "ama" else import_mimic_cxr
    return importer(
        args.source,
        args.out,
        split=args.split,
        limit=args.limit,
        ids=args.ids,
        report_mode=args.report_mode,
    )
