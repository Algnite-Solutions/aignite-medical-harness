"""CLI adapter for the existing MIMIC-CXR importer."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from .mimic_cxr import import_mimic_cxr


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default="validate",
        choices=["train", "validate", "test"],
        help="source split (default: validate)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--study-id", action="append", dest="ids")


def import_from_cli(args: Namespace) -> dict[str, Any]:
    return import_mimic_cxr(
        args.source,
        args.out,
        split=args.split,
        limit=args.limit,
        ids=args.ids,
    )
