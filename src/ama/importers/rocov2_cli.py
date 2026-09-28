"""CLI adapter for the existing ROCOv2 importer."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from .rocov2 import import_rocov2


def configure_cli(parser: ArgumentParser) -> None:
    parser.add_argument("--split", default="test", help="source split (default: test)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append", dest="ids")


def import_from_cli(args: Namespace) -> dict[str, Any]:
    return import_rocov2(args.source, args.out, split=args.split, limit=args.limit, ids=args.ids)
