"""Importers: offline conversions into AMA Dataset. The runner never knows
where data came from. Each importer emits a dataset directory + adapter report."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple


class ImportCommand(NamedTuple):
    name: str
    configure: Callable[[ArgumentParser], None]
    run: Callable[[Namespace], dict[str, Any]]


def register_importers(subparsers: Any) -> None:
    """Register each importer's arguments and handler in one explicit catalog."""
    from . import rocov2_cli, symptom2disease

    commands = (
        ImportCommand("rocov2", rocov2_cli.configure_cli, rocov2_cli.import_from_cli),
        ImportCommand("symptom2disease", symptom2disease.configure_cli,
                      symptom2disease.import_from_cli),
    )
    for command in commands:
        parser = subparsers.add_parser(command.name)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        command.configure(parser)
        parser.set_defaults(import_handler=command.run)
