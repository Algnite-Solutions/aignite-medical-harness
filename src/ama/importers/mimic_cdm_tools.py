"""Trusted AMA tool adapter for the MIMIC-CDM interactive dataset.

Pass this file with ``ama run ... --tools PATH``. It binds a fresh set of tools
to the current episode; case data are cached for the lifetime of the process.
"""
from functools import lru_cache
from pathlib import Path

from ama.importers.mimic_cdm import load_cases, make_case_tools


@lru_cache(maxsize=2)
def _dataset(path: str):
    return load_cases(Path(path))


def make_tools(dataset_dir: Path, episode_id: str):
    if dataset_dir is None or episode_id is None:
        raise ValueError("MIMIC-CDM tools require an active dataset episode")
    return make_case_tools(_dataset(str(Path(dataset_dir).resolve())), episode_id)
