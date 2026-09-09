import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pytest

from medical_harness.runner import load_inputs, load_gold

DATA = REPO / "data" / "sim"


@pytest.fixture(scope="session")
def inputs():
    return {c.case_id: c for c in load_inputs(DATA / "inputs")}


@pytest.fixture(scope="session")
def gold():
    return load_gold(DATA / "gold")
