"""Check the shared-data entry point offline, including corrupted NAS copies."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ama.importers import medagentbench as med
from test_medagentbench import source, resources, snapshot


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_medagentbench.sh"


def command(*args, **settings):
    env = dict(os.environ, PYTHON_BIN=sys.executable, **settings)
    # Offline commands must not require API credentials or a personal config.
    env.pop("MEDAI_API_KEY", None)
    env.pop("QWEN_API_KEY", None)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          capture_output=True, text=True, timeout=20)


def test_help_without_data_or_keys():
    result = command("--help", AMA_DATA_PATH="/not-mounted", CONFIG="/no-config")
    assert result.returncode == 0
    assert "all does NOT import" in result.stdout


def test_import_and_validate_using_snapshot(source, snapshot, tmp_path):
    root = tmp_path / "nas"
    folder = root / "processed/MedAgentBench"
    result = command("import", AMA_DATA_PATH=str(root), SOURCE_DIR=str(source),
                     SNAPSHOT_DIR=str(snapshot), FHIR_BASE="")
    assert result.returncode == 0, result.stderr
    result = command("validate", AMA_DATA_PATH=str(root))
    assert result.returncode == 0, result.stderr
    assert "300 tasks, fixed smoke=30" in result.stdout
    # The same command must refuse to overwrite the shared data.
    result = command("import", AMA_DATA_PATH=str(root), SOURCE_DIR=str(source),
                     SNAPSHOT_DIR=str(snapshot), FHIR_BASE="")
    assert result.returncode == 1
    assert folder.is_dir()


@pytest.mark.parametrize("corruption", ["smoke", "snapshot", "facts"])
def test_validate_rejects_corrupt_data(source, snapshot, tmp_path, corruption):
    folder = tmp_path / "processed"
    med.import_medagentbench(source, folder, snapshot=snapshot)
    if corruption == "smoke":
        path = folder / "dataset.json"
        data = json.loads(path.read_text())
        data["splits"]["smoke"] = data["splits"]["smoke"][::-1]
        med._json(path, data)
    elif corruption == "snapshot":
        with (folder / "snapshot/Observation.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        med._jsonl(folder / "cases.jsonl", [])
    result = command("validate", DATASET_DIR=str(folder))
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
