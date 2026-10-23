"""Offline checks for exporting an existing frozen cohort."""
import json
import shutil

import pytest

from ama.data import load_dataset, validate_dataset
from ama.importers.vqa_rad import _sha
from scripts.export_vqa_rad_cohort import export_cohort
from test_vqa_rad_benchmark import cohort_fixture


def test_export_is_exact_and_portable(tmp_path):
    source, root, cohort = cohort_fixture(tmp_path)
    original = (root / 'cohort.json').read_bytes()
    out = tmp_path / 'export'
    export_cohort(root, out)
    for name in ('episodes.jsonl', 'targets.jsonl', 'provenance.jsonl'):
        records = {r['id']: r for r in map(json.loads, (source / name).read_text().splitlines())}
        assert list(map(json.loads, (out / name).read_text().splitlines())) == [
            records[eid] for eid in cohort['episode_ids']]
    assert (root / 'cohort.json').read_bytes() == original
    relocated = tmp_path / 'nas' / 'dataset'
    relocated.parent.mkdir()
    shutil.move(out, relocated)
    shutil.rmtree(source)
    assert validate_dataset(relocated) == []
    ds = load_dataset(relocated, with_targets=True)
    assert ds.info.splits == {'test': cohort['episode_ids']}
    assert len(ds.episodes) == len(ds.targets) == 3
    assert not load_dataset(relocated, with_targets=False).targets
    receipt = json.loads((relocated / 'source_cohort.json').read_text())
    assert 'dataset_dir' not in receipt
    for name, digest in json.loads((relocated / 'checksums.json').read_text())['files'].items():
        assert _sha(relocated / name) == digest
    with pytest.raises(FileExistsError):
        export_cohort(root, relocated)


def test_export_rejects_changed_frozen_inputs(tmp_path):
    source, root, _ = cohort_fixture(tmp_path)
    (source / 'instructions.txt').write_text('changed')
    out = tmp_path / 'export'
    with pytest.raises(ValueError):
        export_cohort(root, out)
    assert not out.exists()
