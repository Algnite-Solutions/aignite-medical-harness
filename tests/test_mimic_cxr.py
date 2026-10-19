"""All fixtures are synthetic; no real MIMIC case material belongs in Git."""
import csv
import gzip
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from ama.cli import main
from ama.data import load_dataset, validate_dataset
from ama.importers.mimic_cxr import digest, import_mimic_cxr, parse_sections, reuse_mimic_cxr
from ama.importers.mimic_cxr_raw import prepare_raw


def csv_gz(path, fields, rows):
    with gzip.open(path, 'wt', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_source(tmp_path):
    root = tmp_path / 'synthetic-source'
    root.mkdir()
    split_rows, meta = [], []
    studies = [('10', ['b', 'a'], 'validate'), ('20', ['c'], 'validate'),
               ('30', ['d'], 'validate'), ('40', ['e'], 'validate'), ('50', ['f'], 'test')]
    for sid, images, split in studies:
        for iid in images:
            row = {'subject_id': '10000001', 'study_id': sid, 'dicom_id': iid}
            split_rows.append({**row, 'split': split})
            meta.append({**row, 'ViewPosition': 'AP' if iid != 'a' else '', 'StudyDate': '21000101', 'StudyTime': '120000'})
            path = root / f'files/p10/p10000001/s{sid}/{iid}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), 'white').save(path)
    csv_gz(root / 'mimic-cxr-2.0.0-split.csv.gz', list(split_rows[0]), split_rows)
    csv_gz(root / 'mimic-cxr-2.0.0-metadata.csv.gz', list(meta[0]), meta)
    with zipfile.ZipFile(root / 'mimic-cxr-reports.zip', 'w') as z:
        for sid, _, _ in studies:
            report = 'Findings: SYNTHETIC_FINDINGS\nImpression: SYNTHETIC_IMPRESSION' if sid != '30' else 'IMPRESSION: SYNTHETIC_ONLY'
            z.writestr(f'files/p10/p10000001/s{sid}.txt', report)
    return root


@pytest.mark.parametrize('report,expected', [
    ('findings: A\nB\nImPrEsSiOn: C', {'findings': 'A B', 'impression': 'C'}),
    ('FINDINGS\n A\nIMPRESSION\n B', {'findings': 'A', 'impression': 'B'}),
    ('FINDINGS: A\nHISTORY: hidden\nFINDINGS: B\nIMPRESSION: C\nIMPRESSION: D', {'findings': 'A B', 'impression': 'C D'}),
    ('FINDINGS: \nIMPRESSION: C', {'findings': '', 'impression': 'C'}),
    ('full report\n\nlast paragraph', {'findings': '', 'impression': ''}),
    ('CONCLUSION: C\nCHEST: A', {'findings': '', 'impression': ''}),
    ('FINDINGS:\r\n A\r\nIMPRESSION:\r\nB', {'findings': 'A', 'impression': 'B'}),
])
def test_sections(report, expected):
    assert parse_sections(report) == expected


def test_raw_selection_hashes_and_standalone(tmp_path):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    report = prepare_raw(source, raw)
    assert report['selected_study_ids'] == ['10', '20', '40']
    assert report['n_images'] == 4
    assert report['excluded'] == {'missing_findings': 1}
    assert report['n_not_selected_due_limit'] == 0
    rows = [json.loads(line) for line in (raw / 'manifest.jsonl').read_text().splitlines()]
    assert [i['dicom_id'] for i in rows[0]['images']] == ['a', 'b']
    assert 'view_position' not in rows[0]['images'][0]
    for row in rows:
        assert digest(raw / row['report']) == row['report_sha256']
        for item in row['images']:
            assert digest(raw / item['file']) == item['sha256']
            with Image.open(raw / item['file']) as im:
                im.load()
    # Rename the entire incoming fixture: importer must not access it.
    source.rename(tmp_path / 'unavailable-incoming')
    out = tmp_path / 'processed'
    result = import_mimic_cxr(raw, out)
    assert result['n_imported'] == 3 and result['n_images'] == 4
    assert validate_dataset(out) == []
    ds = load_dataset(out, with_targets=True)
    assert ds.info.splits == {'validate': ['mimic-cxr-s10', 'mimic-cxr-s20', 'mimic-cxr-s40']}
    assert ds.eval_config.scorer == 'unscored'
    visible = (out / 'episodes.jsonl').read_text() + (out / 'instructions.txt').read_text()
    assert 'SYNTHETIC_FINDINGS' not in visible and 'SYNTHETIC_IMPRESSION' not in visible
    assert '21000101' not in visible and 'subject_id' not in visible
    assert ds.targets['mimic-cxr-s10'].turns['t1']['answer']['findings'] == 'SYNTHETIC_FINDINGS'
    ev = ds.episodes[0].turns[0].evidence
    assert ev[0].text is None and ev[1].text == 'ViewPosition: AP'
    assert (out / 'DATASET_CARD.zh-CN.md').exists()


@pytest.mark.parametrize('sid,split', [('999', 'validate'), ('50', 'validate'), ('30', 'validate')])
def test_raw_explicit_invalid(tmp_path, sid, split):
    source = synthetic_source(tmp_path)
    out = tmp_path / 'raw'
    with pytest.raises(ValueError):
        prepare_raw(source, out, split, study_ids=[sid])
    assert not out.exists()


@pytest.mark.parametrize('kind', ['missing', 'corrupt'])
def test_raw_bad_image_aborts(tmp_path, kind):
    source = synthetic_source(tmp_path)
    path = source / 'files/p10/p10000001/s10/a.jpg'
    if kind == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'not jpeg')
    out = tmp_path / 'raw'
    with pytest.raises((OSError, ValueError)):
        prepare_raw(source, out)
    assert not out.exists() and not list(tmp_path.glob('.mimic-cxr-*'))


@pytest.mark.parametrize('kind', ['duplicate', 'association', 'missing_metadata'])
def test_raw_bad_csv(tmp_path, kind):
    source = synthetic_source(tmp_path)
    path = source / 'mimic-cxr-2.0.0-metadata.csv.gz'
    with gzip.open(path, 'rt') as fh:
        rows = list(csv.DictReader(fh))
    if kind == 'duplicate':
        rows.append(rows[0])
    elif kind == 'association':
        rows[0]['subject_id'] = '999'
    else:
        rows.pop()
    csv_gz(path, list(rows[0]), rows)
    with pytest.raises(ValueError):
        prepare_raw(source, tmp_path / 'raw')


def test_split_filter_limit_and_cli(tmp_path, capsys):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    report = prepare_raw(source, raw, split='test', limit=1, study_ids=['50'])
    assert report['n_selected'] == 1 and report['n_after_id_filter'] == 1
    out = tmp_path / 'out'
    assert main(['import', 'mimic-cxr', '--source', str(raw), '--out', str(out), '--split', 'test', '--study-id', '50']) == 0
    assert load_dataset(out).info.splits == {'test': ['mimic-cxr-s50']}
    assert main(['import', 'mimic-cxr', '--source', str(raw), '--out', str(tmp_path / 'bad'), '--study-id', '999']) == 1
    assert not (tmp_path / 'bad').exists()


def test_existing_output_protection(tmp_path):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw)
    before = (raw / 'manifest.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        prepare_raw(source, raw)
    assert (raw / 'manifest.jsonl').read_bytes() == before
    out = tmp_path / 'out'
    import_mimic_cxr(raw, out)
    before = (out / 'targets.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        import_mimic_cxr(raw, out)
    assert (out / 'targets.jsonl').read_bytes() == before


def test_validator_failure_cleans_staging(tmp_path, monkeypatch):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw)
    monkeypatch.setattr('ama.importers.mimic_cxr.validate_dataset', lambda path: ['synthetic validation failure'])
    with pytest.raises(ValueError, match='synthetic validation failure'):
        import_mimic_cxr(raw, tmp_path / 'out')
    assert not (tmp_path / 'out').exists() and not list(tmp_path.glob('.mimic-cxr-*'))


@pytest.mark.parametrize('kind', ['duplicate_study', 'duplicate_image', 'unsafe', 'hash', 'report', 'missing'])
def test_import_invalid_raw(tmp_path, kind):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw)
    path = raw / 'manifest.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == 'duplicate_study':
        rows.append(rows[0])
    elif kind == 'duplicate_image':
        rows[0]['images'].append(rows[0]['images'][0])
    elif kind == 'unsafe':
        rows[0]['images'][0]['file'] = '../outside.jpg'
    elif kind == 'hash':
        rows[0]['images'][0]['sha256'] = 'wrong'
    elif kind == 'report':
        (raw / rows[0]['report']).write_text('IMPRESSION: only')
        rows[0]['report_sha256'] = digest(raw / rows[0]['report'])
    else:
        (raw / rows[0]['images'][0]['file']).unlink()
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises((ValueError, OSError)):
        import_mimic_cxr(raw, tmp_path / 'out')
    assert not (tmp_path / 'out').exists() and not list(tmp_path.glob('.mimic-cxr-*'))


@pytest.mark.parametrize('kind', ['missing_report', 'wrong_subject', 'duplicate_report', 'duplicate_split'])
def test_raw_report_and_split_integrity(tmp_path, kind):
    source = synthetic_source(tmp_path)
    if kind == 'duplicate_split':
        path = source / 'mimic-cxr-2.0.0-split.csv.gz'
        with gzip.open(path, 'rt') as fh:
            rows = list(csv.DictReader(fh))
        rows.append(rows[0])
        csv_gz(path, list(rows[0]), rows)
    else:
        path = source / 'mimic-cxr-reports.zip'
        with zipfile.ZipFile(path) as z:
            reports = {n: z.read(n) for n in z.namelist()}
        name = 'files/p10/p10000001/s10.txt'
        body = reports.pop(name)
        if kind == 'wrong_subject':
            reports['files/p10/p10000002/s10.txt'] = body
        elif kind == 'duplicate_report':
            reports[name] = body
            reports['other/' + name] = body
        with zipfile.ZipFile(path, 'w') as z:
            for n, content in reports.items():
                z.writestr(n, content)
    out = tmp_path / 'raw'
    if kind == 'missing_report':
        result = prepare_raw(source, out)
        assert result['excluded']['missing_report'] == 1
        with pytest.raises(ValueError, match='lacks material'):
            prepare_raw(source, tmp_path / 'explicit', study_ids=['10'])
    else:
        with pytest.raises(ValueError):
            prepare_raw(source, out)
        assert not out.exists()


def test_explicit_id_limit_and_duplicates(tmp_path):
    source = synthetic_source(tmp_path)
    for ids, limit in [(['10', '20'], 1), (['10', '10'], 3)]:
        with pytest.raises(ValueError):
            prepare_raw(source, tmp_path / 'bad', limit=limit, study_ids=ids)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw)
    with pytest.raises(ValueError, match='duplicate'):
        import_mimic_cxr(raw, tmp_path / 'out', ids=['10', '10'])
    result = import_mimic_cxr(raw, tmp_path / 'limited', ids=['10', '40'], limit=1)
    assert result['selected_study_ids'] == ['10']
    assert result['n_not_selected_due_id_filter'] == 1
    assert result['n_not_selected_due_limit'] == 1


@pytest.mark.parametrize('kind', ['corrupt_image', 'symlink', 'report_hash'])
def test_raw_package_damage(tmp_path, kind):
    source = synthetic_source(tmp_path)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw)
    manifest = raw / 'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    path = raw / rows[0]['images'][0]['file']
    if kind == 'corrupt_image':
        path.write_bytes(b'corrupt')
        rows[0]['images'][0]['sha256'] = digest(path)
    elif kind == 'symlink':
        replacement = raw / 'image-copy.jpg'
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    else:
        (raw / rows[0]['report']).write_text('FINDINGS: tampered\nIMPRESSION: tampered')
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises((ValueError, OSError)):
        import_mimic_cxr(raw, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_findings_only_without_impression_and_cli(tmp_path):
    source = synthetic_source(tmp_path)
    archive = source / 'mimic-cxr-reports.zip'
    with zipfile.ZipFile(archive) as z:
        reports = {n: z.read(n) for n in z.namelist()}
    reports['files/p10/p10000001/s10.txt'] = b'FINDINGS: SYNTHETIC_FINDINGS_ONLY'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, body in reports.items():
            z.writestr(name, body)
    raw = tmp_path / 'raw'
    prepare_raw(source, raw, study_ids=['10'], limit=1, report_mode='findings-only')
    with pytest.raises(ValueError, match='required report sections'):
        import_mimic_cxr(raw, tmp_path / 'default')
    out = tmp_path / 'findings'
    assert main(['import', 'mimic-cxr', '--source', str(raw), '--out', str(out),
                 '--report-mode', 'findings-only']) == 0
    ds = load_dataset(out, with_targets=True)
    from ama.importers.mimic_cxr import FINDINGS_INSTRUCTIONS
    assert (out / 'instructions.txt').read_text() == FINDINGS_INSTRUCTIONS
    assert ds.targets['mimic-cxr-s10'].turns['t1']['answer'] == {'findings': 'SYNTHETIC_FINDINGS_ONLY'}
    report = json.loads((out / 'import_report.json').read_text())
    assert report['task']['task_id'] == 'mimic-cxr-report-task'
    assert report['task']['official_cohort_reproduced'] is False


def frozen_dataset(tmp_path):
    source = synthetic_source(tmp_path)
    raw, frozen = tmp_path / 'raw', tmp_path / 'frozen'
    prepare_raw(source, raw)
    import_mimic_cxr(raw, frozen)
    # The reused structured sections must win over any flattened full report.
    rows = [json.loads(s) for s in (frozen / 'targets.jsonl').read_text().splitlines()]
    for row in rows:
        row['turns']['t1']['report'] = 'SYNTHETIC_FULL_REPORT_NOT_A_FINDINGS_REFERENCE'
    (frozen / 'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return frozen


def test_reuse_frozen_cohort_and_native_run(tmp_path, capsys):
    from ama.runner import execute
    source = frozen_dataset(tmp_path)
    source_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    out = tmp_path / 'reused'
    assert main(['import', 'mimic-cxr', '--source', str(source), '--out', str(out),
                 '--source-format', 'ama', '--report-mode', 'findings-only']) == 0
    original, reused = load_dataset(source, True), load_dataset(out, True)
    assert reused.info.splits == original.info.splits
    for before, after in zip(original.episodes, reused.episodes):
        assert before.id == after.id
        for old, new in zip(before.turns[0].evidence, after.turns[0].evidence):
            assert (old.id, old.text) == (new.id, new.text)
            assert digest(source / old.file) == digest(out / new.file)
        assert reused.targets[after.id].turns['t1']['answer'] == {'findings': 'SYNTHETIC_FINDINGS'}
    assert source_hashes == {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    report = json.loads((out / 'import_report.json').read_text())
    assert report['n_imported'] == 3 and report['n_images'] == 4
    assert report['source_sha256']['targets.jsonl'] == digest(source / 'targets.jsonl')
    assert reused.eval_config.scorer == 'mimic_cxr_automedbench_v1'

    # Identical synthetic JPEG bytes may share a copied artifact. Their distinct
    # evidence IDs and per-view provenance must survive a second migration too.
    again = tmp_path / 'reused-again'
    reuse_mimic_cxr(out, again, split='validate')
    assert load_dataset(again).info.splits == reused.info.splits
    assert json.loads((again / 'import_report.json').read_text())['n_images'] == 4

    class Model:
        name, config, timeout, last_usage = 'fake', {}, 1, None

        def complete(self, history, tools=None):
            assert 'SYNTHETIC_FINDINGS' not in json.dumps(history)
            assert 'SYNTHETIC_FULL_REPORT' not in json.dumps(history)
            text = [p['text'] for p in history[-1]['content'] if p['type'] == 'text']
            evidence_id = next(s.split()[1] for s in text if s.startswith('Evidence:'))
            return {'role': 'assistant', 'content': json.dumps({
                'turn_id': 't1', 'answer': {'findings': 'A generated synthetic observation.'},
                'citations': [evidence_id], 'reasoning_summary': 'Based on the supplied image.'})}

    # Enforce the same evaluator boundary as production: a run must not open targets.
    target_path = out / 'targets.jsonl'
    saved = target_path.read_text()
    target_path.write_text('not readable target JSON')
    run = execute(out, Model(), runs_root=tmp_path / 'runs', max_calls=1)
    target_path.write_text(saved)
    assert main(['eval', str(run)]) == 0
    metrics = json.loads((run / 'metrics.json').read_text())
    assert metrics['scorer'] == 'mimic_cxr_automedbench_v1'
    assert metrics['scored']['aggregate']['output_quality']['strict_output_compliance']['num'] == 3


@pytest.mark.parametrize('ids,limit,expected', [
    (None, None, ['40', '10', '20']),
    (None, 2, ['40', '10']),
    (['20', '40'], None, ['40', '20']),
    (['20', '40'], 1, ['40']),
])
def test_reuse_preserves_split_order_before_filter_and_limit(tmp_path, ids, limit, expected):
    source = frozen_dataset(tmp_path)
    info_path = source / 'dataset.json'
    info = json.loads(info_path.read_text())
    # Split order deliberately differs from the source episodes.jsonl order.
    info['splits']['validate'] = [f'mimic-cxr-s{sid}' for sid in ['40', '10', '20']]
    info_path.write_text(json.dumps(info))
    out = tmp_path / 'reused'
    report = reuse_mimic_cxr(source, out, split='validate', ids=ids, limit=limit)
    reused = load_dataset(out)
    expected_ids = [f'mimic-cxr-s{sid}' for sid in expected]
    assert [ep.id for ep in reused.episodes] == expected_ids
    assert reused.info.splits['validate'] == expected_ids
    assert report['selected_episode_ids'] == expected_ids


@pytest.mark.parametrize('kind', ['image_hash', 'missing_findings', 'duplicate_origin',
                                 'visible_report', 'no_images', 'invalid_answer_type'])
def test_reuse_rejects_damage_and_leakage(tmp_path, kind):
    source = frozen_dataset(tmp_path)
    if kind == 'image_hash':
        rows = [json.loads(s) for s in (source / 'provenance.jsonl').read_text().splitlines()]
        rows[0]['images'][0]['sha256'] = 'tampered'
        (source / 'provenance.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif kind in {'missing_findings', 'invalid_answer_type'}:
        rows = [json.loads(s) for s in (source / 'targets.jsonl').read_text().splitlines()]
        if kind == 'invalid_answer_type':
            rows[0]['turns']['t1']['answer'] = 'not structured sections'
        else:
            rows[0]['turns']['t1']['answer']['findings'] = ''
        (source / 'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif kind == 'duplicate_origin':
        path = source / 'provenance.jsonl'
        path.write_text(path.read_text() + path.read_text().splitlines()[0]+'\n')
    else:
        rows = [json.loads(s) for s in (source / 'episodes.jsonl').read_text().splitlines()]
        if kind == 'no_images':
            rows[0]['turns'][0]['evidence'] = []
        else:
            rows[0]['turns'][0]['evidence'][0]['text'] = 'SYNTHETIC_GOLD_LEAK'
        (source / 'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = tmp_path / 'out'
    with pytest.raises(ValueError):
        reuse_mimic_cxr(source, out, split='validate')
    assert not out.exists()


@pytest.fixture
def snapshot_package(tmp_path):
    """A complete synthetic snapshot; the public lock contains no clinical data."""
    source = synthetic_source(tmp_path)
    raw, dataset = tmp_path / 'raw', tmp_path / 'dataset'
    prepare_raw(source, raw, split='test', limit=1, report_mode='findings-only')
    import_mimic_cxr(raw, dataset, split='test', report_mode='findings-only')
    files = [{'path': p.relative_to(dataset).as_posix(), 'bytes': p.stat().st_size,
              'sha256': digest(p)} for p in sorted(dataset.rglob('*')) if p.is_file()]
    manifest = tmp_path / 'transfer-manifest.json'
    manifest.write_text(json.dumps({'files': files}))
    lock = {'snapshot': 'synthetic', 'transfer_manifest_sha256': digest(manifest),
            'file_count': len(files), 'episodes': 1, 'images': 1, 'split': 'test',
            'scorer': 'mimic_cxr_automedbench_v1'}
    return dataset, manifest, lock


def test_snapshot_verification(snapshot_package):
    from scripts.verify_mimic_cxr_snapshot import verify_snapshot
    dataset, manifest, lock = snapshot_package
    assert verify_snapshot(dataset, manifest, lock) == {
        'snapshot': 'synthetic', 'verified_files': lock['file_count'],
        'episodes': 1, 'images': 1, 'sha256_mismatches': 0}


@pytest.mark.parametrize('changed', ['manifest', 'instructions', 'targets', 'image'])
def test_snapshot_rejects_changed_inputs(snapshot_package, changed):
    from scripts.verify_mimic_cxr_snapshot import verify_snapshot
    dataset, manifest, lock = snapshot_package
    path = {'manifest': manifest, 'instructions': dataset / 'instructions.txt',
            'targets': dataset / 'targets.jsonl',
            'image': next((dataset / 'artifacts').glob('*.jpg'))}[changed]
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='manifest differs|SHA256 mismatch'):
        verify_snapshot(dataset, manifest, lock)


@pytest.mark.parametrize('field,value', [('episodes', 2), ('images', 2),
                                         ('scorer', 'unscored'), ('file_count', 1)])
def test_snapshot_rejects_wrong_cohort(snapshot_package, field, value):
    from scripts.verify_mimic_cxr_snapshot import verify_snapshot
    dataset, manifest, lock = snapshot_package
    with pytest.raises(ValueError, match='mismatch'):
        verify_snapshot(dataset, manifest, {**lock, field: value})
