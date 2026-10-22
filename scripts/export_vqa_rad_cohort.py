"""Export an already frozen VQA-RAD cohort as a portable, standalone AMA dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from ama.data import load_dataset, validate_dataset
from ama.importers.vqa_rad import _json, _jsonl, _sha
from scripts.vqa_rad_benchmark import verify


def export_cohort(root: Path, out: Path) -> dict:
    """Copy the exact evaluated records; never resample or overwrite an output."""
    root, out = Path(root), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    cohort = verify(root, evaluator=True)
    source = Path(cohort['dataset_dir'])
    ids = cohort['episode_ids']
    selected = {}
    for name in ('episodes.jsonl', 'targets.jsonl', 'provenance.jsonl'):
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        indexed = {row['id']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f'duplicate IDs in {name}')
        selected[name] = [indexed[eid] for eid in ids]
    images = sorted({e['file'] for row in selected['episodes.jsonl']
                     for turn in row['turns'] for e in turn['evidence'] if 'file' in e})
    # verify() checks these paths against the frozen hashes before any copying.
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in images):
        raise ValueError('unsafe artifact path')
    types = dict(Counter(row['turns']['t1']['question_type']
                         for row in selected['targets.jsonl']))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vqarad-export-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        stage.mkdir()
        for name, rows in selected.items():
            _jsonl(stage / name, rows)
        for name in ('instructions.txt', 'eval.json', *images):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            if _sha(target) != _sha(source / name):
                raise ValueError(f'copy mismatch: {name}')
        info = json.loads((source / 'dataset.json').read_text())
        info.update(name=out.name, splits={'test': ids})
        _json(stage / 'dataset.json', info)
        # The original cohort remains byte-for-byte in its historical location.
        # This receipt carries portable selection data without local absolute paths.
        portable = {key: value for key, value in cohort.items() if key != 'dataset_dir'}
        portable['original_cohort_sha256'] = _sha(root / 'cohort.json')
        _json(stage / 'source_cohort.json', portable)
        report = {
            'source': {'revision': cohort['revision']}, 'scorer': cohort['scorer'],
            'n_imported': len(ids), 'n_artifacts': len(images),
            'question_type_counts': {'test': types},
            'selection': {key: portable[key] for key in
                          ('selection', 'seed', 'source_test_count', 'selected_test_indices')},
            'original_cohort_sha256': portable['original_cohort_sha256'],
            'source_import_report_sha256': _sha(source / 'import_report.json'),
            'record_transform': 'Selected source JSON records unchanged; test split only.',
        }
        _json(stage / 'import_report.json', report)
        for name in ('DATASET_CARD.md', 'DATASET_CARD.zh-CN.md'):
            text = (source / name).read_text()
            if name == 'DATASET_CARD.md':
                note = (f'# {out.name}\n\nStandalone frozen test subset: {len(ids)} QA episodes, '
                        f'{len(images)} images; no train split. Seed {cohort["seed"]}; '
                        'exact IDs and source positions in source_cohort.json. This is our '
                        'comparison subset, not an official AutoMedBench cohort. Answers remain '
                        'evaluator-only. All artifact paths are relative.\n\n'
                        'The following original full-dataset card describes the parent dataset; '
                        'its counts/splits do not describe this subset.\n\n')
            else:
                note = (f'# {out.name}\n\n独立固定测试子集：{len(ids)}条问答、{len(images)}张图像，'
                        f'无训练集；seed={cohort["seed"]}。准确选择位置及ID见source_cohort.json。'
                        '这是本次比较自选子集，不是官方指定队列。答案仅供evaluator读取，图像使用相对路径。'
                        '\n\n以下保留原全量数据卡，其数量和划分描述父数据集，不代表本子集。\n\n')
            (stage / name).write_text(note + text, encoding='utf-8')
        errors = validate_dataset(stage)
        if errors:
            raise ValueError(errors)
        ds = load_dataset(stage, with_targets=True)
        if ([e.id for e in ds.episodes] != ids or len(ds.targets) != len(ids)
                or load_dataset(stage, with_targets=False).targets):
            raise ValueError('export coverage or target isolation differs')
        # Verify record equality after serialization, including answers/provenance.
        for name, rows in selected.items():
            if [json.loads(line) for line in (stage / name).read_text().splitlines()] != rows:
                raise ValueError(f'records changed: {name}')
        files = {p.relative_to(stage).as_posix(): _sha(p)
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        aggregate = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        _json(stage / 'checksums.json', {'files': files, 'overall_sha256': aggregate,
              'algorithm': 'SHA256 of UTF-8 json.dumps(files, sort_keys=True); excludes checksums.json'})
        if out.exists() or out.is_symlink():
            raise FileExistsError(out)
        stage.rename(out)
    return {**report, 'overall_sha256': aggregate}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = export_cohort(args.cohort, args.out)
    print(json.dumps({key: result[key] for key in
                     ('n_imported', 'n_artifacts', 'question_type_counts', 'overall_sha256')}, indent=2))
