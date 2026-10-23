"""Phase-one offline preparation audit; may read targets, never runs models.

This is NOT ama regression check or an acceptance result. Output contains only
aggregate counts and fixed error categories; no paths, case IDs or exception text.
"""
import argparse
import json
from pathlib import Path

from ama.data import load_dataset, validate_dataset
from scripts.pathvqa import prepare
from scripts.verify_mimic_cxr_snapshot import verify_snapshot, LOCK_PATH

KNOWN = {'mimic-cdm', 'mimic-cxr', 'pathvqa', 'rocov2', 'symptom2disease'}


def audit(config):
    if not isinstance(config, dict) or config.get('schema') != 'ama-regression-site-draft-v1':
        raise ValueError('unsupported config schema')
    entries = config.get('benchmarks')
    if not isinstance(entries, dict) or not entries or set(entries) - KNOWN:
        raise ValueError('invalid benchmark selection')
    suite = config.get('suite', 'sample')
    if suite not in {'sample', 'full'}:
        raise ValueError('invalid suite')
    rows = []
    for name, entry in entries.items():
        if not isinstance(entry, dict) or type(entry.get('enabled')) is not bool:
            raise ValueError('enabled must be boolean')
        if not entry['enabled']:
            rows.append({'benchmark': name, 'status': 'disabled'})
            continue
        row = {'benchmark': name, 'status': 'blocked', 'category': 'scope_not_configured'}
        rows.append(row)
        scope = entry.get(suite)
        if not scope:
            continue
        try:
            if name == 'mimic-cdm' and entry.get('view') != 'interactive':
                raise ValueError('interactive required')
            dataset = Path(scope['dataset'])
            if validate_dataset(dataset):
                row['category'] = 'dataset_invalid'
                continue
            ds = load_dataset(dataset, with_targets=True)
            lock = Path(scope['lock'])
            if name == 'pathvqa':
                if suite != 'sample':
                    row['category'] = 'full_lock_verifier_pending'
                    continue
                result = prepare(dataset, lock)
                ids = result['episode_ids']
                selected = [ds.select(episode_id=e)[0] for e in ids]
            elif name == 'mimic-cxr' and suite == 'sample':
                pinned = json.loads(LOCK_PATH.read_text())
                if json.loads(lock.read_text()) != pinned:
                    raise ValueError('unexpected snapshot lock')
                verify_snapshot(dataset, Path(scope['manifest']), pinned)
                selected = ds.select(split=pinned['split'])
            else:
                row['category'] = 'cohort_lock_verifier_pending'
                continue
            if suite == 'sample' and len(selected) != entry.get('sample_size', 100):
                raise ValueError('sample count mismatch')
            if any(ep.id not in ds.targets or any(t.id not in ds.targets[ep.id].turns
                   for t in ep.turns) for ep in selected):
                raise ValueError('targets incomplete')
            row.update(status='verified', category='preparation_only', episodes=len(selected),
                       turns=sum(len(ep.turns) for ep in selected),
                       images=sum(bool(e.file) for ep in selected for t in ep.turns for e in t.evidence))
        except (OSError, ValueError, KeyError, TypeError):
            row['category'] = 'data_or_lock_invalid'
    return {'schema': 'ama-regression-preparation-audit-v1', 'suite': suite,
            'acceptance': 'not_run', 'benchmarks': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(json.loads(args.config.read_text()))
    except (OSError, ValueError, TypeError):
        print(json.dumps({'status': 'blocked', 'category': 'config_invalid'}))
        return 2
    print(json.dumps(result, indent=2))
    return 2 if any(r['status'] == 'blocked' for r in result['benchmarks']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
