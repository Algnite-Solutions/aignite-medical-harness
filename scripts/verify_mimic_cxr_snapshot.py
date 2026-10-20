#!/usr/bin/env python3
"""Verify the restricted findings100-v1 snapshot without inference or data changes."""
import argparse
import json
from pathlib import Path

from ama.data import load_dataset, validate_dataset
from ama.importers.mimic_cxr import digest, safe_path

LOCK_PATH = Path(__file__).resolve().parents[1] / 'datasets/mimic_cxr/findings100-v1.lock.json'


def verify_snapshot(dataset_dir: Path, manifest_path: Path, lock: dict) -> dict:
    """Pin the file list first, then check bytes and native AMA task structure."""
    if digest(manifest_path) != lock['transfer_manifest_sha256']:
        raise ValueError('transfer manifest differs from the frozen snapshot')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest['files']
    paths = [row['path'] for row in files]
    if len(files) != lock['file_count'] or len(set(paths)) != len(paths):
        raise ValueError('snapshot file count or uniqueness mismatch')
    root = dataset_dir.resolve()
    for row in files:
        path = safe_path(root, row['path'])
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('snapshot file size or SHA256 mismatch')
    if validate_dataset(root):
        raise ValueError('snapshot is not a valid AMA dataset')
    dataset = load_dataset(root, with_targets=True)
    episodes = dataset.select(split=lock['split'])
    if len(episodes) != len(dataset.episodes) or len(episodes) != lock['episodes']:
        raise ValueError('snapshot episode count or split mismatch')
    if any(len(ep.turns) != 1 for ep in episodes):
        raise ValueError('expected one turn per study')
    images = sum(bool(e.file) for ep in episodes for e in ep.turns[0].evidence)
    if images != lock['images'] or set(dataset.targets) != {ep.id for ep in episodes}:
        raise ValueError('snapshot image count or target coverage mismatch')
    if dataset.eval_config is None or dataset.eval_config.scorer != lock['scorer']:
        raise ValueError('snapshot scorer mismatch')
    return {'snapshot': lock['snapshot'], 'verified_files': len(files),
            'episodes': len(episodes), 'images': images, 'sha256_mismatches': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
        result = verify_snapshot(args.dataset, args.manifest, lock)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
