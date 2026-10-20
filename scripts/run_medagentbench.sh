#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON_BIN:-python}" - "$@" <<'PY'
"""Local orchestration only: existing importer, runner and scorer stay unchanged."""
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HELP = """Usage: bash scripts/run_medagentbench.sh COMMAND [MODEL ... | RUN_DIR ...]

  import           Import into a NEW processed dataset; never overwrite NAS data.
  validate         Check the full 300-case dataset, fixed smoke, facts and snapshot.
  run MODEL ...    Run the fixed 30-case smoke; keep answers and tool histories.
  eval RUN_DIR ... Re-score saved runs offline; write metrics and summary.json.
  all MODEL ...    Validate existing processed data, run smoke, then score.
  --help           Show this help without loading data or requiring API keys.

export AMA_DATA_PATH=/Volumes/10.214.105.104/data
bash scripts/run_medagentbench.sh validate
bash scripts/run_medagentbench.sh all deepseek-v4-flash
bash scripts/run_medagentbench.sh eval runs/<batch>/model-1/<run-id>

Defaults and overrides:
  DATASET_DIR  $AMA_DATA_PATH/processed/MedAgentBench
  CONFIG       ama.json (API keys from its api_key_env variables or project .env)
  RUNS_DIR     runs/medagentbench-smoke (each invocation creates a new batch)
  PYTHON_BIN   python from your activated, existing environment
  MAX_CALLS=8 TIMEOUT=120 TEMPERATURE=0 MAX_TOKENS=2048
  SPLIT=smoke (set test explicitly for the full 300 cases)

Only import needs SOURCE_DIR (original MedAgentBench checkout) plus SNAPSHOT_DIR
or FHIR_BASE (default http://localhost:8080/fhir/). Existing processed data needs
neither original checkout nor Docker. all does NOT import or resample data.
run does not score; use eval or all. No automatic retry, resume or pacing.
All failed/not-executed cases remain in the denominator. Non-completed runs
return exit 1 after saving/scoring their records; inspect summary.json.
"""
command = sys.argv[1] if len(sys.argv) > 1 else '--help'
args = sys.argv[2:]
if command in {'--help', '-h', 'help'}:
    print(HELP)
    raise SystemExit(0)
if command not in {'import', 'validate', 'run', 'eval', 'all'}:
    raise SystemExit(HELP)

from ama.cli import _eval_run, _load_dotenv
from ama.data import load_dataset, validate_dataset
from ama.importers.medagentbench import SMOKE_IDS, SNAPSHOT_SCHEMA, _sha
from ama.recorder import write_json

def dataset_path():
    if os.environ.get('DATASET_DIR'):
        return Path(os.environ['DATASET_DIR']).expanduser().resolve()
    if not os.environ.get('AMA_DATA_PATH'):
        raise ValueError('Set AMA_DATA_PATH or DATASET_DIR')
    return (Path(os.environ['AMA_DATA_PATH']).expanduser() / 'processed/MedAgentBench').resolve()

def validate(folder):
    errors = validate_dataset(folder)
    if errors:
        raise ValueError('; '.join(errors))
    ds = load_dataset(folder, with_targets=True)
    expected = {f'task{t}_{i}' for t in range(1, 11) for i in range(1, 31)}
    if {ep.id for ep in ds.episodes} != expected or set(ds.targets) != expected:
        raise ValueError('Expected exactly 300 tasks and matching evaluation targets')
    if set(ds.info.splits.get('test', [])) != expected or len(ds.info.splits['test']) != 300:
        raise ValueError('test split must contain all 300 tasks exactly once')
    if ds.info.splits.get('smoke') != list(SMOKE_IDS):
        raise ValueError('smoke must match the original fixed 30 IDs in importer.SMOKE_IDS')
    if ds.eval_config is None or ds.eval_config.scorer != 'medagentbench':
        raise ValueError('Expected medagentbench scorer')
    meta = json.loads((folder / 'snapshot/metadata.json').read_text())
    if meta.get('schema') != SNAPSHOT_SCHEMA or meta.get('complete') is not True:
        raise ValueError('Incomplete or unsupported snapshot')
    hashes = {}
    counts = {}
    for kind in ('Patient', 'Observation'):
        path = folder / 'snapshot' / f'{kind}.jsonl'
        print(f'Checking snapshot {kind}...', flush=True)
        hashes[str(path.relative_to(folder))] = _sha(path)
        stats = meta['resources'][kind]
        if hashes[str(path.relative_to(folder))] != stats['sha256']:
            raise ValueError(f'{kind} snapshot checksum mismatch')
        counts[kind] = stats['count']
    # Parse processed facts, check identifiers and count every observation.
    patients = [json.loads(line) for line in (folder / 'patients.jsonl').read_text().splitlines() if line.strip()]
    patient_ids = [p['patient_id'] for p in patients]
    if len(set(patient_ids)) != len(patient_ids) or len(patients) != counts['Patient']:
        raise ValueError('Processed patient count or identifiers mismatch')
    case_ids, observation_count = set(), 0
    with (folder / 'cases.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                case = json.loads(line)
                if case['patient_id'] in case_ids or case['patient_id'] not in patient_ids:
                    raise ValueError('Duplicate or unknown case patient')
                case_ids.add(case['patient_id'])
                observation_count += len(case['observations'])
    if case_ids != set(patient_ids) or observation_count != counts['Observation']:
        raise ValueError('Processed case/observation counts do not match snapshot')
    for name in ('dataset.json', 'episodes.jsonl', 'instructions.txt', 'patients.jsonl',
                 'cases.jsonl', 'targets.jsonl', 'eval.json', 'import_report.json',
                 'snapshot/metadata.json', 'evaluator/refsol.py', 'evaluator/utils.py',
                 'evaluator/funcs_v1.json'):
        hashes[name] = _sha(folder / name)
    report = {'dataset_dir': str(folder), 'task_count': 300, 'smoke_ids': list(SMOKE_IDS),
              'patient_count': len(patient_ids), 'observation_count': observation_count,
              'sha256': hashes, 'validated_at_utc': datetime.now(timezone.utc).isoformat()}
    print('Validated: 300 tasks, fixed smoke=30, patients=' + str(len(patient_ids)) +
          ', observations=' + str(observation_count), flush=True)
    return report

def evaluate(path):
    _eval_run(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    metrics = json.loads((path / 'metrics.json').read_text())['scored']
    statuses = Counter(row['termination'] for row in metrics['per_episode'].values())
    summary = {'run_id': manifest['run_id'], 'model': manifest['model'],
               'model_config': manifest['model_config'], 'max_calls': manifest['max_calls'],
               'timeout': manifest['timeout'], 'dataset_dir': manifest['dataset_dir'],
               'episode_ids': manifest['episode_ids'], 'aggregate': metrics['aggregate'],
               'termination_counts': dict(statuses), 'api_error_count': statuses['api_error'],
               'score_scope': 'AMA adaptation; not native MedAgentBench results'}
    write_json(path / 'summary.json', summary)
    print(f'Summary: {path / "summary.json"}; API errors: {statuses["api_error"]}', flush=True)
    return summary

try:
    if command == 'eval':
        if not args:
            raise ValueError('eval requires one or more saved RUN_DIR paths')
        for value in args:
            evaluate(Path(value).expanduser().resolve())
        raise SystemExit(0)
    folder = dataset_path()
    if command == 'import':
        if args:
            raise ValueError('import takes environment settings, not model arguments')
        from ama.importers.medagentbench import main as importer
        source = os.environ.get('SOURCE_DIR')
        if not source:
            raise ValueError('import requires SOURCE_DIR (original MedAgentBench checkout)')
        opts = ['--source', source, '--out', str(folder)]
        if os.environ.get('SNAPSHOT_DIR'):
            if os.environ.get('FHIR_BASE'):
                raise ValueError('Choose SNAPSHOT_DIR or FHIR_BASE, not both')
            opts += ['--snapshot', os.environ['SNAPSHOT_DIR']]
        else:
            opts += ['--fhir-base', os.environ.get('FHIR_BASE', 'http://localhost:8080/fhir/')]
        raise SystemExit(importer(opts))
    if command == 'validate':
        if args:
            raise ValueError('validate does not take model arguments')
        validate(folder)
        raise SystemExit(0)
    if not args:
        raise ValueError('Provide at least one model alias')
    split = os.environ.get('SPLIT', 'smoke')
    if split not in {'smoke', 'test'}:
        raise ValueError('SPLIT must be smoke or test')
    from ama.model import Model
    from ama.runner import execute
    _load_dotenv()
    config = Path(os.environ.get('CONFIG', 'ama.json')).expanduser()
    timeout = float(os.environ.get('TIMEOUT', '120'))
    max_calls = int(os.environ.get('MAX_CALLS', '8'))
    temperature = float(os.environ.get('TEMPERATURE', '0'))
    max_tokens = int(os.environ.get('MAX_TOKENS', '2048'))
    if max_calls < 1 or max_tokens < 1:
        raise ValueError('MAX_CALLS and MAX_TOKENS must be positive')
    models = [Model.from_config(alias, config, timeout=timeout) for alias in args]
    for model in models:
        model.config.update(temperature=temperature, max_tokens=max_tokens)
    validation = validate(folder)
    root = Path(os.environ.get('RUNS_DIR', 'runs/medagentbench-smoke')).expanduser().resolve()
    batch = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    batch.mkdir(parents=True, exist_ok=False)
    write_json(batch / 'validation.json', validation)
    def git(*opts):
        return subprocess.check_output(['git', *opts], text=True).strip()
    write_json(batch / 'code-version.json', {
        'commit': git('rev-parse', 'HEAD'),
        'tracked_status': git('status', '--porcelain', '--untracked-files=no'),
        'orchestration_sha256': _sha(Path('scripts/run_medagentbench.sh')),
        'source_sha256': {str(p): _sha(p) for p in sorted(Path('src/ama').rglob('*.py'))},
        'split': split, 'models': args})
    print(f'Batch: {batch}', flush=True)
    failures = False
    for i, model in enumerate(models, 1):
        path = execute(str(folder), model, mode='run', split=split,
                       runs_root=batch / f'model-{i}', max_calls=max_calls,
                       tools_path=Path('src/ama/importers/medagentbench.py'))
        if command == 'all':
            evaluate(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        failures |= manifest['termination'] != 'completed'
        print(f'{model.name}: {manifest["termination"]}; saved: {path}', flush=True)
        if command == 'run':
            print(f'Next: bash scripts/run_medagentbench.sh eval {path}', flush=True)
    raise SystemExit(1 if failures else 0)
except (OSError, ValueError, KeyError, ImportError) as exc:
    print(f'ERROR: {exc}', file=sys.stderr)
    raise SystemExit(1)
PY
