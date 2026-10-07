"""Frozen measurements of completed clinical runs, owned by ``ama probe``.

Cumulative inputs are controlled evidence reconstructions, not native within-batch
agent states. Endpoint inputs preserve the original submission verbatim.
"""
from __future__ import annotations

import copy
import fcntl
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .agent import Agent
from .data import load_dataset
from .model import Model, ModelError
from .original_agent_probing import (
    _validate_replay_history, candidate, file_hash, original_label_probability,
    probe_history, read_rows,
)
from .probing import digest, PTRUE, SUFFICIENT
from .probing_transport import PacedProbeModel, ProbeTransport
from .recorder import write_json

SCHEMA = 'ama-probe-v1'
TOOLS = {'physical_examination', 'laboratory_results', 'microbiology', 'imaging', 'list_imaging'}
SYSTEM = ('Evaluate the supplied fixed proposed diagnosis using only the patient '
          'evidence provided. Do not infer findings from missing records. '
          'This is an independent evidence-sufficiency measurement, not a new clinical submission.')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def _text(message):
    value = message.get('content')
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(p, dict) and p.get('type') == 'text'
                                       and isinstance(p.get('text'), str) for p in value):
        return '\n'.join(p['text'] for p in value)
    raise ValueError('cumulative v1 requires text-only patient observations and tool results')


def cumulative_snapshots(history):
    """Extract every result once; catalog and assistant reasoning never become evidence."""
    _validate_replay_history(history)
    if not history or history[0].get('role') != 'system':
        raise ValueError('original system message is required')
    first = next((i for i, m in enumerate(history) if m.get('role') == 'user'), None)
    if first != 1:
        raise ValueError('cumulative v1 requires one initial patient observation after system')
    hpi = _text(history[first])
    results, excluded, seen, calls = [], [], set(), {}
    batch = -1
    for index, message in enumerate(history):
        value = message.get('content')
        if isinstance(value, list) and any(not isinstance(p, dict) or p.get('type') != 'text' for p in value):
            raise ValueError('cumulative v1 does not support image or other non-text attachments')
        if message.get('role') == 'user' and index != first:
            raise ValueError('cumulative v1 does not support additional user observations/attachments')
        requests = message.get('tool_calls') or []
        if requests:
            batch += 1
            for position, request in enumerate(requests):
                function = request.get('function') or {}
                name = function.get('name')
                if name not in TOOLS or request.get('type') != 'function':
                    raise ValueError('unsupported cumulative tool: '+str(name))
                calls[request['id']] = (name, function.get('arguments', '{}'), batch, position)
        if message.get('role') != 'tool':
            continue
        name, arguments, batch_id, position = calls[message['tool_call_id']]
        text = _text(message)
        record = {'source_message_index': index, 'tool_call_id': message['tool_call_id'],
                  'tool': name, 'arguments': arguments, 'batch_index': batch_id,
                  'batch_position': position, 'content_sha256': digest(text)}
        if name == 'list_imaging':
            excluded.append({**record, 'reason': 'navigation_only'})
            continue
        fingerprint = digest([name, arguments, text])
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        failed = isinstance(parsed, dict) and 'error' in parsed
        empty = not text.strip() or text.strip() == 'No microbiology results available for this admission.'
        if isinstance(parsed, dict) and parsed.get('results') == []:
            empty = True
        results.append({**record, 'text': text, 'duplicate': fingerprint in seen,
                        'failed': failed, 'no_result': empty,
                        'is_clinical_finding': not (failed or empty)})
        seen.add(fingerprint)
    snapshots = []
    for step in range(len(results)+1):
        visible = results[:step]
        context = '[Patient history]\n'+hpi
        for i, record in enumerate(visible, 1):
            context += f"\n\n[Evidence {i}: {record['tool']}]\n"+record['text']
        snapshots.append({'step': step, 'turn_id': f'e{step:04d}', 'kind': 'cumulative',
                          'context': context, 'prefix_sha256': digest(context),
                          'evidence_count': step, 'unique_clinical_result_count': sum(r['is_clinical_finding'] and not r['duplicate'] for r in visible),
                          'new_evidence': {k: v for k, v in visible[-1].items() if k != 'text'} if visible else None})
    return snapshots, excluded


def _source(root):
    manifest = _json(root/'manifest.json')
    if manifest.get('mode') != 'run' or manifest.get('output_contract_version') != 2:
        raise ValueError('source must be a standard AMA clinical Decision run')
    if manifest.get('termination') != 'completed':
        raise ValueError('source clinical run must be completed')
    rows = read_rows(root/'decisions.jsonl')
    indexed = {r['episode_id']: r for r in rows}
    ids = manifest['episode_ids']
    if len(indexed) != len(rows) or len(set(ids)) != len(ids) or set(indexed) != set(ids):
        raise ValueError('probe v1 requires one clinical Decision per episode')
    histories = _json(root/'messages.json')
    if set(histories) != set(ids):
        raise ValueError('source histories and Decision cohort differ')
    return manifest, indexed, histories


def _build(source, episodes, mode, kinds, config, overrides):
    manifest, rows, histories = _source(source)
    if any(ep not in rows for ep in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError('unknown or duplicate selected episode')
    if set(overrides)-set(episodes):
        raise ValueError('candidate overrides contain unselected/unknown episodes')
    snapshots, candidates, excluded, jobs = [], [], [], []
    for ep in episodes:
        history = histories[ep]
        if not history or history[-1].get('role') != 'assistant' or history[-1].get('tool_calls'):
            raise ValueError('clinical source must end with its final assistant submission')
        original = candidate(rows[ep])
        diagnosis = overrides.get(ep, original)
        if diagnosis is not None and (not isinstance(diagnosis, str) or not diagnosis.strip()):
            diagnosis = None
        candidates.append({'episode_id': ep, 'original': original, 'candidate': diagnosis,
                           'overridden': ep in overrides, 'source_decision_sha256': digest(rows[ep]),
                           'reason': None if diagnosis else 'missing_final_diagnosis'})
        if mode == 'cumulative':
            snaps, skipped = cumulative_snapshots(history)
            excluded.extend({'episode_id': ep, **r} for r in skipped)
        else:
            _validate_replay_history(history)
            snaps = [{'step': 0, 'turn_id': 'endpoint', 'kind': 'post_submission',
                      'history': copy.deepcopy(history), 'prefix_sha256': digest(history)}]
        for snapshot in snaps:
            snapshot.update(episode_id=ep, candidate=diagnosis,
                            target='final_hindsight' if mode == 'cumulative' else 'original_submission')
            snapshots.append(snapshot)
            if diagnosis is None:
                continue
            for kind in kinds:
                if mode == 'endpoint':
                    request = probe_history(snapshot, kind, diagnosis)
                else:
                    template = SUFFICIENT if kind == 'sufficient' else PTRUE
                    request = [{'role': 'system', 'content': SYSTEM}, {'role': 'user',
                               'content': template.format(question=snapshot['context'], answer=diagnosis)}]
                identity = {'episode_id': ep, 'turn_id': snapshot['turn_id'], 'step': snapshot['step'],
                            'mode': mode, 'kind': kind, 'target': snapshot['target'], 'candidate': diagnosis,
                            'prefix_sha256': snapshot['prefix_sha256'], 'request_sha256': digest(request),
                            'model_config': config, 'source_decision_sha256': digest(rows[ep]), 'schema': SCHEMA}
                jobs.append({**identity, 'job_id': digest(identity), 'request': request})
    return snapshots, candidates, excluded, jobs


def prepare(source_run, *, mode, model_name, config, episodes=None, candidates_file=None,
            kinds=None, runs_root=Path('runs'), timeout=60, max_probe_tokens=32, top_logprobs=20):
    source = Path(source_run).resolve()
    source_manifest, _, histories = _source(source)
    kinds = kinds or ['sufficient']
    if mode not in {'cumulative', 'endpoint'} or not kinds or len(set(kinds)) != len(kinds) or set(kinds)-{'sufficient', 'true'}:
        raise ValueError('unsupported probe mode or duplicate/unknown kinds')
    if any(config.get(k) != source_manifest['model_config'].get(k) for k in ['model', 'base_url']):
        raise ValueError('probe model/provider differs from original generation')
    if not isinstance(config.get('api_key_env'), str) or not config['api_key_env']:
        raise ValueError('model config must name api_key_env')
    if type(max_probe_tokens) is not int or max_probe_tokens < 1 or type(top_logprobs) is not int or not 1 <= top_logprobs <= 20:
        raise ValueError('invalid probe token/logprob limits')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    cfg = dict(config, temperature=0, logprobs=True, top_logprobs=top_logprobs, max_tokens=max_probe_tokens)
    if cfg.get('n', 1) != 1:
        raise ValueError('binary probe requires n=1')
    hashes = {str(source/name): file_hash(source/name) for name in ['manifest.json', 'messages.json', 'decisions.jsonl']}
    if (source/'prefixes.jsonl').exists():
        hashes[str(source/'prefixes.jsonl')] = file_hash(source/'prefixes.jsonl')
        for row in read_rows(source/'prefixes.jsonl'):
            history = row.get('history', [])
            if row.get('episode_id') not in histories or history != histories[row['episode_id']][:len(history)]:
                raise ValueError('recorded prefix is not a source history prefix')
    overrides = {}
    if candidates_file:
        path = Path(candidates_file).resolve()
        overrides = _json(path)
        if not isinstance(overrides, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in overrides.items()):
            raise ValueError('candidates file must map episode IDs to nonempty diagnosis strings')
        hashes[str(path)] = file_hash(path)
    ids = episodes if episodes is not None else source_manifest['episode_ids']
    snapshots, candidates, excluded, jobs = _build(source, ids, mode, kinds, cfg, overrides)
    # Detect edits during preparation before creating an executable frozen task.
    if any(file_hash(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('source changed during preparation')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    root = Path(runs_root).resolve()/run_id
    root.mkdir(parents=True)
    (root/'requests').mkdir()
    for job in jobs:
        write_json(root/'requests'/(job['request_sha256']+'.json'), job['request'])
    _jsonl(root/'snapshots.jsonl', snapshots)
    _jsonl(root/'jobs.jsonl', [{k: v for k, v in j.items() if k != 'request'} for j in jobs])
    write_json(root/'candidates.json', candidates)
    write_json(root/'excluded_results.json', excluded)
    write_json(root/'missing_original_candidates.json', [r for r in candidates if r['reason']])
    archive = root/'original_submission'; archive.mkdir()
    for name in ['manifest.json', 'decisions.jsonl']:
        (archive/name).write_bytes((source/name).read_bytes())
    for name in ['probe_calls.jsonl', 'diagnostics.jsonl', 'transport_attempts.jsonl']:
        (root/name).touch()
    plan = {'schema': SCHEMA, 'source_run': str(source), 'source_hashes': hashes,
            'episode_ids': ids, 'mode': mode, 'kinds': kinds, 'model': model_name, 'model_config': cfg,
            'timeout': timeout, 'candidate_overrides': overrides, 'label_readout': 'first_complete_label',
            'transport': {'lease_path': str(Path(runs_root).resolve()/'.probe-api.lock'), 'request_delay_seconds': 3.0}}
    write_json(root/'plan.json', plan)
    write_json(root/'manifest.json', {'schema': SCHEMA, 'mode': 'probe', 'run_id': run_id,
                                    'plan_sha256': digest(plan), 'termination': 'prepared',
                                    'model': model_name, 'measurement_design': mode})
    validate(root)
    return root


def validate(root):
    root = Path(root)
    manifest, plan = _json(root/'manifest.json'), _json(root/'plan.json')
    if manifest.get('schema') != SCHEMA or plan.get('schema') != SCHEMA or digest(plan) != manifest['plan_sha256']:
        raise ValueError('frozen probe plan changed')
    for path, expected in plan['source_hashes'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError('frozen source changed: '+path)
    snapshots, candidates, excluded, jobs = _build(Path(plan['source_run']), plan['episode_ids'], plan['mode'],
                                                  plan['kinds'], plan['model_config'], plan['candidate_overrides'])
    if read_rows(root/'snapshots.jsonl') != snapshots or _json(root/'candidates.json') != candidates or _json(root/'excluded_results.json') != excluded:
        raise ValueError('frozen snapshots/candidates/provenance changed')
    if _json(root/'missing_original_candidates.json') != [r for r in candidates if r['reason']]:
        raise ValueError('missing-candidate record changed')
    if read_rows(root/'jobs.jsonl') != [{k: v for k, v in j.items() if k != 'request'} for j in jobs]:
        raise ValueError('frozen jobs changed')
    for job in jobs:
        if _json(root/'requests'/(job['request_sha256']+'.json')) != job['request']:
            raise ValueError('frozen probe request changed')
    for name in ['manifest.json', 'decisions.jsonl']:
        if file_hash(root/'original_submission'/name) != plan['source_hashes'][str(Path(plan['source_run'])/name)]:
            raise ValueError('archived clinical submission changed')
    return manifest, plan, snapshots, jobs


def _recorded(root, jobs):
    expected = {j['job_id']: {k: v for k, v in j.items() if k != 'request'} for j in jobs}
    rows = read_rows(root/'probe_calls.jsonl')
    indexed = {}
    for row in rows:
        jid = row.get('job_id')
        if jid not in expected or jid in indexed or any(row.get(k) != v for k, v in expected[jid].items()):
            raise ValueError('duplicate, unexpected or changed recorded probe job')
        indexed[jid] = row
    return indexed


def _probability(row, plan):
    bp = original_label_probability(row.get('response_metadata'), plan['label_readout'])
    if row.get('error') or (row.get('response_metadata') or {}).get('model') != plan['model_config']['model']:
        bp.update(valid=False, reason='missing_or_failed_or_model_mismatch', p_ab=None)
    return bp


def execute(root):
    root = Path(root).resolve()
    manifest, plan, _, jobs = validate(root)
    def append(name, value):
        with (root/name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    with (root/'process.lock').open('a') as lock, Path(plan['transport']['lease_path']).open('a') as lease:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recorded = _recorded(root, jobs)
        started = {r['job_id'] for r in read_rows(root/'diagnostics.jsonl') if r.get('kind') == 'probe_start'}
        transport = ProbeTransport(plan['transport'], lambda r: append('transport_attempts.jsonl', r))
        manifest['termination'] = 'running'; write_json(root/'manifest.json', manifest)
        try:
            for job in jobs:
                jid = job['job_id']
                if jid in recorded:
                    if not _probability(recorded[jid], plan)['valid']:
                        raise ValueError('previous failed/invalid probe requires review; not automatically resubmitted')
                    continue
                if jid in started:
                    raise ValueError('unresolved in-flight probe; no automatic resubmission')
                model = PacedProbeModel(plan['model'], plan['model_config'], plan['timeout'], transport, jid)
                identity = {k: v for k, v in job.items() if k != 'request'}
                append('diagnostics.jsonl', {'kind': 'probe_start', 'job_id': jid, 'time': time.time()})
                agent = Agent(model, max_calls=1)
                agent.history = copy.deepcopy(job['request'][:-1])
                began = time.monotonic()
                reply, error = None, None
                try:
                    reply = agent.chat(job['request'][-1]['content'])
                except Exception as exc:
                    # Persist known responses/usage even for malformed replies;
                    # never silently retry an invalid format or tool request.
                    error = f'{type(exc).__name__}: {exc}'
                row = {**identity, 'reply': reply, 'error': error, 'usage': model.last_usage,
                       'response_metadata': model.last_response_metadata,
                       'http_attempts': len(model.transport_attempts),
                       'duration_ms': round((time.monotonic()-began)*1000)}
                append('probe_calls.jsonl', row)
                recorded[jid] = row
                if not _probability(row, plan)['valid']:
                    raise ValueError('probe failed or binary probability unavailable; stopped before next job')
            manifest['termination'] = 'completed'
        except BaseException:
            manifest['termination'] = 'interrupted'
            raise
        finally:
            write_json(root/'manifest.json', manifest)
            write_json(root/'progress.json', {'completed_jobs': len(recorded), 'expected_jobs': len(jobs),
                                             'termination': manifest['termination']})
    return root


def resume(root):
    root = Path(root).resolve()
    manifest = _json(root/'manifest.json')
    if manifest.get('mode') == 'probe':
        return execute(root)
    if not manifest.get('probing_plan'):
        raise ValueError('resume requires a probe task or an archived probing run')
    # Archived plans retain their exact schema, hashes, prompts and model settings.
    from .probing import execute_probing
    model = Model(manifest['model'], manifest['model_config'], manifest.get('timeout', 60))
    return execute_probing(manifest['dataset_dir'], model, Path(manifest['probing_plan']),
                           runs_root=root.parent, resume=root)


def evaluate(root):
    from .scorer import REGISTRY
    root = Path(root)
    manifest, plan, snapshots, jobs = validate(root)
    rows = _recorded(root, jobs)
    features, previous = [], {}
    by_measurement = {(j['episode_id'], j['turn_id'], j['kind']): j for j in jobs}
    for snap in snapshots:
        entry = {k: v for k, v in snap.items() if k not in {'history', 'context'}}
        entry['probes'] = {}
        for kind in plan['kinds']:
            job = by_measurement.get((snap['episode_id'], snap['turn_id'], kind))
            row = rows.get(job['job_id'], {}) if job else {}
            bp = _probability(row, plan)
            if not job:
                bp.update(valid=False, reason='missing_final_diagnosis', p_ab=None)
            key = (snap['episode_id'], kind)
            prev = previous.get(key)
            bp['delta_p'] = bp['p_ab']-prev['p_ab'] if bp['valid'] and prev and prev['valid'] else None
            previous[key] = bp
            entry['probes'][kind] = bp
        features.append(entry)
    source_manifest = _json(root/'original_submission/manifest.json')
    dataset = load_dataset(Path(source_manifest['dataset_dir']), with_targets=True)
    dataset.episodes = [dataset.select(episode_id=ep)[0] for ep in plan['episode_ids']]
    scorer = dataset.eval_config.scorer if dataset.eval_config else 'unscored'
    decisions = {r['episode_id']: [r] for r in read_rows(root/'original_submission/decisions.jsonl')}
    original = REGISTRY[scorer](dataset, decisions)
    probabilities = [p for f in features for p in f['probes'].values()]
    started = {r['job_id'] for r in read_rows(root/'diagnostics.jsonl') if r.get('kind') == 'probe_start'}
    unresolved = started-set(rows)
    result = {'measurement_design': plan['mode'], 'target_interpretation': 'retrospective_fixed_final_diagnosis',
              'per_prefix': features, 'original_clinical_score': {'scorer': scorer, 'scored': original},
              'aggregate': {'admissions': len(plan['episode_ids']), 'prefixes': len(snapshots),
                            'expected_probes': len(jobs), 'recorded_probes': len(rows),
                            'valid_probes': sum(p['valid'] for p in probabilities),
                            'invalid_probes': sum(not p['valid'] for p in probabilities),
                            'missing_candidates': sum(s['candidate'] is None for s in snapshots if s['step'] == 0),
                            'new_http_attempts': sum(r.get('http_attempts', 0) for r in rows.values()),
                            'total_tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in rows.values()),
                            'unresolved_in_flight_jobs': len(unresolved),
                            'unknown_usage_jobs': len(unresolved)+sum(r.get('usage') is None for r in rows.values())}}
    write_json(root/'metrics.json', {'run_id': manifest['run_id'], 'model': manifest['model'],
                                    'scorer': SCHEMA, 'scored': result})
    print(json.dumps(result['aggregate'], ensure_ascii=False, indent=2))
    return 0
