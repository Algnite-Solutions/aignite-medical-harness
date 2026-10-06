"""Native AMA measurements of fixed original submissions, without rediagnosis."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import time
from pathlib import Path

from .agent import Agent
from .data import load_dataset, resolve_dataset_dir, validate_visible_files
from .model import Model, ModelError
from .recorder import Recorder, write_json
from .probing_transport import ProbeTransport, PacedProbeModel
from .probing import binary_probability, digest, extract_snapshots, PTRUE, SUFFICIENT

from .probing_contracts import (ORIGINAL_SCHEMA as SCHEMA, ORIGINAL_SCHEMAS,
                                ORIGINAL_CONTRACT, ORIGINAL_CONTRACTS)
MODE_EXCEPTION = ('\n\nMeasurement mode exception, applicable only to the following additional '
    'measurement request: retain the original clinical task and evidence, but do not make '
    'a new diagnosis, request tools, or submit another clinical Decision JSON. '
    'Judge only the explicitly supplied fixed proposed diagnosis. '
    'Output exactly one uppercase letter A or B, without explanation or formatting. '
    'This measurement is separate from the original clinical submission.')
TARGETS = [('sufficient', 'original_hpi'), ('sufficient', 'original_final_hindsight'),
           ('true', 'original_final_hindsight')]
POST_SUBMISSION_REQUEST = ('This is an additional measurement after your completed clinical '
    'submission. Keep that original submission fixed; do not revise the diagnosis, '
    'request tools, or emit another clinical Decision JSON. For this measurement only, '
    'answer the following question with exactly one uppercase letter A or B, '
    'without explanation or formatting.\n\n')

REPLAY_REQUEST = ('This is an independent measurement from a saved clinical history. '
    'Judge the supplied fixed proposed diagnosis using only the evidence visible in '
    'this history. Do not revise the original clinical trajectory, request tools, '
    'or emit another clinical Decision JSON. For this measurement only, answer '
    'exactly one uppercase letter A or B, without explanation or formatting.\n\n')
REPLAY_FIELDS = ('source_message_end', 'batch_size', 'evidence_types', 'tool_failures',
                 'duplicate_results', 'diagnostic_impression')


def selected_probe_kinds(plan):
    kinds = plan.get('probe_kinds', ['sufficient', 'true'])
    if (not isinstance(kinds, list) or not kinds or
            not all(isinstance(k, str) and k in {'sufficient', 'true'} for k in kinds) or
            len(set(kinds)) != len(kinds)):
        raise ValueError('probe_kinds must be a non-empty unique list of sufficient/true')
    return [k for k in ['sufficient', 'true'] if k in kinds]


def _validate_replay_history(history):
    if not history:
        raise ValueError('replay requires a saved original history')
    pending = set()
    seen = set()
    for message in history:
        calls = message.get('tool_calls') or []
        if calls:
            if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
                raise ValueError('replay tool calls must be a list of objects')
            ids = [c.get('id') for c in calls]
            if (pending or message.get('role') != 'assistant' or
                    any(not isinstance(i, str) or not i for i in ids) or
                    len(ids) != len(set(ids)) or seen.intersection(ids)):
                raise ValueError('replay requires complete tool batches with unique call IDs')
            pending.update(ids); seen.update(ids)
        elif message.get('role') == 'tool':
            call_id = message.get('tool_call_id')
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError('replay tool result has no pending original call')
            pending.remove(call_id)
        elif pending:
            raise ValueError('replay tool batch is incomplete before the next message')
    if pending:
        raise ValueError('replay tool batch is incomplete')


def measurement_snapshots(histories, design='evidence_prefix'):
    if design == 'evidence_prefix':
        return extract_snapshots(histories['interactive'])
    if design == 'tool_history_replay':
        history = histories['interactive']
        _validate_replay_history(history)
        snapshots = [s for s in extract_snapshots(history)
                     if s['kind'] in {'initial', 'after_tool_batch'}]
        for step, snapshot in enumerate(snapshots):
            snapshot.update(step=step, source_view='interactive',
                            measurement_design='tool_history_replay',
                            progress=step / (len(snapshots) - 1) if len(snapshots) > 1 else 0.0)
        return snapshots
    if design != 'post_submission':
        raise ValueError('unsupported measurement design')
    snapshots = []
    for step, view in enumerate(['hpi', 'interactive']):
        history = histories[view]
        if not history or history[-1].get('role') != 'assistant' or history[-1].get('tool_calls'):
            raise ValueError('post-submission history must end with the original assistant submission')
        snapshots.append({'history': copy.deepcopy(history), 'step': step,
                          'kind': 'post_submission', 'source_view': view,
                          'prefix_sha256': digest(history)})
    return snapshots


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def candidate(row):
    answer = (row.get('decision') or {}).get('answer')
    value = answer.get('diagnosis') if isinstance(answer, dict) else answer
    return value if isinstance(value, str) and value.strip() else None


def validate_sources(plan, model_config=None):
    if plan['schema'] not in ORIGINAL_SCHEMAS or plan.get('workers') != 1:
        raise ValueError('original-agent probes require a supported schema and one worker')
    selected_probe_kinds(plan)
    sources = {}
    for view in ['hpi', 'interactive']:
        source = plan['original_sources'][view]
        for path, expected in source['hashes'].items():
            if file_hash(path) != expected:
                raise ValueError('original AMA source changed: ' + path)
        root = Path(source['run'])
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('mode') != 'run' or manifest.get('output_contract_version') != 2:
            raise ValueError('original submission must be a standard AMA Decision run')
        if model_config is not None and (manifest['model_config']['model'] != model_config['model'] or
                manifest['model_config']['base_url'] != model_config['base_url']):
            raise ValueError('probe model/provider differs from original generation')
        rows = read_rows(root/'decisions.jsonl')
        indexed = {r['episode_id']: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(manifest['episode_ids']):
            raise ValueError('expected exactly one original submission per admission')
        sources[view] = {'rows': indexed, 'manifest': manifest,
                         'histories': json.loads((root/'messages.json').read_text())}
    if set(sources['hpi']['rows']) != set(sources['interactive']['rows']):
        raise ValueError('HPI and interactive source cohorts differ')
    return sources


def probe_history(snapshot, kind, diagnosis):
    history = copy.deepcopy(snapshot['history'])
    if not history or history[0].get('role') != 'system' or not isinstance(history[0].get('content'), str):
        raise ValueError('original system message is required')
    # Legacy evidence-prefix mode keeps its original system exception. New
    # measurements preserve the entire source prefix and append only a user turn.
    post_submission = snapshot.get('kind') == 'post_submission'
    replay = snapshot.get('measurement_design') == 'tool_history_replay'
    if not post_submission and not replay:
        history[0]['content'] += MODE_EXCEPTION
    template = SUFFICIENT if kind == 'sufficient' else PTRUE
    prompt = template.format(question='', answer=diagnosis).strip()
    if post_submission:
        prompt = POST_SUBMISSION_REQUEST + prompt
    elif replay:
        prompt = REPLAY_REQUEST + prompt
    return history + [{'role': 'user', 'content': prompt}]


def job_list(snapshots, sources, base_config, plan):
    jobs, missing = [], []
    kinds = selected_probe_kinds(plan)
    design = plan.get('measurement_design', 'evidence_prefix')
    if design not in {'evidence_prefix', 'post_submission', 'tool_history_replay'}:
        raise ValueError('unsupported measurement design')
    cfg = dict(base_config, logprobs=True, top_logprobs=plan['top_logprobs'],
               max_tokens=plan.get('max_probe_tokens', 32))
    for s in snapshots:
        post_submission = design == 'post_submission'
        replay = design == 'tool_history_replay'
        if replay != (s.get('measurement_design') == 'tool_history_replay'):
            raise ValueError('snapshot and measurement design differ')
        if post_submission != (s.get('kind') == 'post_submission'):
            raise ValueError('snapshot and measurement design differ')
        if post_submission:
            targets = [(k, 'original_hpi' if s['source_view'] == 'hpi' else 'original_final') for k in kinds]
        elif replay:
            targets = [(k, 'original_final_hindsight') for k in kinds]
        else:
            targets = [(k, target) for k, target in TARGETS if k in kinds]
        for kind, target in targets:
            view = s['source_view'] if post_submission else ('hpi' if target == 'original_hpi' else 'interactive')
            diagnosis = candidate(sources[view]['rows'][s['episode_id']])
            if diagnosis is None:
                missing.append({'episode_id': s['episode_id'], 'turn_id': s['turn_id'],
                                'kind': kind, 'target': target, 'reason': 'no_original_diagnosis'})
                continue
            history = probe_history(s, kind, diagnosis)
            identity = {k: s[k] for k in ['episode_id', 'turn_id', 'step', 'prefix_sha256']}
            identity.update(kind=kind, target=target, candidate=diagnosis,
                            request_sha256=digest(history), model_config=cfg,
                            source_decision_sha256=digest(sources[view]['rows'][s['episode_id']]),
                            measurement_mode_sha256=digest(POST_SUBMISSION_REQUEST if post_submission else
                                                          REPLAY_REQUEST if replay else MODE_EXCEPTION))
            if replay:
                identity.update({k: s[k] for k in REPLAY_FIELDS})
                identity.update(source_view='interactive', measurement_design='tool_history_replay')
            jobs.append({**identity, 'job_id': digest(identity), '_history': history})
    return jobs, missing


def original_label_probability(metadata, readout='single_label'):
    if readout == 'single_label':
        return binary_probability(metadata)
    if readout != 'first_complete_label':
        raise ValueError('unsupported original-agent label readout')
    text = (metadata or {}).get('logprob_content')
    records = ((metadata or {}).get('logprobs') or {}).get('content') or []
    try:
        first = bytes(records[0]['bytes']).decode()
    except (IndexError, KeyError, TypeError, ValueError, UnicodeError):
        return {'valid': False, 'reason': 'missing_first_token_bytes', 'p_ab': None}
    if not isinstance(text, str) or first.strip() not in {'A', 'B'} or not text.startswith(first) or (len(text) > len(first) and not text[len(first)].isspace()):
        return {'valid': False, 'reason': 'first_token_not_complete_binary_label', 'p_ab': None}
    isolated = copy.deepcopy(metadata)
    isolated['logprob_content'] = first
    result = binary_probability(isolated)
    result.update(single_label_format_compliant=text.strip() in {'A', 'B'},
                  probability_readout='first_complete_label',
                  trailing_output_present=bool(text[len(first):].strip()))
    return result

def validate_measurement_snapshots(snapshots, sources, plan):
    design = plan.get('measurement_design', 'evidence_prefix')
    for ep in {s['episode_id'] for s in snapshots}:
        actual = sorted([s for s in snapshots if s['episode_id'] == ep], key=lambda s: s['step'])
        frozen = measurement_snapshots({v: sources[v]['histories'][ep] for v in sources},
                                       design)
        if len(actual) != len(frozen):
            raise ValueError('prefix count differs from original AMA history')
        for a, b in zip(actual, frozen):
            if a['history'] != b['history'] or a['prefix_sha256'] != digest(b['history']):
                raise ValueError('prefix is not the original native AMA history')
            if design == 'post_submission' and any(a.get(k) != b[k] for k in ['step', 'kind', 'source_view']):
                raise ValueError('post-submission source view changed')
            if design == 'tool_history_replay' and any(a.get(k) != b[k] for k in
                    ['step', 'kind', 'source_view', 'measurement_design', 'progress', *REPLAY_FIELDS]):
                raise ValueError('replay boundary or source metadata changed')


def execute_original_agent_probing(dataset_dir, model, plan_path, *, runs_root, resume=None):
    dataset = load_dataset(resolve_dataset_dir(dataset_dir))
    validate_visible_files(dataset)
    plan = json.loads(Path(plan_path).read_text())
    sources = validate_sources(plan, model.config)
    if file_hash(plan['snapshots_file']) != plan['snapshots_sha256']:
        raise ValueError('frozen prefix file changed')
    snapshots = read_rows(plan['snapshots_file'])
    expected = {(ep.id, t.id) for ep in dataset.episodes for t in ep.turns}
    if {(s['episode_id'], s['turn_id']) for s in snapshots} != expected or len(snapshots) != len(expected):
        raise ValueError('snapshot IDs do not match expected turns')
    validate_measurement_snapshots(snapshots, sources, plan)
    jobs, missing = job_list(snapshots, sources, model.config, plan)
    if resume:
        root = Path(resume)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['plan_sha256'] != digest(plan) or manifest['model_config'] != model.config:
            raise ValueError('resume plan/model mismatch')
    else:
        recorder = Recorder(runs_root, dataset=dataset, episodes=dataset.episodes, mode='run', model=model, max_calls=1)
        root, manifest = recorder.run_dir, recorder.manifest
        manifest.update(probing_plan=str(Path(plan_path).resolve()), plan_sha256=digest(plan),
                        output_contract_version=ORIGINAL_CONTRACT, source_decisions_reused=True)
        write_json(root/'manifest.json', manifest)
        (root/'probe_calls.jsonl').touch()
        # No invented prefix Decisions. Original submissions have their own exact archive.
        for view in ['hpi', 'interactive']:
            archive = root/'original_submissions'/view
            archive.mkdir(parents=True)
            for name in ['decisions.jsonl', 'manifest.json']:
                (archive/name).write_bytes((Path(plan['original_sources'][view]['run'])/name).read_bytes())
        write_json(root/'messages.json', {s['episode_id']+'/'+s['turn_id']: s['history'] for s in snapshots})
    (root/'requests').mkdir(exist_ok=True)
    write_json(root/'missing_original_candidates.json', missing)
    lock = (root/'process.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = Path(plan['transport']['lease_path']).open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def append(name, row):
        with (root/name).open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    transport = ProbeTransport(plan['transport'], lambda row: append('transport_attempts.jsonl', row))
    completed = {r['job_id']: r for r in read_rows(root/'probe_calls.jsonl')}
    identities = {j['job_id']: {k: v for k, v in j.items() if k != '_history'} for j in jobs}
    if not resume:
        for previous in plan.get('reuse_runs', []):
            prior_manifest = json.loads((Path(previous)/'manifest.json').read_text())
            if prior_manifest.get('output_contract_version') not in ORIGINAL_CONTRACTS:
                raise ValueError('cannot reuse older rediagnosis experiment')
            for row in read_rows(Path(previous)/'probe_calls.jsonl'):
                identity = identities.get(row['job_id'])
                if identity and all(row.get(k) == v for k, v in identity.items()) and not row.get('error') and original_label_probability(row.get('response_metadata'), plan.get('label_readout', 'single_label'))['valid'] and row['job_id'] not in completed:
                    write_json(root/'requests'/(row['request_sha256']+'.json'), next(j['_history'] for j in jobs if j['job_id'] == row['job_id']))
                    row = {**row, 'reused_from': str(previous)}
                    completed[row['job_id']] = row
                    append('probe_calls.jsonl', row)
    started = {r['job_id'] for r in read_rows(root/'diagnostics.jsonl') if r.get('kind') == 'probe_start'}
    print(f'records: {root}', flush=True)
    try:
        for job in jobs:
            if job['job_id'] in completed:
                continue
            if job['job_id'] in started:
                raise ModelError('unresolved in-flight job; no automatic resubmission')
            history = job.pop('_history')
            write_json(root/'requests'/(job['request_sha256']+'.json'), history)
            append('diagnostics.jsonl', {'kind': 'probe_start', **job, 'time': time.time()})
            local = PacedProbeModel(model.name, job['model_config'], model.timeout, transport, job['job_id'])
            agent = Agent(local, max_calls=1)
            agent.history = copy.deepcopy(history[:-1])
            began = time.monotonic()
            reply, error = None, None
            try:
                reply = agent.chat(history[-1]['content'])
            except ModelError as exc:
                error = str(exc)
            row = {**job, 'reply': reply, 'error': error, 'termination': 'api_error' if error else 'completed',
                   'usage': local.last_usage, 'response_metadata': local.last_response_metadata,
                   'http_attempts': len(local.transport_attempts), 'duration_ms': round((time.monotonic()-began)*1000)}
            append('probe_calls.jsonl', row)
            completed[job['job_id']] = row
            write_json(root/'progress.json', {'completed_jobs': len(completed), 'expected_jobs': len(jobs),
                       'expected_missing_candidates': len(missing), 'termination': 'running'})
            if error:
                raise ModelError(error)
            bp = original_label_probability(row['response_metadata'], plan.get('label_readout', 'single_label'))
            if not bp['valid']:
                raise ModelError('binary coverage failed; stopped before next request: '+str(bp['reason']))
            if row['response_metadata'].get('model') != model.config['model']:
                raise ModelError('response model mismatch')
        manifest['termination'] = 'completed'
    except BaseException:
        manifest['termination'] = 'interrupted'
        raise
    finally:
        write_json(root/'manifest.json', manifest)
        write_json(root/'progress.json', {'completed_jobs': len(completed), 'expected_jobs': len(jobs),
                   'expected_missing_candidates': len(missing), 'termination': manifest['termination']})
        lease.close()
        lock.close()
    return root


def evaluate_original_agent_probing(run_dir, dataset, manifest, scorer_override=None):
    from .scorer import REGISTRY
    from .decisions import output_metrics
    from .scorer import expected_rows
    if scorer_override not in (None, 'original_agent_probing'):
        raise ValueError('original-agent run requires original_agent_probing scorer')
    plan = json.loads(Path(manifest['probing_plan']).read_text())
    if digest(plan) != manifest['plan_sha256'] or file_hash(plan['snapshots_file']) != plan['snapshots_sha256']:
        raise ValueError('scoring plan/prefix hash mismatch')
    sources = validate_sources(plan, manifest['model_config'])
    for source in plan['original_sources'].values():
        for path, expected in source.get('eval_hashes', {}).items():
            if file_hash(path) != expected: raise ValueError('original evaluator input changed: '+path)
    snapshots = read_rows(plan['snapshots_file'])
    validate_measurement_snapshots(snapshots, sources, plan)
    expected_jobs, missing = job_list(snapshots, sources, manifest['model_config'], plan)
    rows = read_rows(run_dir/'probe_calls.jsonl')
    indexed = {r['job_id']: r for r in rows}
    if len(indexed) != len(rows) or not set(indexed) <= {j['job_id'] for j in expected_jobs}:
        raise ValueError('duplicate/unexpected probe jobs')
    ids = [ep.id for ep in dataset.episodes]
    source_scores, comparison = {}, {}
    for view in ['hpi', 'interactive']:
        archive = run_dir/'original_submissions'/view/'decisions.jsonl'
        if file_hash(archive) != plan['original_sources'][view]['hashes'][str(Path(plan['original_sources'][view]['run'])/'decisions.jsonl')]:
            raise ValueError('archived original decisions changed')
        original_dataset = load_dataset(Path(sources[view]['manifest']['dataset_dir']), with_targets=True)
        original_dataset.episodes = [original_dataset.select(episode_id=ep)[0] for ep in sources[view]['manifest']['episode_ids']]
        decisions = {ep: [r] for ep, r in sources[view]['rows'].items()}
        full = REGISTRY['mimic_cdm_open'](original_dataset, decisions)
        baseline = json.loads((Path(plan['original_sources'][view]['run'])/'metrics.json').read_text())['scored']
        differences = [ep for ep in full['per_episode'] if full['per_episode'][ep] != baseline['per_episode'][ep]]
        comparison[view] = {'cohort': len(full['per_episode']), 'per_episode_differences': differences,
                            'exact_historical_match': not differences}
        selected = copy.copy(original_dataset)
        selected.episodes = [original_dataset.select(episode_id=ep)[0] for ep in ids]
        result = REGISTRY['mimic_cdm_open'](selected, decisions)
        result['aggregate']['output_quality'] = output_metrics([r for ep in selected.episodes for r in expected_rows(ep, decisions)], 2)
        source_scores[view] = result
        out = run_dir/'original_ama_eval'/view
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'metrics.json', {'scorer': 'mimic_cdm_open', 'source_run': plan['original_sources'][view]['run'],
                                       'original_decisions_reused': True, 'scored': result, 'full_cohort_accuracy': full['aggregate']['diagnosis_accuracy_auto']})
    features = []
    for s in snapshots:
        entry = {k: s[k] for k in ['episode_id', 'turn_id', 'step', 'prefix_sha256']}
        entry.update({k: s[k] for k in ['progress', 'kind', 'source_view', 'measurement_design', *REPLAY_FIELDS] if k in s})
        entry.update(probes={})
        for j in [j for j in expected_jobs if j['episode_id'] == s['episode_id'] and j['turn_id'] == s['turn_id']]:
            r = indexed.get(j['job_id'], {})
            if r and any(r.get(k) != v for k, v in j.items() if k != '_history'):
                raise ValueError('probe identity/candidate/config changed')
            if r and json.loads((run_dir/'requests'/(j['request_sha256']+'.json')).read_text()) != j['_history']:
                raise ValueError('saved request differs from frozen native prefix')
            bp = original_label_probability(r.get('response_metadata'), plan.get('label_readout', 'single_label'))
            if r.get('error') or (r.get('response_metadata') or {}).get('model') != manifest['model_config']['model']:
                bp.update(valid=False, reason='missing_or_failed_or_model_mismatch', p_ab=None)
            entry['probes'][j['kind']+'/'+j['target']] = {**bp, 'candidate': j['candidate'], 'job_id': j['job_id']}
        features.append(entry)
    endpoints = []
    for ep in ids:
        a, b = source_scores['hpi']['per_episode'][ep], source_scores['interactive']['per_episode'][ep]
        group = ('consistent' if a['predicted_pathology'] == b['predicted_pathology'] else 'inconsistent') if a['mapping'] == b['mapping'] == 'mapped' else 'unresolved'
        ep_features = sorted([f for f in features if f['episode_id'] == ep], key=lambda f: f['step'])
        endpoints.append({'episode_id': ep, 'agreement': group, 'hpi': a, 'interactive': b,
                          'first_turn_id': ep_features[0]['turn_id'], 'final_turn_id': ep_features[-1]['turn_id']})
    probes = [p for f in features for p in f['probes'].values()]
    usage = [r['usage'] for r in rows if not r.get('reused_from') and isinstance(r.get('usage'), dict)]
    result = {'scorer': 'original_agent_probing', 'measurement_design': plan.get('measurement_design', 'evidence_prefix'), 'original_ama_eval': source_scores,
              'historical_comparison': comparison, 'per_prefix': features, 'endpoints': endpoints,
              'missing_original_candidates': missing, 'aggregate': {
                  'admissions': len(ids), 'prefixes': len(snapshots), 'expected_probes': len(expected_jobs),
                  'recorded_probes': len(rows), 'valid_probes': sum(p['valid'] for p in probes),
                  'invalid_probes': sum(not p['valid'] for p in probes), 'missing_original_candidates': len(missing),
                  'noncompliant_single_label_replies': sum(p.get('trailing_output_present', False) for p in probes),
                  'api_errors': sum(bool(r.get('error')) for r in rows), 'reused_jobs': sum(bool(r.get('reused_from')) for r in rows),
                  'new_http_attempts': sum(r.get('http_attempts', 0) for r in rows if not r.get('reused_from')),
                  'total_tokens': sum(u.get('total_tokens', 0) for u in usage),
                  'unknown_usage_jobs': sum(not r.get('reused_from') and not isinstance(r.get('usage'), dict) for r in rows)}}
    write_json(run_dir/'metrics.json', {'run_id': manifest['run_id'], 'model': manifest['model'],
                                      'scorer': 'original_agent_probing', 'scored': result})
    print(json.dumps(result['aggregate'], indent=2))
    return 0
