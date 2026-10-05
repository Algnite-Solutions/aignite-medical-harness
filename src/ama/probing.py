"""Native AMA sidecar probes: frozen prefixes, independent forks, resumable calls.

Inference runs only under ama run; probabilities/consistency are scored by ama eval.
No probe answer is ever inserted into a trajectory being observed.
"""
from __future__ import annotations
import concurrent.futures
import copy
import hashlib
import json
import math
import re
import threading
import time
import unicodedata
from pathlib import Path

from .probing_transport import ProbeTransport, PacedProbeModel
from .agent import Agent
from .data import load_dataset, resolve_dataset_dir, validate_visible_files
from .model import Model, ModelError
from .recorder import Recorder, write_json

SUFFICIENT = '''{question}
Based only on the information and hints provided above, does that information sufficiently entail that the correct answer is exactly {answer}?
A. Yes — the information is sufficient to conclude {answer}.
B. No — the information is insufficient, allows alternatives, or contradicts {answer}.
Output exactly one uppercase letter: A or B. Do not include Markdown, whitespace, punctuation, or explanation:'''
PTRUE = '''{question}
Proposed Answer: {answer}
Is the proposed answer:
A. True
B. False
Output exactly one uppercase letter: A or B. Do not include Markdown, whitespace, punctuation, or explanation:'''
CLINICAL = '''{question}
Proposed diagnosis: {answer}
Based only on the patient evidence provided, is the evidence sufficient to commit to this diagnosis, while reasonably excluding competing diagnoses that require different management?
A. Yes — the current evidence is sufficient to support a definite diagnosis of {answer}.
B. No — the current evidence is insufficient, permits relevant alternatives, or contradicts {answer}.
Output exactly one uppercase letter: A or B. Do not include Markdown, whitespace, punctuation, or explanation:'''
DIAGNOSIS = '''{question}
Based only on the patient evidence available above, give your single best current diagnosis. If information is incomplete, give your best provisional diagnosis. Return one concise English diagnosis phrase only, with no explanation, probability, JSON, alternatives, or tool call.'''


TEMPLATES = {'diagnosis': DIAGNOSIS, 'sample': DIAGNOSIS, 'sufficient': SUFFICIENT, 'true': PTRUE, 'clinical': CLINICAL}

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize_diagnosis(text):
    if not isinstance(text, str) or not text.strip():
        return None
    if '\n' in text.strip() or len(text.split()) > 40 or '<think>' in text or '</think>' in text:
        return None
    # Frozen lexical matching; no target-based mapping or LLM judge.
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def binary_probability(metadata):
    result = {'valid': False, 'reason': None, 'p_A_provider': None,
              'p_B_provider': None, 'ab_mass': None, 'p_ab': None}
    if not isinstance(metadata, dict):
        result['reason'] = 'missing_metadata'
        return result
    text = metadata.get('logprob_content')
    logprobs = metadata.get('logprobs')
    records = logprobs.get('content') if isinstance(logprobs, dict) else None
    if not isinstance(text, str) or text.strip() not in {'A', 'B'}:
        result['reason'] = 'not_single_label'
        return result
    result['label'] = text.strip()
    if not isinstance(records, list) or not records:
        result['reason'] = 'missing_logprobs'
        return result
    first = records[0]
    try:
        raw_bytes = bytes(first['bytes'])
        token_text = raw_bytes.decode()
    except (KeyError, TypeError, ValueError, UnicodeError):
        result['reason'] = 'missing_token_bytes'
        return result
    result['first_token'] = first.get('token')
    result['first_token_bytes'] = list(raw_bytes)
    result['chosen_logprob'] = first.get('logprob')
    if token_text.strip() != text.strip() or not text.encode().startswith(raw_bytes) or text[len(token_text):].strip():
        result['reason'] = 'label_not_first_complete_token'
        return result
    candidates = first.get('top_logprobs') or []
    values = {}
    for item in [*candidates, first]:
        token, lp = item.get('token'), item.get('logprob')
        if not isinstance(token, str) or type(lp) not in (int, float) or not math.isfinite(lp):
            continue
        if lp > 0:
            result['reason'] = 'invalid_positive_logprob'
            return result
        if token in values and values[token] != lp:
            result['reason'] = 'conflicting_duplicate_token'
            return result
        values[token] = lp
    groups = {label: {t: p for t, p in values.items() if t.strip().upper() == label} for label in ['A', 'B']}
    result['candidates'] = groups
    result['top_candidates_returned'] = len(candidates)
    result['exact_uppercase_logprobs'] = {label: values.get(label) for label in ['A', 'B']}
    result['exact_uppercase_provider_p_A'] = math.exp(values['A']) if 'A' in values else None
    if not all(groups.values()):
        result['reason'] = 'missing_A_or_B_no_imputation'
        return result
    pivot = max(p for group in groups.values() for p in group.values())
    stable = {label: sum(math.exp(p - pivot) for p in group.values()) for label, group in groups.items()}
    masses = {label: sum(math.exp(p) for p in group.values()) for label, group in groups.items()}
    result.update(valid=True, p_A_provider=masses['A'], p_B_provider=masses['B'],
                  ab_mass=sum(masses.values()), p_ab=stable['A'] / sum(stable.values()))
    return result


def binary_run_metrics(run_dir):
    rows = []
    for line in (Path(run_dir) / 'diagnostics.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') == 'model_call':
            rows.append({'episode_id': row['episode_id'], 'turn_id': row['turn_id'],
                         **binary_probability(row.get('response_metadata'))})
    return {'calls': len(rows), 'new_http_attempts': sum(r.get('model_calls', 0) for r in rows if not r.get('reused_from')),
        'reused_jobs': sum(bool(r.get('reused_from')) for r in rows), 'valid': sum(r['valid'] for r in rows), 'rows': rows}


def content_text(message):
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(part.get('text', '') for part in content if isinstance(part, dict))
    return ''


def context_fragments(history):
    fragments = []
    for message in history:
        if message.get('role') == 'system':
            continue
        text = content_text(message)
        # Remove only known transport directives; preserve visible clinical text.
        text = '\n'.join(line for line in text.splitlines()
                         if not line.startswith('Final Decision turn_id:'))
        if message.get('tool_calls'):
            requests = [request.get('function') for request in message['tool_calls']]
            text += '\nTool requests: ' + json.dumps(requests, ensure_ascii=False)
        if text.strip():
            fragments.append((message.get('role', 'unknown'), text))
    return fragments


def render_context(snapshot, variant):
    full = context_fragments(snapshot['history'])
    initial = context_fragments(snapshot['initial_history'])
    additions = full[len(initial):]
    if variant == 'summary':
        # Lossless mechanical concatenation: same exact text fragments, no rewrite.
        return '\n\n'.join(text for _, text in full)
    if variant in {'placebo', 'repeat'}:
        repeated = '\n\n'.join(text for _, text in initial)
        if variant == 'repeat':
            additions = [(role, 'Repeated previously available information:\n' + repeated) for role, _ in additions]
        else:
            additions = [(role, 'No additional patient information is available at this turn.') for role, _ in additions]
        full = initial + additions
    return '\n\n'.join(role.title() + ': ' + text for role, text in full)


def extract_snapshots(history):
    """Historical tool batches are atomic; never invent within-batch timestamps."""
    first_user = next((i for i, message in enumerate(history) if message.get('role') == 'user'), None)
    if first_user is None:
        raise ValueError('trajectory has no initial user observation')
    initial = copy.deepcopy(history[:first_user + 1])
    snapshots = [{'history': initial, 'initial_history': initial, 'source_message_end': first_user + 1,
                  'batch_size': 0, 'evidence_types': ['hpi'], 'tool_failures': 0,
                  'duplicate_results': 0, 'diagnostic_impression': False, 'kind': 'initial'}]
    requests = {}
    seen = set()
    index = first_user + 1
    while index < len(history):
        message = history[index]
        for request in message.get('tool_calls') or []:
            requests[request['id']] = request.get('function', {})
        if message.get('role') != 'tool':
            index += 1
            continue
        end = index
        batch = []
        while end < len(history) and history[end].get('role') == 'tool':
            batch.append(history[end]); end += 1
        # Include actual post-tool attachments; no result is split into fake turns.
        if end < len(history) and history[end].get('role') == 'user' and isinstance(history[end].get('content'), list):
            end += 1
        types, failures, duplicates, impression = [], 0, 0, False
        for result in batch:
            function = requests.get(result.get('tool_call_id'), {})
            name = function.get('name', 'unknown')
            text = content_text(result)
            failures += int(bool(re.match(r'\s*\{\s*"error"\s*:', text)))
            fingerprint = digest([name, function.get('arguments'), text])
            duplicates += int(fingerprint in seen)
            seen.add(fingerprint)
            types.append({'imaging': 'imaging', 'laboratory_results': 'laboratory',
                          'physical_examination': 'examination', 'microbiology': 'microbiology',
                          'list_imaging': 'catalog'}.get(name, 'other'))
            impression |= name == 'imaging' and bool(re.search(r'\b(impression|conclusion)\b', text, re.I))
        snapshots.append({'history': copy.deepcopy(history[:end]), 'initial_history': initial,
            'source_message_end': end, 'batch_size': len(batch), 'evidence_types': types,
            'tool_failures': failures, 'duplicate_results': duplicates,
            'diagnostic_impression': impression, 'kind': 'after_tool_batch'})
        index = end
    # Identify the actual final submission boundary, excluding its answer.
    finals = [i for i, m in enumerate(history) if m.get('role') == 'assistant' and not m.get('tool_calls')]
    if finals:
        final_end = finals[-1]
        if final_end == snapshots[-1]['source_message_end']:
            snapshots[-1]['final_before_submission'] = True
        elif final_end > snapshots[-1]['source_message_end']:
            snapshots.append({'history': copy.deepcopy(history[:final_end]), 'initial_history': initial,
                'source_message_end': final_end, 'batch_size': 0, 'evidence_types': ['other'],
                'tool_failures': 0, 'duplicate_results': 0, 'diagnostic_impression': False,
                'kind': 'before_final_submission', 'final_before_submission': True})
    for step, snapshot in enumerate(snapshots):
        snapshot['step'] = step
        snapshot['prefix_sha256'] = digest(snapshot['history'])
        snapshot['progress'] = step / max(1, len(snapshots) - 1)
    return snapshots


def probe_config(base, plan, job_id, kind):
    cfg = dict(base)
    cfg.update(temperature=1.0 if kind == 'sample' else 0.0,
               max_tokens=plan.get('max_probe_tokens', 128))
    if plan.get('provider_seed_supported'):
        cfg['seed'] = int(digest([plan['seed'], job_id])[:8], 16)
    if plan.get('top_p') is not None:
        cfg['top_p'] = plan['top_p']
    if kind == 'sample' and plan.get('sample_batch_size', 1) > 1:
        cfg['n'] = plan['sample_batch_size']
    if kind in {'sufficient', 'true', 'clinical'}:
        cfg.update(logprobs=True, top_logprobs=plan['top_logprobs'])
    else:
        cfg.pop('logprobs', None); cfg.pop('top_logprobs', None)
    return cfg


def execute_probing(dataset_dir, model, plan_path, *, runs_root, resume=None):
    import fcntl
    dataset = load_dataset(resolve_dataset_dir(dataset_dir))
    validate_visible_files(dataset)
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('schema') != 'ama-interactive-probing-v1' or plan.get('sc_samples') != 20:
        raise ValueError('unsupported probing plan or SC sample count')
    if plan.get('sample_batch_size', 1) not in (1, 4, 10, 20):
        raise ValueError('sample_batch_size must divide 20')
    if not 1 <= plan.get('workers', 1) <= 16:
        raise ValueError('probing workers must be 1..16')
    scope = plan.get('job_scope', 'full')
    if scope not in {'core', 'consistency', 'full'}:
        raise ValueError('job_scope must be core, consistency or full')
    paced = 'transport' in plan
    if paced and plan.get('workers', 1) != 1:
        raise ValueError('paced probing requires exactly one worker')
    source_path = Path(plan['snapshots_file'])
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != plan['snapshots_sha256']:
        raise ValueError('frozen snapshots hash mismatch')
    snapshots = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    expected = {(ep.id, turn.id) for ep in dataset.episodes for turn in ep.turns}
    if {(r['episode_id'], r['turn_id']) for r in snapshots} != expected:
        raise ValueError('snapshot IDs do not match dataset expected turns')
    if any(digest(r['history']) != r['prefix_sha256'] for r in snapshots):
        raise ValueError('prefix hash mismatch')
    if resume:
        root = Path(resume)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('plan_sha256') != digest(plan) or manifest['model_config'] != model.config:
            raise ValueError('resume plan/model mismatch')
    else:
        recorder = Recorder(runs_root, dataset=dataset, episodes=dataset.episodes, mode='run', model=model,
                            max_calls=1)
        root, manifest = recorder.run_dir, recorder.manifest
        manifest.update(probing_plan=str(Path(plan_path).resolve()), plan_sha256=digest(plan),
                        output_contract_version='interactive-probing-v1')
        write_json(root / 'manifest.json', manifest)
        (root / 'probe_calls.jsonl').touch()
        write_json(root / 'messages.json', {r['episode_id'] + '/' + r['turn_id']: r['history'] for r in snapshots})
    (root / 'requests').mkdir(exist_ok=True)
    lockfile = (root / 'process.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = None
    if paced:
        lease = Path(plan['transport']['lease_path']).open('a')
        # Shared across stages/models: an accidental second launcher cannot send calls.
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def allowed(row):
        if scope == 'full': return True
        return row['variant'] == 'real' and (
            row['kind'] == 'diagnosis' or
            (row['kind'] == 'sufficient' and row['target'] in {'current', 'initial_fixed'}) or
            (row['kind'] == 'true' and row['target'] == 'current') or
            (scope == 'consistency' and row['kind'] == 'sample'))
    completed = {}
    allowed_prefixes = {(r['episode_id'], r['turn_id'], r['prefix_sha256']): r for r in snapshots}
    if not resume:
        reused_ids = set()
        for reuse in plan.get('reuse_runs', []):
            reuse = Path(reuse)
            original_manifest = json.loads((reuse / 'manifest.json').read_text())
            if original_manifest['model_config'] != model.config:
                raise ValueError('reused run has a different model configuration')
            for line in (reuse / 'probe_calls.jsonl').read_text().splitlines():
                row = json.loads(line)
                if not allowed(row) or row.get('error') or (row.get('episode_id'), row.get('turn_id'), row.get('prefix_sha256')) not in allowed_prefixes:
                    continue
                if row['kind'] == 'sample' and row['model_config'].get('n', 1) != plan.get('sample_batch_size', 1):
                    continue
                snapshot = allowed_prefixes[(row['episode_id'], row['turn_id'], row['prefix_sha256'])]
                template = TEMPLATES[row['kind']]
                prompt = template.format(question=render_context(snapshot, row['variant']), answer=row.get('candidate') or '')
                expected_request = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}]
                if row.get('request_sha256') != digest(expected_request):
                    continue
                identity = {k: row[k] for k in ['episode_id', 'turn_id', 'step', 'progress', 'prefix_sha256',
                                                'variant', 'kind', 'target', 'candidate', 'sample_index']}
                if row['kind'] == 'sample': identity['sampling_batch_size'] = plan.get('sample_batch_size', 1)
                identity['prompt_sha256'] = digest(template)
                original_jid = row['job_id']
                new_jid = digest(identity)
                if row.get('model_config') != probe_config(model.config, plan, new_jid, row['kind']):
                    continue
                row.update(identity, job_id=new_jid, source_job_id=original_jid)
                if row['job_id'] in reused_ids:
                    continue
                reused_ids.add(row['job_id'])
                row['reused_from'] = str(reuse)
                with (root / 'probe_calls.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for line in (root / 'probe_calls.jsonl').read_text().splitlines():
        row = json.loads(line); completed[row['job_id']] = row
    # A crash-in-flight is unresolved, not automatically retried/charged as zero.
    started_ids = set()
    for line in (root / 'diagnostics.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item.get('kind') == 'probe_start': started_ids.add(item['job_id'])
    io_lock = threading.Lock()
    def append(filename, value):
        with (root / filename).open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    transport = ProbeTransport(plan['transport'], lambda value: append('transport_attempts.jsonl', value)) if paced else None
    def run_job(job):
        jid = job['job_id']
        if jid in completed:
            return completed[jid]
        if jid in started_ids:
            row = {**job, 'error': 'unresolved_in_flight_from_previous_execution', 'reply': None,
                   'usage': None, 'model_calls': 0, 'termination': 'unknown'}
            with io_lock:
                append('probe_calls.jsonl', row); completed[jid] = row
            return row
        cfg = probe_config(model.config, plan, jid, job['kind'])
        context = job.pop('_context')
        if len(context) > plan.get('max_context_chars', 120000):
            row = {**job, 'error': 'context_character_limit', 'reply': None, 'usage': None, 'model_calls': 0, 'termination': 'not_executed'}
            with io_lock:
                append('probe_calls.jsonl', row); completed[jid] = row
            return row
        template = TEMPLATES[job['kind']]
        prompt = template.format(question=context, answer=job.get('candidate', ''))
        request_history = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                           {'role': 'user', 'content': prompt}]
        request_hash = digest(request_history)
        with io_lock:
            path = root / 'requests' / (request_hash + '.json')
            if not path.exists(): write_json(path, request_history)
            append('diagnostics.jsonl', {'kind': 'probe_start', 'job_id': jid,
                'request_sha256': request_hash, 'model_config': cfg, 'time': time.time()})
        local = PacedProbeModel(model.name, cfg, model.timeout, transport, jid) if transport else Model(model.name, cfg, model.timeout)
        # An independent one-call Agent, never the Agent producing the observed trajectory.
        agent = Agent(local, system='You are a helpful assistant.', max_calls=1)
        began = time.monotonic()
        reply, error = None, None
        try:
            reply = agent.chat(prompt)
        except Exception as exc:
            error = str(exc) if isinstance(exc, ModelError) else type(exc).__name__
        row = {**job, 'reply': reply, 'error': error, 'termination': 'completed' if error is None else 'api_error',
               'request_sha256': request_hash, 'model_config': cfg, 'model_calls': len(agent.calls),
               'duration_ms': round((time.monotonic() - began) * 1000),
               'usage': local.last_usage, 'response_metadata': local.last_response_metadata,
               'http_status': agent.calls[-1].get('http_status') if agent.calls else None,
               'rate_limit_headers': agent.calls[-1].get('rate_limit_headers') if agent.calls else None,
               'http_attempts': len(local.transport_attempts) if transport else len(agent.calls),
               'transport_attempts': local.transport_attempts if transport else []}
        if job['kind'] == 'sample':
            choices = (local.last_response_metadata or {}).get('choices')
            row['sample_replies'] = [c.get('message', {}).get('content') for c in sorted(choices, key=lambda c: c['index'])] if choices and error is None else [reply]
        with io_lock:
            append('probe_calls.jsonl', row)
            append('diagnostics.jsonl', {'kind': 'model_call', **row})
            completed[jid] = row
            if len(completed) % 25 == 0:
                write_json(root / 'progress.json', {'completed_jobs': len(completed),
                    'api_errors': sum(bool(v.get('error')) for v in completed.values()), 'time': time.time()})
                print(f'probe progress: {len(completed)} calls recorded', flush=True)
        return row
    def job(snapshot, variant, kind, candidate=None, target='current', sample=None):
        base = {k: snapshot[k] for k in ['episode_id', 'turn_id', 'step', 'progress', 'prefix_sha256']}
        value = {**base, 'variant': variant, 'kind': kind, 'target': target,
                 'candidate': candidate, 'sample_index': sample, 'prompt_sha256': digest(TEMPLATES[kind]),
                 **({'sampling_batch_size': plan.get('sample_batch_size', 1)} if kind == 'sample' else {}),
                 '_context': render_context(snapshot, variant)}
        value['job_id'] = digest({k: v for k, v in value.items() if k != '_context'})
        return value
    def execute_jobs(jobs):
        if plan.get('workers', 1) == 1:
            result = []
            for item in jobs:
                if transport and transport.stop_reason:
                    raise ModelError(transport.stop_reason)
                result.append(run_job(item))
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            return list(pool.map(run_job, jobs))
    grouped = {}
    for snap in snapshots: grouped.setdefault(snap['episode_id'], []).append(snap)
    selected = set()
    for ep, snaps in grouped.items():
        # Predeclared positions, based on trajectory length, never outcomes.
        for index in {0, (len(snaps) - 1) // 2, len(snaps) - 1}:
            selected.add((ep, snaps[index]['turn_id']))
    variants = ['real', 'placebo', 'repeat', 'summary'] if scope == 'full' else ['real']
    print(f'records: {root}', flush=True)
    dx = {}
    def diagnose(snapshot, variant):
        row = execute_jobs([job(snapshot, variant, 'diagnosis')])[0]
        candidate = row.get('reply') if not row.get('error') and normalize_diagnosis(row.get('reply')) else None
        dx[(snapshot['episode_id'], snapshot['turn_id'], variant)] = candidate
        return candidate
    try:
        # Case -> prefix -> variant -> probes. All requests start with identical
        # evidence text before the task suffix, keeping shared prefixes adjacent.
        for ep, snaps in grouped.items():
            first, final = snaps[0]['turn_id'], snaps[-1]['turn_id']
            if scope == 'full':
                # Hindsight anchor is obtained independently and never injected into agent state.
                diagnose(snaps[-1], 'real')
            for s in snaps:
                tid = s['turn_id']
                for v in variants:
                    current = diagnose(s, v)
                    jobs = []
                    if scope != 'core' and (v == 'real' or (ep, tid) in selected):
                        jobs.extend(job(s, v, 'sample', sample=i) for i in range(0, 20, plan.get('sample_batch_size', 1)))
                    anchors = [('current', current)]
                    if v == 'real':
                        anchors += [('initial_fixed', dx.get((ep, first, 'real')))]
                        if scope == 'full':
                            anchors += [('final_fixed_hindsight', dx.get((ep, final, 'real')))]
                    else:
                        anchors += [('real_current_fixed', dx.get((ep, tid, 'real')))]
                    for target, candidate in anchors:
                        if candidate:
                            for kind in ['sufficient', 'true']:
                                item = job(s, v, kind, candidate, target)
                                if allowed(item): jobs.append(item)
                    if scope == 'full' and (ep, tid) in selected and current:
                        jobs.append(job(s, v, 'clinical', current))
                    execute_jobs(jobs)
            append('stage_checkpoints.jsonl', {'episode_id': ep, 'job_scope': scope,
                                             'completed_jobs': len(completed), 'time': time.time()})
        # Standard decisions retain current diagnosis per true prefix. No gold read.
        (root / 'decisions.jsonl').write_text('')
        for s in snapshots:
            r = completed.get(job(s, 'real', 'diagnosis')['job_id'], {})
            answer = r.get('reply') if normalize_diagnosis(r.get('reply')) else None
            append('decisions.jsonl', {'episode_id': s['episode_id'], 'turn_id': s['turn_id'],
                'decision': {'turn_id': s['turn_id'], 'answer': {'diagnosis': answer}, 'citations': [],
                    'reasoning_summary': ''} if answer else None,
                'termination': r.get('termination', 'not_executed'), 'error': r.get('error'),
                'usage': r.get('usage'), 'model_calls': r.get('model_calls', 0), 'duration_ms': r.get('duration_ms', 0)})
        manifest['termination'] = 'completed_with_errors' if any(r.get('error') for r in completed.values()) else 'completed'
    except BaseException:
        manifest['termination'] = 'interrupted'
        raise
    finally:
        write_json(root / 'manifest.json', manifest)
        write_json(root / 'progress.json', {'completed_jobs': len(completed),
            'api_errors': sum(bool(v.get('error')) for v in completed.values()), 'termination': manifest['termination']})
        lockfile.close()
        if lease: lease.close()
    return root


def evaluate_probing(run_dir, dataset, manifest, decisions, scorer_override=None):
    if scorer_override not in (None, 'interactive_probing'):
        raise ValueError('probing runs require interactive_probing scorer')
    plan = json.loads(Path(manifest['probing_plan']).read_text())
    snapshots = [json.loads(line) for line in Path(plan['snapshots_file']).read_text().splitlines()]
    if digest(plan) != manifest['plan_sha256'] or hashlib.sha256(Path(plan['snapshots_file']).read_bytes()).hexdigest() != plan['snapshots_sha256']:
        raise ValueError('scoring input hash mismatch')
    rows = [json.loads(line) for line in (run_dir / 'probe_calls.jsonl').read_text().splitlines()]
    indexed = {}
    for r in rows:
        key = (r['episode_id'], r['turn_id'], r['variant'])
        indexed.setdefault(key, []).append(r)
    features = []
    for s in snapshots:
        for variant in (['real', 'placebo', 'repeat', 'summary'] if plan.get('job_scope', 'full') == 'full' else ['real']):
            key = (s['episode_id'], s['turn_id'], variant)
            calls = indexed.get(key, [])
            diagnosis = next((r.get('reply') for r in calls if r['kind'] == 'diagnosis' and not r.get('error')), None)
            samples = [reply for r in sorted((r for r in calls if r['kind'] == 'sample'), key=lambda x: x['sample_index']) for reply in (r.get('sample_replies') or [r.get('reply')])]
            canonical = [normalize_diagnosis(t) for t in samples]
            valid_samples = sum(t is not None for t in canonical)
            candidate = normalize_diagnosis(diagnosis)
            sc = sum(t == candidate for t in canonical) / 20 if len(samples) == 20 and valid_samples == 20 and candidate else None
            entry = {k: s[k] for k in ['episode_id', 'turn_id', 'step', 'progress', 'batch_size',
                'evidence_types', 'tool_failures', 'duplicate_results', 'diagnostic_impression', 'prefix_sha256']}
            entry.update(variant=variant, diagnosis=diagnosis, normalized_diagnosis=candidate,
                         samples=samples, samples_recorded=len(samples), valid_samples=valid_samples,
                         sc_exact_match=sc, context_chars=len(render_context(s, variant)), probes={})
            for r in calls:
                if r['kind'] not in {'sufficient', 'true', 'clinical'}: continue
                bp = binary_probability(r.get('response_metadata'))
                metadata = r.get('response_metadata') or {}
                if metadata.get('model') != manifest['model_config']['model'] or r.get('error'):
                    bp.update(valid=False, reason='model_mismatch_or_api_failure', p_ab=None)
                entry['probes'][r['kind'] + '/' + r['target']] = {**bp,
                    'candidate': r['candidate'], 'job_id': r['job_id'], 'model': metadata.get('model'),
                    'model_config': r['model_config'], 'finish_reason': metadata.get('finish_reason')}
            target = dataset.targets.get(s['episode_id'])
            entry['target'] = target.turns.get(s['turn_id']) if target else None
            # Frozen benchmark mapper, evaluator-only; never feed labels to a probe.
            from .importers.mimic_cdm_benchmark import _diagnosis_label
            predicted, mapping = _diagnosis_label(diagnosis)
            wanted = (entry['target'] or {}).get('answer', {}).get('diagnosis')
            entry['diagnosis_assessment'] = {'predicted_pathology': predicted, 'mapping': mapping,
                'reference_pathology': wanted, 'correct_auto': predicted == wanted if wanted is not None else None}
            features.append(entry)
    usage = [r['usage'] for r in rows if not r.get('reused_from') and isinstance(r.get('usage'), dict)]
    probes = [p for f in features for p in f['probes'].values()]
    if plan.get('encoder_python') and plan.get('encoder_path') and any(f['sc_exact_match'] is not None for f in features):
        import os, subprocess
        input_path, output_path = run_dir / 'embedding_inputs.json', run_dir / 'embedding_scores.json'
        write_json(input_path, [f['samples'] if f['sc_exact_match'] is not None else [] for f in features])
        command = [plan['encoder_python'], '-m', 'ama.probing_embeddings', str(input_path), str(output_path), '--encoder', plan['encoder_path']]
        write_json(run_dir / 'embedding_command.json', {'argv': command, 'encoder_revision': plan.get('encoder_revision')})
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        with (run_dir / 'embedding_stdout.log').open('w') as stream:
            status = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if status.returncode == 0:
            embedding = json.loads(output_path.read_text())
            if len(embedding['scores']) != len(features): raise ValueError('embedding row count mismatch')
            for feature, score in zip(features, embedding['scores']): feature['semantic_embedding_consistency'] = score
        else:
            for feature in features: feature['semantic_embedding_consistency'] = None
    else:
        for feature in features: feature['semantic_embedding_consistency'] = None
    endpoints = []
    grouped_real = {}
    for f in features:
        if f['variant'] == 'real': grouped_real.setdefault(f['episode_id'], []).append(f)
    for ep, values in grouped_real.items():
        values = sorted(values, key=lambda f: f['step'])
        first, final = values[0], values[-1]
        a, b = first['diagnosis_assessment'], final['diagnosis_assessment']
        agreement = ('consistent' if a['predicted_pathology'] == b['predicted_pathology'] else 'inconsistent') if a['mapping'] == b['mapping'] == 'mapped' else 'unresolved'
        endpoints.append({'episode_id': ep, 'first_turn_id': first['turn_id'], 'final_turn_id': final['turn_id'],
            'agreement': agreement, 'agreement_rule': 'same frozen unambiguous pathology mapping; unresolved otherwise',
            'exact_text_agreement': bool(first['normalized_diagnosis']) and first['normalized_diagnosis'] == final['normalized_diagnosis'],
            'hpi': a, 'final_prefix': b})
    endpoint_summary = {'admissions': len(endpoints), 'hpi_accuracy': None, 'final_prefix_accuracy': None,
                        'endpoint_label_coverage': sum(e['hpi']['correct_auto'] is not None and e['final_prefix']['correct_auto'] is not None for e in endpoints)}
    if endpoint_summary['endpoint_label_coverage'] == len(endpoints) and endpoints:
        endpoint_summary.update(hpi_accuracy=sum(e['hpi']['correct_auto'] for e in endpoints)/len(endpoints),
                                final_prefix_accuracy=sum(e['final_prefix']['correct_auto'] for e in endpoints)/len(endpoints))
    result = {'scorer': 'interactive_probing', 'per_prefix': features, 'endpoints': endpoints,
              'endpoint_summary': endpoint_summary, 'aggregate': {
        'patients': len(dataset.episodes), 'prefixes': len(snapshots), 'calls': len(rows),
        'new_http_attempts': sum(r.get('http_attempts', r.get('model_calls', 0)) for r in rows if not r.get('reused_from')),
        'reused_jobs': sum(bool(r.get('reused_from')) for r in rows),
        'job_scope': plan.get('job_scope', 'full'),
        'expected_real_prefixes': len(snapshots),
        'valid_real_current_sufficient': sum(f['variant'] == 'real' and f['probes'].get('sufficient/current', {}).get('valid', False) for f in features),
        'api_errors': sum(bool(r.get('error')) for r in rows), 'probe_calls': len(probes),
        'valid_probes': sum(p['valid'] for p in probes), 'invalid_probes': sum(not p['valid'] for p in probes),
        'total_tokens': sum(u.get('total_tokens', 0) for u in usage),
        'prompt_tokens': sum(u.get('prompt_tokens', 0) for u in usage),
        'completion_tokens': sum(u.get('completion_tokens', 0) for u in usage),
        'unknown_usage_jobs': sum(not r.get('reused_from') and not isinstance(r.get('usage'), dict) for r in rows), 'sc_complete_contexts': sum(f['sc_exact_match'] is not None for f in features)}}
    write_json(run_dir / 'metrics.json', {'run_id': manifest['run_id'], 'model': manifest['model'],
        'scorer': 'interactive_probing', 'scored': result})
    print(json.dumps(result['aggregate'], indent=2))
    print(f'metrics: {run_dir / "metrics.json"}')
    return 0
