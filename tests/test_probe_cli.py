"""Real CLI entrypoints with synthetic records and a mocked provider; no paid calls."""
import copy
import json
from pathlib import Path

import pytest

from ama.cli import main
from ama.model import Model
from ama.original_agent_probing import probe_history
from ama.probe import cumulative_snapshots
from ama.probing import digest
from ama.probing_transport import ProbeTransport
from test_data import write_dataset
from test_probing import metadata


def call(cid, name):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}


def history():
    return [
        {'role': 'system', 'content': 'ORIGINAL_SYSTEM'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'SYNTHETIC_HPI'}]},
        {'role': 'assistant', 'content': 'ASSISTANT_EXPLANATION', 'reasoning_content': 'PRIVATE_REASONING',
         'tool_calls': [call('pe', 'physical_examination'), call('lab', 'laboratory_results'), call('cat', 'list_imaging')]},
        {'role': 'tool', 'tool_call_id': 'pe', 'content': 'EXAM_RESULT'},
        {'role': 'tool', 'tool_call_id': 'lab', 'content': 'LAB_RESULT'},
        {'role': 'tool', 'tool_call_id': 'cat', 'content': 'CATALOG_MUST_NOT_LEAK'},
        {'role': 'assistant', 'content': 'INTERMEDIATE_REFLECTION', 'tool_calls': [call('img', 'imaging')]},
        {'role': 'tool', 'tool_call_id': 'img', 'content': 'FUTURE_IMAGE_FINDING'},
        {'role': 'assistant', 'content': json.dumps({'answer': {'diagnosis': 'Acute appendicitis'},
                                                  'reasoning_summary': 'FINAL_SUMMARY'})},
    ]


def setup_source(tmp_path, monkeypatch, *, h=None, diagnosis='Acute appendicitis'):
    monkeypatch.chdir(tmp_path)
    source = tmp_path/'source'; source.mkdir()
    data = source/'data'; data.mkdir()
    write_dataset(data, [{'id': 'ep', 'turns': [{'id': 't1', 'evidence': [{'id': 'hpi', 'text': 'SYNTHETIC_HPI'}]}]}],
                  targets=[{'id': 'ep', 'turns': {'t1': {'answer': {'diagnosis': 'appendicitis'}}}}])
    (data/'eval.json').write_text(json.dumps({'scorer': 'mimic_cdm_open'}))
    cfg = {'model': 'test', 'base_url': 'https://fiction.invalid/v1', 'api_key_env': 'SYNTHETIC_PROBE_KEY', 'temperature': 0.8}
    (tmp_path/'ama.json').write_text(json.dumps({'models': {'test': cfg}}))
    h = copy.deepcopy(history() if h is None else h)
    manifest = {'mode': 'run', 'output_contract_version': 2, 'termination': 'completed',
                'episode_ids': ['ep'], 'model_config': cfg, 'dataset_dir': str(data), 'expected_turns': {'ep': ['t1']}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    (source/'messages.json').write_text(json.dumps({'ep': h}))
    row = {'episode_id': 'ep', 'turn_id': 't1', 'termination': 'completed',
           'decision': {'turn_id': 't1', 'answer': {'diagnosis': diagnosis}, 'citations': [], 'reasoning_summary': 'FINAL_SUMMARY'}}
    (source/'decisions.jsonl').write_text(json.dumps(row)+'\n')
    prefixes = [{'episode_id': 'ep', 'kind': 'before_model', 'history': h[:2], 'evidence_ids': []},
                {'episode_id': 'ep', 'kind': 'after_tool_batch', 'history': h[:-1], 'evidence_ids': []},
                {'episode_id': 'ep', 'kind': 'before_model', 'history': h[:-1], 'evidence_ids': []}]
    (source/'prefixes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prefixes))
    requests = []
    monkeypatch.setenv('SYNTHETIC_PROBE_KEY', 'test-only')
    init = ProbeTransport.__init__
    def transport(self, settings, on_attempt, **kwargs):
        return init(self, settings, on_attempt, sleep=lambda _: None)
    monkeypatch.setattr(ProbeTransport, '__init__', transport)
    def complete(model, messages, tools=None):
        assert tools is None
        assert model.config['temperature'] == 0
        requests.append(copy.deepcopy(messages))
        model.last_usage = {'total_tokens': 7}
        model.last_response_metadata = metadata()
        return {'role': 'assistant', 'content': 'A'}
    monkeypatch.setattr(Model, 'complete', complete)
    return source, requests


def prepare_cli(tmp_path, source, *extra, mode='cumulative', success=True):
    args = ['probe', str(source), '--mode', mode, '--model', 'test', '--runs-root', str(tmp_path/'probes'), *extra]
    before = set((tmp_path/'probes').glob('*/manifest.json'))
    status = main(args)
    assert status == (0 if success else 1)
    new = set((tmp_path/'probes').glob('*/manifest.json'))-before
    return new.pop().parent if new else None


def test_cumulative_real_cli_leakage_and_frozen_offline_eval(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in source.glob('*') if p.is_file()}
    read = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.name in {'targets.jsonl', 'eval.json', 'metrics.json'}:
            raise AssertionError('probe inference must not read gold or scores')
        return read(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_text', guarded)
        root = prepare_cli(tmp_path, source)
    assert len(requests) == 4
    joined = [json.dumps(r) for r in requests]
    for forbidden in ['CATALOG_MUST_NOT_LEAK', 'ASSISTANT_EXPLANATION', 'PRIVATE_REASONING', 'INTERMEDIATE_REFLECTION', 'FINAL_SUMMARY']:
        assert all(forbidden not in r for r in joined)
    assert ['EXAM_RESULT' in r for r in joined] == [False, True, True, True]
    assert ['LAB_RESULT' in r for r in joined] == [False, False, True, True]
    assert ['FUTURE_IMAGE_FINDING' in r for r in joined] == [False, False, False, True]
    assert all('Acute appendicitis' in r for r in joined)
    assert main(['eval', str(root)]) == 0
    metrics = json.loads((root/'metrics.json').read_text())['scored']
    assert metrics['aggregate']['valid_probes'] == 4
    assert [p['probes']['sufficient']['delta_p'] for p in metrics['per_prefix']] == [None, 0, 0, 0]
    assert metrics['original_clinical_score']['scored']['aggregate']['diagnosis_accuracy_auto']['value'] == 1
    assert len(requests) == 4
    assert main(['probe', '--resume', str(root)]) == 0
    assert len(requests) == 4
    assert all(p.read_bytes() == b for p, b in before.items())


def test_prepare_only_without_credentials_and_endpoint_parity(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    monkeypatch.delenv('SYNTHETIC_PROBE_KEY')
    root = prepare_cli(tmp_path, source, '--prepare-only', '--kind', 'sufficient', '--kind', 'true', mode='endpoint')
    assert not requests
    saved = [json.loads((root/'requests'/(j['request_sha256']+'.json')).read_text()) for j in
             [json.loads(line) for line in (root/'jobs.jsonl').read_text().splitlines()]]
    for kind, request in zip(['sufficient', 'true'], saved):
        assert request == probe_history({'kind': 'post_submission', 'history': history()}, kind, 'Acute appendicitis')
        assert 'CATALOG_MUST_NOT_LEAK' in json.dumps(request)
        assert 'FINAL_SUMMARY' in json.dumps(request)
    assert main(['eval', str(root)]) == 0
    assert not requests
    monkeypatch.setenv('SYNTHETIC_PROBE_KEY', 'test-only')
    assert main(['probe', '--resume', str(root)]) == 0
    assert len(requests) == 2


def test_duplicates_feedback_and_no_tool_cases():
    h = history()
    h[-1:-1] = [
        {'role': 'assistant', 'tool_calls': [call('dup', 'imaging'), call('err', 'microbiology'), call('empty', 'microbiology')]},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': 'FUTURE_IMAGE_FINDING'},
        {'role': 'tool', 'tool_call_id': 'err', 'content': '{"error":"unavailable"}'},
        {'role': 'tool', 'tool_call_id': 'empty', 'content': 'No microbiology results available for this admission.'},
    ]
    snaps, excluded = cumulative_snapshots(h)
    assert len(snaps) == 7 and len(excluded) == 1
    assert snaps[4]['new_evidence']['duplicate']
    assert snaps[5]['new_evidence']['failed']
    assert snaps[6]['new_evidence']['no_result']
    assert len(cumulative_snapshots([*h[:2], h[-1]])[0]) == 1


@pytest.mark.parametrize('problem', ['incomplete', 'unknown_tool', 'image', 'extra_user', 'duplicate_call'])
def test_unsupported_inputs_fail_before_calls(tmp_path, monkeypatch, problem):
    h = history()
    if problem == 'incomplete': h.pop(4)
    if problem == 'unknown_tool': h[2]['tool_calls'][0]['function']['name'] = 'unknown'
    if problem == 'image': h[1]['content'].append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}})
    if problem == 'extra_user': h.insert(-1, {'role': 'user', 'content': 'UNTRACKED_EVIDENCE'})
    if problem == 'duplicate_call': h[2]['tool_calls'][1]['id'] = 'pe'
    source, requests = setup_source(tmp_path, monkeypatch, h=h)
    prepare_cli(tmp_path, source, success=False)
    assert not requests


def test_missing_candidate_and_explicit_override(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch, diagnosis=None)
    root = prepare_cli(tmp_path, source)
    assert not requests
    assert main(['eval', str(root)]) == 0
    metrics = json.loads((root/'metrics.json').read_text())['scored']
    assert metrics['aggregate']['missing_candidates'] == 1
    assert metrics['aggregate']['admissions'] == 1
    override = tmp_path/'candidates.json'; override.write_text(json.dumps({'ep': 'Acute pancreatitis'}))
    root = prepare_cli(tmp_path, source, '--candidates', str(override))
    assert len(requests) == 4
    assert all('Acute pancreatitis' in json.dumps(r) for r in requests)
    record = json.loads((root/'candidates.json').read_text())[0]
    assert record['original'] is None and record['overridden']
    override.write_text(json.dumps({'ep': 'changed'}))
    assert main(['probe', '--resume', str(root)]) == 1
    assert len(requests) == 4


@pytest.mark.parametrize('tamper', ['source', 'snapshot', 'request', 'config', 'job'])
def test_resume_integrity_fails_before_calls(tmp_path, monkeypatch, tamper):
    source, requests = setup_source(tmp_path, monkeypatch)
    root = prepare_cli(tmp_path, source, '--prepare-only')
    if tamper == 'source': target = source/'messages.json'
    elif tamper == 'snapshot': target = root/'snapshots.jsonl'
    elif tamper == 'request': target = next((root/'requests').glob('*.json'))
    elif tamper == 'config': target = root/'plan.json'
    else: target = root/'jobs.jsonl'
    target.write_text('{}\n')
    assert main(['probe', '--resume', str(root)]) == 1
    assert not requests


def test_failed_probability_stops_and_delta_does_not_bridge_gap(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    valid_complete = Model.complete
    def invalid_second(model, messages, tools=None):
        response = valid_complete(model, messages, tools)
        if len(requests) == 2:
            model.last_response_metadata = metadata(top=[{'token': 'A', 'logprob': -0.1}])
        return response
    monkeypatch.setattr(Model, 'complete', invalid_second)
    root = prepare_cli(tmp_path, source, success=False)
    assert len(requests) == 2
    assert main(['probe', '--resume', str(root)]) == 1
    assert len(requests) == 2
    assert main(['eval', str(root)]) == 0
    features = json.loads((root/'metrics.json').read_text())['scored']['per_prefix']
    assert [x['probes']['sufficient']['delta_p'] for x in features] == [None]*4


def test_inflight_not_resubmitted_and_old_cli_removed(tmp_path, monkeypatch, capsys):
    source, requests = setup_source(tmp_path, monkeypatch)
    root = prepare_cli(tmp_path, source, '--prepare-only')
    job = json.loads((root/'jobs.jsonl').read_text().splitlines()[0])
    (root/'diagnostics.jsonl').write_text(json.dumps({'kind': 'probe_start', 'job_id': job['job_id']})+'\n')
    assert main(['probe', '--resume', str(root)]) == 1
    assert not requests
    assert main(['run', 'unused', '--probing-plan', 'old.json']) == 1
    assert 'ama probe' in capsys.readouterr().out


def test_partial_success_resume_and_adjacent_gap(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    root = prepare_cli(tmp_path, source)
    rows = (root/'probe_calls.jsonl').read_text().splitlines()
    # A valid later score must not bridge an unobserved intermediate point.
    (root/'probe_calls.jsonl').write_text('\n'.join([rows[0], rows[2], rows[3]])+'\n')
    assert main(['eval', str(root)]) == 0
    ps = [r['probes']['sufficient'] for r in json.loads((root/'metrics.json').read_text())['scored']['per_prefix']]
    assert ps[2]['valid'] and ps[2]['delta_p'] is None
    # Simulate clean stop between jobs, not an unknown in-flight call.
    (root/'probe_calls.jsonl').write_text(rows[0]+'\n')
    diagnostics = [json.loads(x) for x in (root/'diagnostics.jsonl').read_text().splitlines()]
    (root/'diagnostics.jsonl').write_text(json.dumps(diagnostics[0])+'\n')
    assert main(['probe', '--resume', str(root)]) == 0
    assert len(requests) == 7


def test_archived_task_resumes_through_new_cli(tmp_path, monkeypatch):
    from test_original_agent_probing import fixture, fake_transport
    from ama.probing import execute_probing
    data, plan, cfg, _ = fixture(tmp_path)
    requests = []; fake_transport(monkeypatch, cfg, requests)
    run = execute_probing(data, Model('test', cfg), plan, runs_root=tmp_path/'runs')
    prior = len(requests)
    frozen = plan.read_bytes()
    assert main(['probe', '--resume', str(run)]) == 0
    assert len(requests) == prior and plan.read_bytes() == frozen
    assert main(['eval', str(run)]) == 0


def test_unexpected_tool_reply_is_recorded_before_stopping(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    complete = Model.complete
    def unexpected(model, messages, tools=None):
        complete(model, messages, tools)
        return {'role': 'assistant', 'content': None, 'tool_calls': [call('unexpected', 'imaging')]}
    monkeypatch.setattr(Model, 'complete', unexpected)
    root = prepare_cli(tmp_path, source, success=False)
    records = [json.loads(r) for r in (root/'probe_calls.jsonl').read_text().splitlines()]
    assert len(requests) == len(records) == 1
    assert records[0]['usage']['total_tokens'] == 7
    assert records[0]['response_metadata']['model'] == 'test'
    assert 'CallLimitExceeded' in records[0]['error']
    assert main(['probe', '--resume', str(root)]) == 1
    assert len(requests) == 1


def test_no_scorer_is_not_guessed(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    (source/'data/eval.json').unlink()
    root = prepare_cli(tmp_path, source, '--prepare-only')
    assert main(['eval', str(root)]) == 0
    metrics = json.loads((root/'metrics.json').read_text())['scored']
    assert metrics['original_clinical_score']['scorer'] == 'unscored'
    assert not requests


def test_model_mismatch_and_changed_resume_options_rejected(tmp_path, monkeypatch):
    source, requests = setup_source(tmp_path, monkeypatch)
    root = prepare_cli(tmp_path, source, '--prepare-only')
    assert main(['probe', '--resume', str(root), '--mode', 'endpoint']) == 1
    models = json.loads((tmp_path/'ama.json').read_text())
    models['models']['test']['model'] = 'different-model'
    (tmp_path/'ama.json').write_text(json.dumps(models))
    prepare_cli(tmp_path, source, success=False)
    assert not requests
