import copy
import json
from pathlib import Path
import pytest

from ama.probing import digest, execute_probing, extract_snapshots
from ama.original_agent_probing import probe_history, file_hash, MODE_EXCEPTION
from ama.model import Model
from ama.cli import _eval_run
from ama.data import load_dataset
from ama.scorer import REGISTRY
from test_data import write_dataset
from test_probing import metadata


def fixture(tmp_path, *, hpi_abstains=False):
    config = {'model': 'test', 'base_url': 'https://fiction.invalid/v1', 'api_key_env': 'FICTIONAL_TEST_KEY', 'temperature': 0}
    user = {'role': 'user', 'content': 'FICTIONAL_HPI'}
    history = [{'role': 'system', 'content': 'ORIGINAL_AMA_TASK_AND_JSON'}, user,
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function',
            'function': {'name': 'imaging', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'FICTIONAL_NEW_EVIDENCE'}]
    sources = {}
    for view in ['hpi', 'interactive']:
        root = tmp_path/view; root.mkdir()
        data = root/'dataset'; data.mkdir()
        write_dataset(data, [{'id': 'ep', 'turns': [{'id': 't1', 'evidence': [{'id': 'hpi', 'text': 'fiction'}]}]}],
            targets=[{'id': 'ep', 'turns': {'t1': {'answer': {'diagnosis': 'appendicitis'}}}}])
        row = {'episode_id': 'ep', 'turn_id': 't1', 'termination': 'completed', 'model_calls': 1,
            'usage': None, 'duration_ms': 1, 'decision': {'turn_id': 't1',
            'answer': None if view == 'hpi' and hpi_abstains else {'diagnosis': 'Acute appendicitis'},
            'citations': [], 'reasoning_summary': 'Original explanation'}}
        (root/'decisions.jsonl').write_text(json.dumps(row)+'\n')
        (root/'manifest.json').write_text(json.dumps({'mode': 'run', 'output_contract_version': 2,
            'dataset_dir': str(data), 'episode_ids': ['ep'], 'model_config': config}))
        h = copy.deepcopy(history if view == 'interactive' else history[:2])
        h.append({'role': 'assistant', 'content': 'ORIGINAL_FINAL_JSON_NOT_IN_PROBE_EVIDENCE'})
        (root/'messages.json').write_text(json.dumps({'ep': h}))
        baseline = REGISTRY['mimic_cdm_open'](load_dataset(data, with_targets=True), {'ep': [row]})
        (root/'metrics.json').write_text(json.dumps({'scored': baseline}))
        sources[view] = {'run': str(root), 'hashes': {str(root/name): file_hash(root/name)
            for name in ['manifest.json', 'messages.json', 'decisions.jsonl']},
            'eval_hashes': {str(root/'metrics.json'): file_hash(root/'metrics.json')}}
    snaps = extract_snapshots(history+[{'role': 'assistant', 'content': 'FINAL'}])
    for s in snaps: s.update(episode_id='ep', turn_id='s'+str(s['step']))
    path = tmp_path/'snapshots.jsonl'; path.write_text(''.join(json.dumps(s)+'\n' for s in snaps))
    data = tmp_path/'probe_dataset'; data.mkdir()
    write_dataset(data, [{'id': 'ep', 'turns': [{'id': s['turn_id'], 'evidence': [{'id': s['turn_id'], 'text': 'frozen prefix'}]} for s in snaps]}])
    plan = {'schema': 'ama-original-agent-probing-v1', 'workers': 1, 'top_logprobs': 20,
            'original_sources': sources, 'snapshots_file': str(path), 'snapshots_sha256': file_hash(path),
            'transport': {'lease_path': str(tmp_path/'lease'), 'request_delay_seconds': 0}}
    planfile = tmp_path/'plan.json'; planfile.write_text(json.dumps(plan))
    return data, planfile, config, snaps


def fake_transport(monkeypatch, config, requests, *, text='A'):
    monkeypatch.setenv('FICTIONAL_TEST_KEY', 'offline-test-only')
    def complete(model, history, tools=None):
        requests.append(copy.deepcopy(history))
        assert tools is None
        model.last_usage = {'total_tokens': 10}
        model.last_response_metadata = metadata(text=text)
        model.last_response_metadata['model'] = config['model']
        return {'role': 'assistant', 'content': text}
    monkeypatch.setattr(Model, 'complete', complete)


def test_native_history_preserved_and_no_mutation():
    source = {'history': [{'role': 'system', 'content': 'Original'}, {'role': 'user', 'content': 'HPI'},
        {'role': 'assistant', 'tool_calls': [{'id': 'c'}]}, {'role': 'tool', 'tool_call_id': 'c', 'content': 'Lab'}]}
    before = copy.deepcopy(source)
    request = probe_history(source, 'sufficient', 'Fixed diagnosis')
    assert source == before
    assert request[1:-1] == source['history'][1:]
    assert request[0]['content'] == 'Original'+MODE_EXCEPTION
    assert 'Fixed diagnosis' in request[-1]['content']


def test_native_cli_eval_original_accuracy_and_exact_reuse(tmp_path, monkeypatch):
    data, planfile, config, snaps = fixture(tmp_path)
    requests = []; fake_transport(monkeypatch, config, requests)
    model = Model('test', config)
    # Evaluator-only gold/metrics must not be opened as text during inference.
    read = Path.read_text
    def guard(path, *args, **kwargs):
        if path.name in {'targets.jsonl', 'metrics.json'}: raise AssertionError('inference read gold/metrics')
        return read(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_text', guard)
        run = execute_probing(data, model, planfile, runs_root=tmp_path/'runs')
    assert len(requests) == 6
    assert 'ORIGINAL_FINAL_JSON_NOT_IN_PROBE_EVIDENCE' not in json.dumps(requests)
    assert all('Return one concise English diagnosis phrase' not in json.dumps(r) for r in requests)
    assert _eval_run(run) == 0
    assert len(requests) == 6  # eval made no API call
    score = json.loads((run/'metrics.json').read_text())['scored']
    assert score['aggregate']['valid_probes'] == 6
    assert all(v['exact_historical_match'] for v in score['historical_comparison'].values())
    assert score['original_ama_eval']['hpi']['aggregate']['diagnosis_accuracy_auto']['value'] == 1
    plan = json.loads(planfile.read_text()); plan['reuse_runs'] = [str(run)]; planfile.write_text(json.dumps(plan))
    reused = execute_probing(data, model, planfile, runs_root=tmp_path/'runs')
    assert len(requests) == 6 and _eval_run(reused) == 0
    assert json.loads((reused/'metrics.json').read_text())['scored']['aggregate']['reused_jobs'] == 6


def test_abstention_not_replaced_and_score_denominator_preserved(tmp_path, monkeypatch):
    data, planfile, config, _ = fixture(tmp_path, hpi_abstains=True)
    requests = []; fake_transport(monkeypatch, config, requests)
    run = execute_probing(data, Model('test', config), planfile, runs_root=tmp_path/'runs')
    assert len(requests) == 4 and _eval_run(run) == 0
    score = json.loads((run/'metrics.json').read_text())['scored']
    assert score['aggregate']['missing_original_candidates'] == 2
    assert score['original_ama_eval']['hpi']['aggregate']['diagnosis_accuracy_auto']['value'] == 0
    assert score['endpoints'][0]['agreement'] == 'unresolved'


def test_invalid_label_stops_without_resampling(tmp_path, monkeypatch):
    data, planfile, config, _ = fixture(tmp_path)
    requests = []; fake_transport(monkeypatch, config, requests, text='**A**')
    with pytest.raises(Exception, match='binary coverage failed'):
        execute_probing(data, Model('test', config), planfile, runs_root=tmp_path/'runs')
    assert len(requests) == 1


def test_changed_source_rejected_before_request(tmp_path, monkeypatch):
    data, planfile, config, _ = fixture(tmp_path)
    source = tmp_path/'interactive/messages.json'; source.write_text('{}')
    requests = []; fake_transport(monkeypatch, config, requests)
    with pytest.raises(ValueError, match='source changed'):
        execute_probing(data, Model('test', config), planfile, runs_root=tmp_path/'runs')
    assert not requests
