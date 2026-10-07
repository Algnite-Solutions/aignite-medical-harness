"""Synthetic standard-CLI replay/selection fixtures. No real API requests."""
import copy
import json
from pathlib import Path

import pytest

from ama.cli import main
from ama.data import load_dataset
from ama.original_agent_probing import file_hash, measurement_snapshots
from ama.probing import digest
from ama.scorer import REGISTRY
from test_data import write_dataset
from test_original_agent_probing import fixture, fake_transport


RICH_HISTORY = [
    {'role': 'system', 'content': 'ORIGINAL_AMA_TASK_AND_JSON'},
    {'role': 'user', 'content': 'FICTIONAL_HPI'},
    {'role': 'assistant', 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'imaging', 'arguments': '{}'}},
        {'id': 'c2', 'type': 'function', 'function': {'name': 'laboratory_results', 'arguments': '{}'}}]},
    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Impression: FICTIONAL_IMAGE'},
    {'role': 'tool', 'tool_call_id': 'c2', 'content': '{"error":"fictional unavailable"}'},
    {'role': 'user', 'content': [
        {'type': 'text', 'text': 'FICTIONAL_IMAGE_ATTACHMENT'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]},
    {'role': 'assistant', 'tool_calls': [
        {'id': 'c3', 'type': 'function', 'function': {'name': 'imaging', 'arguments': '{}'}}]},
    {'role': 'tool', 'tool_call_id': 'c3', 'content': 'Impression: FICTIONAL_IMAGE'},
    {'role': 'assistant', 'content': 'PRE_FINAL_REFLECTION_MUST_NOT_LEAK'},
    {'role': 'assistant', 'content': 'FINAL_EXPLANATION_MUST_NOT_LEAK'},
]


def prepare(tmp_path, base, design, kinds=None):
    plan = copy.deepcopy(base)
    histories = {v: json.loads((Path(source['run'])/'messages.json').read_text())['ep']
                 for v, source in plan['original_sources'].items()}
    snaps = measurement_snapshots(histories, design)
    for s in snaps:
        s.update(episode_id='ep', turn_id='s'+str(s['step']))
    snapshot_file = tmp_path/(design+'_snapshots.jsonl')
    snapshot_file.write_text(''.join(json.dumps(s)+'\n' for s in snaps))
    data = tmp_path/(design+'_dataset'); data.mkdir()
    write_dataset(data, [{'id': 'ep', 'turns': [
        {'id': s['turn_id'], 'evidence': [{'id': s['turn_id'], 'text': 'Synthetic frozen history'}]}
        for s in snaps]}])
    plan.update(schema='ama-original-agent-probing', measurement_design=design,
                snapshots_file=str(snapshot_file), snapshots_sha256=file_hash(snapshot_file))
    if kinds is not None:
        plan['probe_kinds'] = kinds
    file = tmp_path/(design+'_plan.json'); file.write_text(json.dumps(plan))
    return data, file, snaps


def replay_fixture(tmp_path, *, history=None, final_missing=False, kinds=None):
    _, file, config, _ = fixture(tmp_path)
    plan = json.loads(file.read_text())
    for view, source in plan['original_sources'].items():
        root = Path(source['run'])
        rows = [json.loads(x) for x in (root/'decisions.jsonl').read_text().splitlines()]
        rows[0]['decision']['answer'] = (None if final_missing else {'diagnosis': 'Acute appendicitis'}) if view == 'interactive' else {'diagnosis': 'Acute cholecystitis'}
        (root/'decisions.jsonl').write_text(json.dumps(rows[0])+'\n')
        if view == 'interactive':
            (root/'messages.json').write_text(json.dumps({'ep': copy.deepcopy(history or RICH_HISTORY)}))
        baseline = REGISTRY['mimic_cdm_open'](load_dataset(root/'dataset', with_targets=True), {'ep': rows})
        (root/'metrics.json').write_text(json.dumps({'scored': baseline}))
        source['hashes'] = {str(root/n): file_hash(root/n) for n in ['manifest.json', 'messages.json', 'decisions.jsonl']}
        source['eval_hashes'] = {str(root/'metrics.json'): file_hash(root/'metrics.json')}
    data, file, snaps = prepare(tmp_path, plan, 'tool_history_replay', kinds)
    return data, file, config, snaps


def cli_run(tmp_path, data, plan, config, *, resume=None, success=True):
    models = tmp_path/'models.json'
    models.write_text(json.dumps({'models': {'synthetic': config}}))
    root = tmp_path/'runs'
    before = set(root.glob('*/manifest.json'))
    if resume:
        assert main(['probe', '--resume', str(resume)]) == (0 if success else 1)
    else:
        # Construct an archived-format run through its retained backend. The old
        # CLI flag is intentionally gone; all resumption uses the new entrypoint.
        from ama.probing import execute_probing
        from ama.model import Model, ModelError
        try:
            execute_probing(data, Model('synthetic', config), plan, runs_root=root)
        except (ValueError, ModelError):
            if success:
                raise
        else:
            assert success
    if not success:
        return None
    if resume:
        return resume
    created = set(root.glob('*/manifest.json'))-before
    assert len(created) == 1
    return created.pop().parent


def evaluate(run):
    assert main(['eval', str(run)]) == 0
    return json.loads((run/'metrics.json').read_text())['scored']


@pytest.mark.parametrize('kinds, expected', [(None, 6), (['sufficient'], 3), (['true'], 3)])
def test_standard_cli_fixed_final_replay_and_offline_eval(tmp_path, monkeypatch, kinds, expected):
    data, plan, config, snaps = replay_fixture(tmp_path, kinds=kinds)
    requests = []; fake_transport(monkeypatch, config, requests)
    original = {v: (tmp_path/v/'decisions.jsonl').read_bytes() for v in ['hpi', 'interactive']}
    read = Path.read_text
    def guard(path, *args, **kwargs):
        if path.name in {'targets.jsonl', 'metrics.json'}:
            raise AssertionError('inference read evaluator-only references')
        return read(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_text', guard)
        run = cli_run(tmp_path, data, plan, config)
    assert len(requests) == expected
    rows = [json.loads(x) for x in (run/'probe_calls.jsonl').read_text().splitlines()]
    for row, request in zip(rows, requests):
        s = next(s for s in snaps if s['turn_id'] == row['turn_id'])
        assert request[:-1] == s['history']
        assert request[0] == RICH_HISTORY[0]
        assert request[-1]['role'] == 'user'
        assert row['candidate'] == 'Acute appendicitis'
        assert row['target'] == 'original_final_hindsight'
        assert 'Acute cholecystitis' not in json.dumps(request)
        assert 'FINAL_EXPLANATION_MUST_NOT_LEAK' not in json.dumps(request)
        assert 'PRE_FINAL_REFLECTION_MUST_NOT_LEAK' not in json.dumps(request)
    assert 'FICTIONAL_IMAGE' not in json.dumps(requests[0])
    assert 'c3' not in json.dumps(next(r for r, row in zip(requests, rows) if row['step'] == 1))
    scored = evaluate(run)
    assert len(requests) == expected  # eval performs no model requests
    assert scored['aggregate']['valid_probes'] == expected
    assert all(c['exact_historical_match'] for c in scored['historical_comparison'].values())
    assert scored['original_ama_eval']['hpi']['aggregate']['diagnosis_accuracy_auto']['value'] == 0
    assert scored['original_ama_eval']['interactive']['aggregate']['diagnosis_accuracy_auto']['value'] == 1
    assert [f['source_message_end'] for f in scored['per_prefix']] == [2, 6, 8]
    assert [f['batch_size'] for f in scored['per_prefix']] == [0, 2, 1]
    assert scored['per_prefix'][1]['tool_failures'] == 1
    assert scored['per_prefix'][2]['duplicate_results'] == 1
    assert scored['per_prefix'][1]['diagnostic_impression']
    assert scored['per_prefix'][1]['source_view'] == 'interactive'
    for feature in scored['per_prefix']:
        assert set(p.split('/')[0] for p in feature['probes']) == set(kinds or ['sufficient', 'true'])
        assert all(p['p_ab'] == pytest.approx(7/9) for p in feature['probes'].values())
    for v in original:
        assert (run/'original_submissions'/v/'decisions.jsonl').read_bytes() == original[v]
        assert (tmp_path/v/'decisions.jsonl').read_bytes() == original[v]
    assert (run/'decisions.jsonl').read_text() == ''
    assert json.loads((run/'manifest.json').read_text())['output_contract_version'] == 'original-agent-probing'


@pytest.mark.parametrize('kind', ['single', 'none', 'missing', 'missing_without_submission'])
def test_replay_boundaries_no_tools_and_missing_final(tmp_path, monkeypatch, kind):
    if kind == 'single':
        history = RICH_HISTORY[:3] + [RICH_HISTORY[3], RICH_HISTORY[-1]]
        history = copy.deepcopy(history); history[2]['tool_calls'] = history[2]['tool_calls'][:1]
    elif kind == 'none':
        history = RICH_HISTORY[:2]+RICH_HISTORY[-2:]
    else:
        history = RICH_HISTORY[:8] if kind == 'missing_without_submission' else RICH_HISTORY
    missing = kind.startswith('missing')
    data, plan, config, snaps = replay_fixture(tmp_path, history=history, final_missing=missing, kinds=['sufficient'])
    requests = []; fake_transport(monkeypatch, config, requests)
    run = cli_run(tmp_path, data, plan, config)
    score = evaluate(run)
    expected = 0 if missing else 2 if kind == 'single' else 1
    assert len(requests) == score['aggregate']['valid_probes'] == expected
    assert score['aggregate']['missing_original_candidates'] == (3 if missing else 0)
    assert score['original_ama_eval']['interactive']['aggregate']['diagnosis_accuracy_auto']['den'] == 1
    assert score['endpoints'][0]['agreement'] == ('unresolved' if missing else 'inconsistent')
    if kind == 'none':
        assert len(snaps) == 1 and snaps[0]['kind'] == 'initial' and snaps[0]['step'] == 0


@pytest.mark.parametrize('kinds', [[], None, 'sufficient', 1, ['unknown'], ['sufficient', 'sufficient'], [{}], [True]])
def test_invalid_selection_stops_before_request(tmp_path, monkeypatch, kinds):
    data, plan, config, _ = replay_fixture(tmp_path)
    p = json.loads(plan.read_text()); p['probe_kinds'] = kinds; plan.write_text(json.dumps(p))
    requests = []; fake_transport(monkeypatch, config, requests)
    cli_run(tmp_path, data, plan, config, success=False)
    assert not requests and not list((tmp_path/'runs').glob('*/manifest.json'))


@pytest.mark.parametrize('tamper', ['metadata', 'future'])
def test_replay_tampering_rejected_before_request(tmp_path, monkeypatch, tamper):
    data, plan, config, snaps = replay_fixture(tmp_path)
    if tamper == 'metadata':
        snaps[1]['batch_size'] += 1
    else:
        snaps[0]['history'] = copy.deepcopy(RICH_HISTORY)
        snaps[0]['prefix_sha256'] = digest(snaps[0]['history'])
    p = json.loads(plan.read_text()); file = Path(p['snapshots_file'])
    file.write_text(''.join(json.dumps(s)+'\n' for s in snaps)); p['snapshots_sha256'] = file_hash(file); plan.write_text(json.dumps(p))
    requests = []; fake_transport(monkeypatch, config, requests)
    cli_run(tmp_path, data, plan, config, success=False)
    assert not requests


@pytest.mark.parametrize('broken', ['missing_result', 'duplicate_result'])
def test_pending_or_duplicate_tool_results_are_not_split_into_prefixes(broken):
    history = copy.deepcopy(RICH_HISTORY)
    if broken == 'missing_result':
        del history[4]
    else:
        history.insert(5, copy.deepcopy(history[3]))
    with pytest.raises(ValueError, match='tool'):
        measurement_snapshots({'interactive': history}, 'tool_history_replay')


def test_old_contract_read_selection_reuse_and_resume_identity(tmp_path, monkeypatch):
    data, plan, config, _ = replay_fixture(tmp_path)
    p = json.loads(plan.read_text()); p['schema'] = 'ama-original-agent-probing-v1'; plan.write_text(json.dumps(p))
    requests = []; fake_transport(monkeypatch, config, requests)
    dual = cli_run(tmp_path, data, plan, config)
    assert len(requests) == 6
    manifest = json.loads((dual/'manifest.json').read_text())
    manifest['output_contract_version'] = 'original-agent-probing-v1'
    (dual/'manifest.json').write_text(json.dumps(manifest))
    assert evaluate(dual)['aggregate']['valid_probes'] == 6
    p.update(schema='ama-original-agent-probing', probe_kinds=['sufficient'], reuse_runs=[str(dual)])
    only = tmp_path/'only.json'; only.write_text(json.dumps(p))
    reused = cli_run(tmp_path, data, only, config)
    assert len(requests) == 6
    assert evaluate(reused)['aggregate']['reused_jobs'] == 3
    assert cli_run(tmp_path, data, only, config, resume=reused) == reused
    assert len(requests) == 6
    p['probe_kinds'] = ['true']; changed = tmp_path/'changed.json'; changed.write_text(json.dumps(p))
    frozen_only = only.read_bytes()
    only.write_bytes(changed.read_bytes())
    cli_run(tmp_path, data, only, config, resume=reused, success=False)
    only.write_bytes(frozen_only)
    assert len(requests) == 6
    p.update(probe_kinds=['sufficient', 'true'], reuse_runs=[str(reused)])
    expanded = tmp_path/'expanded.json'; expanded.write_text(json.dumps(p))
    expanded_run = cli_run(tmp_path, data, expanded, config)
    assert len(requests) == 9  # only the three previously unselected True probes are new
    assert evaluate(expanded_run)['aggregate']['reused_jobs'] == 3


def test_completed_endpoints_and_replay_do_not_cross_reuse(tmp_path, monkeypatch):
    data, plan, config, _ = replay_fixture(tmp_path, kinds=['sufficient'])
    requests = []; fake_transport(monkeypatch, config, requests)
    replay = cli_run(tmp_path, data, plan, config)
    assert len(requests) == 3
    base = json.loads(plan.read_text())
    post_data, post_plan, _ = prepare(tmp_path, base, 'post_submission', ['sufficient'])
    p = json.loads(post_plan.read_text()); p['reuse_runs'] = [str(replay)]; post_plan.write_text(json.dumps(p))
    post = cli_run(tmp_path, post_data, post_plan, config)
    assert len(requests) == 5
    score = evaluate(post)
    assert score['aggregate']['reused_jobs'] == 0
    assert 'FINAL_EXPLANATION_MUST_NOT_LEAK' in json.dumps(requests[-1])


@pytest.mark.parametrize('design, kinds, expected', [
    ('post_submission', ['sufficient'], 2), ('post_submission', ['true'], 2),
    ('evidence_prefix', ['sufficient'], 8), ('evidence_prefix', ['true'], 4)])
def test_selection_on_completed_and_legacy_prefix_designs(tmp_path, monkeypatch, design, kinds, expected):
    _, base_plan, config, _ = replay_fixture(tmp_path)
    data, plan, _ = prepare(tmp_path, json.loads(base_plan.read_text()), design, kinds)
    requests = []; fake_transport(monkeypatch, config, requests)
    run = cli_run(tmp_path, data, plan, config)
    assert len(requests) == expected
    assert evaluate(run)['aggregate']['valid_probes'] == expected


@pytest.mark.parametrize('probe_kind', ['sufficient', 'true'])
def test_selected_endpoint_probe_missing_hpi_keeps_denominator(tmp_path, monkeypatch, probe_kind):
    from test_original_agent_probing import post_submission_fixture
    data, plan, config, _ = post_submission_fixture(tmp_path, hpi_abstains=True)
    p = json.loads(plan.read_text()); p.update(schema='ama-original-agent-probing', probe_kinds=[probe_kind])
    plan.write_text(json.dumps(p))
    requests = []; fake_transport(monkeypatch, config, requests)
    run = cli_run(tmp_path, data, plan, config)
    score = evaluate(run)
    assert len(requests) == score['aggregate']['valid_probes'] == 1
    assert score['aggregate']['missing_original_candidates'] == 1
    assert score['original_ama_eval']['hpi']['aggregate']['diagnosis_accuracy_auto'] == {'num': 0, 'den': 1, 'value': 0.0}
    assert score['endpoints'][0]['agreement'] == 'unresolved'
