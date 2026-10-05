import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from ama.model import Model, ModelError
from ama.probing import digest, execute_probing, extract_snapshots
from ama.probing_transport import ProbeTransport, retry_after_seconds
from ama.cli import _eval_run
from test_data import write_dataset
from test_probing import metadata


def transport(monkeypatch, replies, settings=None):
    replies = iter(replies)
    waits, records = [], []
    model = SimpleNamespace(last_usage=None, transport_attempts=[])
    def complete(*args):
        value = next(replies)
        if isinstance(value, BaseException): raise value
        model.last_usage = {'total_tokens': 12}
        return value
    monkeypatch.setattr(Model, 'complete', complete)
    stream = ProbeTransport(settings or {}, records.append, sleep=waits.append, clock=lambda: 1000)
    return stream, model, waits, records


def test_429_recovery_records_each_attempt_and_adapts_pacing(monkeypatch):
    limited = ModelError('HTTP 429', http_status=429, rate_limit_headers={'retry-after': '42'})
    stream, model, waits, records = transport(monkeypatch, [limited, {'content': 'A'}])
    assert stream.complete(model, [], None, 'j') == {'content': 'A'}
    assert waits == [3, 42, 6]
    assert [r['status'] for r in records] == ['started', 'rejected', 'started', 'completed']
    assert len(model.transport_attempts) == 2 and not stream.stop_reason


def test_exhausted_429_stops_before_next_job(monkeypatch):
    limited = ModelError('HTTP 429', http_status=429)
    stream, model, waits, records = transport(monkeypatch, [limited, limited], {'max_429_attempts': 2})
    with pytest.raises(ModelError, match='429'): stream.complete(model, [], None, 'j')
    with pytest.raises(ModelError, match='stopped before next job'): stream.complete(model, [], None, 'next')
    assert len(records) == 4 and waits == [3, 30, 6]


@pytest.mark.parametrize('error', [ModelError('timeout'), ModelError('HTTP 503', http_status=503)])
def test_ambiguous_failures_are_not_retried(monkeypatch, error):
    stream, model, waits, records = transport(monkeypatch, [error])
    with pytest.raises(ModelError): stream.complete(model, [], None, 'j')
    assert len(records) == 2 and waits == [3] and stream.stop_reason


def test_retry_after_date_and_overlong_wait(monkeypatch):
    assert retry_after_seconds({'retry-after':'Thu, 01 Jan 1970 00:20:00 GMT'}, now=1000) == 200
    limited = ModelError('HTTP 429', http_status=429, rate_limit_headers={'retry-after':'1000'})
    stream, model, waits, records = transport(monkeypatch, [limited])
    with pytest.raises(ModelError): stream.complete(model, [], None, 'j')
    assert waits == [3] and stream.stop_reason and len(records) == 2


def test_native_core_then_consistency_reuses_jobs_and_preserves_sc20(tmp_path, monkeypatch):
    import hashlib
    import ama.probing as probing
    history = [{'role':'user','content':'FICTIONAL_INITIAL'},
        {'role':'assistant','tool_calls':[{'id':'c','function':{'name':'imaging','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'c','content':'FICTIONAL_NEW_EVIDENCE'},
        {'role':'assistant','content':'FUTURE_FINAL_MUST_NOT_LEAK'}]
    snapshots = extract_snapshots(history)
    for s in snapshots: s.update(episode_id='ep', turn_id='s'+str(s['step']))
    file = tmp_path/'snapshots.jsonl'
    file.write_text(''.join(json.dumps(s)+'\n' for s in snapshots))
    data = tmp_path/'dataset'; data.mkdir()
    write_dataset(data, [{'id':'ep','turns':[{'id':s['turn_id'],'evidence':[{'id':s['turn_id'],'text':'fiction'}]} for s in snapshots]}])
    config = {'model':'deterministic-test-only'}
    requests = []
    class Sidecar:
        name='test-model'; timeout=60
        def __init__(self, name, cfg, timeout):
            self.config=cfg; self.last_usage=None; self.last_response_metadata=None
        def complete(self, messages, tools=None):
            requests.append(messages)
            phrase='Fictional diagnosis' if 'Return one concise English diagnosis phrase' in messages[-1]['content'] else 'A'
            self.last_usage={'total_tokens':10}
            self.last_response_metadata=metadata()
            self.last_response_metadata['model']=config['model']
            if self.config.get('n'):
                self.last_response_metadata['choices']=[{'index':i,'message':{'content':phrase}} for i in range(self.config['n'])]
            return {'role':'assistant','content':phrase}
    monkeypatch.setattr(probing, 'Model', Sidecar)
    model=Sidecar('test-model', config, 60)
    plan={'schema':'ama-interactive-probing-v1','sc_samples':20,'sample_batch_size':10,
          'workers':1,'job_scope':'core','top_logprobs':20,'seed':1,
          'snapshots_file':str(file),'snapshots_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    planfile=tmp_path/'core.json'; planfile.write_text(json.dumps(plan))
    core=execute_probing(data, model, planfile, runs_root=tmp_path/'runs')
    assert len(requests)==8 and 'FUTURE_FINAL_MUST_NOT_LEAK' not in json.dumps(requests)
    assert _eval_run(core)==0
    metrics=json.loads((core/'metrics.json').read_text())['scored']
    assert len(metrics['per_prefix'])==2 and metrics['aggregate']['valid_real_current_sufficient']==2
    plan.update(job_scope='consistency', reuse_runs=[str(core)])
    planfile=tmp_path/'sc.json'; planfile.write_text(json.dumps(plan))
    sc=execute_probing(data, model, planfile, runs_root=tmp_path/'runs')
    assert len(requests)==12  # Eight reused core jobs; only four n=10 sampling calls.
    assert _eval_run(sc)==0
    metrics=json.loads((sc/'metrics.json').read_text())['scored']
    assert all(f['samples_recorded']==20 and f['sc_exact_match']==1 for f in metrics['per_prefix'])
    assert metrics['aggregate']['new_http_attempts']==4 and metrics['aggregate']['reused_jobs']==8
    assert metrics['aggregate']['total_tokens']==40
    before=len(requests)
    execute_probing(data, model, planfile, runs_root=tmp_path/'runs', resume=sc)
    assert len(requests)==before
    # Nested stages reference the same jobs repeatedly; stochastic outputs must
    # remain exactly 20, not be duplicated into 40 or 60 on import.
    plan.update(reuse_runs=[str(core),str(sc),str(core)])
    planfile=tmp_path/'nested.json';planfile.write_text(json.dumps(plan))
    nested=execute_probing(data,model,planfile,runs_root=tmp_path/'runs')
    assert len(requests)==before and _eval_run(nested)==0
    aggregate=json.loads((nested/'metrics.json').read_text())['scored']['aggregate']
    assert aggregate['new_http_attempts']==0 and aggregate['reused_jobs']==12
    assert aggregate['sc_complete_contexts']==2
    # A format instruction is still a changed model input: never reuse its old
    # probabilities. Identical diagnosis/P(True) jobs may still be reused.
    monkeypatch.setitem(probing.TEMPLATES,'sufficient',probing.SUFFICIENT+'\nPlain A/B only.')
    plan.update(job_scope='core',reuse_runs=[str(core)])
    planfile=tmp_path/'changed_prompt.json';planfile.write_text(json.dumps(plan))
    changed=execute_probing(data,model,planfile,runs_root=tmp_path/'runs')
    assert len(requests)==before+4 and _eval_run(changed)==0
    aggregate=json.loads((changed/'metrics.json').read_text())['scored']['aggregate']
    assert aggregate['new_http_attempts']==4 and aggregate['reused_jobs']==4
