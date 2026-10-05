import copy
import json
import math
from pathlib import Path
import pytest
from ama.probing import binary_probability, extract_snapshots, context_fragments, render_context
from ama.runner import execute
from ama.cli import _eval_run
from ama.agent import Agent
from fakes import FakeModel, calls, call
from test_data import write_dataset


def metadata(text='A',token='A',top=None):
    a={'token':'A','logprob':math.log(.7)};b={'token':'B','logprob':math.log(.2)}
    return {'model':'test','logprob_content':text,'logprobs':{'content':[{
        'token':token,'bytes':list(token.encode()),'logprob':math.log(.7),'top_logprobs':top if top is not None else [a,b]}]}}


def test_binary_requires_both_labels_at_correct_byte_position():
    assert binary_probability(metadata())['p_ab']==pytest.approx(7/9)
    assert not binary_probability(metadata(top=[{'token':'A','logprob':math.log(.7)}]))['valid']
    assert not binary_probability(metadata(token='\n'))['valid']
    assert not binary_probability(metadata(text='{"label":"A"}',token='{'))['valid']


def test_atomic_prefixes_exclude_future_final_and_preserve_original():
    history=[{'role':'system','content':'task'}, {'role':'user','content':'HPI ONLY'},
             calls(call(name='laboratory_results'),call(name='imaging',id='c2')),
             {'role':'tool','tool_call_id':'call-1','content':'LAB_NEW'},
             {'role':'tool','tool_call_id':'c2','content':'IMPRESSION_NEW'},
             {'role':'assistant','content':'FINAL_DIAGNOSIS_NOT_VISIBLE'}]
    original=copy.deepcopy(history);snaps=extract_snapshots(history)
    assert history==original and len(snaps)==2
    assert snaps[1]['batch_size']==2 and snaps[1]['final_before_submission']
    assert 'LAB_NEW' not in render_context(snaps[0],'real')
    assert 'FINAL_DIAGNOSIS_NOT_VISIBLE' not in json.dumps(snaps)
    assert 'LAB_NEW' in render_context(snaps[1],'real')
    assert 'LAB_NEW' not in render_context(snaps[1],'placebo')
    assert 'LAB_NEW' not in render_context(snaps[1],'repeat')
    summary=render_context(snaps[1],'summary')
    assert all(text in summary for _,text in context_fragments(snaps[1]['history']))


def test_prefix_callback_is_an_isolated_copy():
    captured=[]
    def callback(value):
        captured.append(copy.deepcopy(value));value['history'].clear()
    agent=Agent(FakeModel('done'),on_prefix=callback)
    assert agent.chat('visible')=='done'
    assert captured[0]['history'][0]['content']=='visible'
    assert len(agent.history)==2


def test_cli_binary_run_does_not_inject_JSON_and_eval_is_offline(tmp_path):
    root=tmp_path/'data';root.mkdir()
    write_dataset(root,[{'id':'ep','turns':[{'id':'t1','evidence':[{'id':'e','text':'Output A or B'}]}]}])
    (root/'instructions.txt').write_text('You are helpful.')
    model=FakeModel('A');model.last_response_metadata=metadata()
    run=execute(root,model,runs_root=tmp_path/'runs',output_format='binary-label')
    assert 'JSON' not in json.dumps(model.requests)
    assert _eval_run(run)==0 and len(model.requests)==1
    score=json.loads((run/'metrics.json').read_text())['scored']['binary_probe']
    assert score['valid']==1
