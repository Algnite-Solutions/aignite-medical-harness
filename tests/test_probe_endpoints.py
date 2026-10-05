import hashlib
import json
import pytest
from ama.data import load_dataset
from ama.probing import digest, extract_snapshots, evaluate_probing
from test_data import write_dataset


@pytest.mark.parametrize('initial,final,agreement,initial_ok,final_ok', [
    ('Acute appendicitis','Appendiceal inflammation','consistent',True,True),
    ('Appendicitis','Acute pancreatitis','inconsistent',True,False),
    ('Colitis','Pancreatitis','unresolved',False,False),
])
def test_endpoints_use_frozen_mapper_and_do_not_treat_unmapped_as_agreement(tmp_path,initial,final,agreement,initial_ok,final_ok):
    history=[{'role':'user','content':'fictional HPI'},
             {'role':'assistant','tool_calls':[{'id':'c','function':{'name':'imaging','arguments':'{}'}}]},
             {'role':'tool','tool_call_id':'c','content':'fictional evidence'},
             {'role':'assistant','content':'SOURCE_FINAL_EXCLUDED'}]
    snapshots=extract_snapshots(history)
    for s in snapshots:s.update(episode_id='ep',turn_id='s'+str(s['step']))
    source=tmp_path/'snapshots.jsonl'
    source.write_text(''.join(json.dumps(s)+'\n' for s in snapshots))
    data=tmp_path/'data';data.mkdir()
    write_dataset(data,[{'id':'ep','turns':[{'id':s['turn_id'],'evidence':[{'id':s['turn_id'],'text':'fictional'}]} for s in snapshots]}],
        targets=[{'id':'ep','turns':{s['turn_id']:{'answer':{'diagnosis':'appendicitis'}} for s in snapshots}}])
    plan={'job_scope':'core','snapshots_file':str(source),'snapshots_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan))
    run=tmp_path/'run';run.mkdir()
    rows=[{'episode_id':'ep','turn_id':s['turn_id'],'variant':'real','kind':'diagnosis','reply':answer,
           'error':None,'usage':None,'model_calls':1} for s,answer in zip(snapshots,[initial,final])]
    (run/'probe_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'probing_plan':str(planfile),'plan_sha256':digest(plan),'model_config':{'model':'test'},'model':'test','run_id':'test'}
    assert evaluate_probing(run,load_dataset(data,with_targets=True),manifest,{})==0
    scored=json.loads((run/'metrics.json').read_text())['scored']
    e=scored['endpoints'][0]
    assert e['agreement']==agreement and e['hpi']['correct_auto']==initial_ok and e['final_prefix']['correct_auto']==final_ok
    assert scored['endpoint_summary']['hpi_accuracy']==float(initial_ok)
    assert scored['endpoint_summary']['final_prefix_accuracy']==float(final_ok)
