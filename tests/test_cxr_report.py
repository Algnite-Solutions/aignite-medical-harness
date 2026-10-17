"""Synthetic offline scoring and upstream-parity checks; no clinical data/API."""
import hashlib
import json
from pathlib import Path

import pytest

from ama.cli import main
from ama.cxr_report import METRICS, SCORER, score_mimic_cxr_report, score_mimic_cxr_report_mlrg
from ama.data import load_dataset
from ama._vendor.automedbench_report import mlrg_metrics, report_scorer

REPORT = 'The lungs are clear. No pleural effusion or pneumothorax is identified.'


def fixture(tmp_path, references=(REPORT, REPORT)):
    root = tmp_path / 'data'
    root.mkdir()
    episodes = [{'id': f'case-{i}', 'turns': [{'id': 't1', 'evidence': [
        {'id': 'image', 'text': 'Synthetic image placeholder.'}]}]} for i in range(len(references))]
    (root / 'dataset.json').write_text(json.dumps({'schema': 'ama-dataset', 'name': 'synthetic',
                                                  'splits': {'test': [e['id'] for e in episodes]}}))
    (root / 'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    (root / 'targets.jsonl').write_text(''.join(json.dumps({'id': e['id'], 'turns': {
        't1': {'answer': {'findings': ref}}}})+'\n' for e, ref in zip(episodes, references)))
    (root / 'eval.json').write_text(json.dumps({'scorer': SCORER}))
    return load_dataset(root, with_targets=True)


def decisions(ds, texts):
    return {e.id: [{'episode_id': e.id, 'turn_id': 't1', 'termination': 'completed',
                   'decision': {'answer': {'findings': text}}}] for e, text in zip(ds.episodes, texts)}


def test_vendored_source_is_pinned_and_unmodified():
    root = Path(report_scorer.__file__).parent
    source = json.loads((root / 'SOURCE.json').read_text())
    assert source['revision'] == 'a4e64e5502c728cee744edce2b596000fcd60fdd'
    for name, metadata in source['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == metadata['sha256']


def test_perfect_and_negation_pairs_match_upstream(tmp_path):
    ds = fixture(tmp_path)
    result = score_mimic_cxr_report(ds, decisions(ds, [REPORT, REPORT]))
    assert result['aggregate']['case_mean_score'] == 1
    assert result['aggregate']['complete_submission_score'] == 1
    assert result['aggregate']['mlrg_metrics'] == dict.fromkeys(METRICS)
    negative = report_scorer.score_report_pair(REPORT, 'Pleural effusion is present. The cardiac silhouette is normal in size.')
    result = score_mimic_cxr_report(ds, decisions(ds, [REPORT, 'Pleural effusion is present. The cardiac silhouette is normal in size.']))
    assert result['per_episode']['case-1']['turns'][0]['observation_f1'] == negative['observation_f1']
    assert result['per_episode']['case-1']['turns'][0]['rouge_l_f1'] == negative['report_similarity']
    assert report_scorer.rouge_l_f1('A B C', 'A C') == 0.8


@pytest.mark.parametrize('failure', ['missing', 'parser_error', 'short', 'non_ascii', 'wrong_type'])
def test_invalid_cases_are_zero_and_keep_denominator(tmp_path, failure):
    ds = fixture(tmp_path)
    rows = decisions(ds, [REPORT, REPORT])
    row = rows['case-1'][0]
    if failure == 'missing':
        rows.pop('case-1')
    elif failure == 'parser_error':
        row.update(termination='invalid_decision', decision=None)
    else:
        row['decision']['answer']['findings'] = {
            'short': 'Clear.', 'non_ascii': REPORT + ' — normal.', 'wrong_type': 23}[failure]
    result = score_mimic_cxr_report(ds, rows)
    agg = result['aggregate']
    assert agg['cases_total'] == 2 and agg['valid_reports'] == 1
    assert agg['case_mean_score'] == 0.5 and agg['valid_case_mean_score'] == 1
    assert agg['complete_submission_score'] == 0
    assert result['per_episode']['case-1']['turns'][0]['observation_f1'] == 0


def test_missing_reference_is_not_silently_dropped(tmp_path):
    ds = fixture(tmp_path)
    ds.targets['case-1'].turns['t1']['answer']['findings'] = ''
    with pytest.raises(ValueError, match='missing reference'):
        score_mimic_cxr_report(ds, decisions(ds, [REPORT, REPORT]))


def test_mlrg_missing_resources_does_not_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv('AMA_CXR_MLRG_CONFIG', raising=False)
    ds = fixture(tmp_path)
    with pytest.raises(ValueError, match='no lightweight fallback'):
        score_mimic_cxr_report_mlrg(ds, decisions(ds, [REPORT, REPORT]))


def test_mlrg_corpus_metric_mapping_and_gating(tmp_path, monkeypatch):
    ds = fixture(tmp_path)
    ds.eval_config.rules = {'mlrg': {k: str(tmp_path) for k in
        ['mlrg_repo_path', 'chexbert_path', 'bert_path', 'radgraph_path']}}
    monkeypatch.delenv('AMA_CXR_MLRG_CONFIG', raising=False)
    seen = []

    def backend(refs, preds, config):
        seen.append((refs, preds))
        return {**{k: 0.5 for k in METRICS},
                **{f'BLEU_{i}': 0.5 for i in range(1, 5)},
                'chexbert_all_micro_p': 0.5, 'chexbert_all_micro_r': 0.5,
                'chexbert_all_micro_f1': 0.5, 'F1-Radgraph-partial': 0.5}

    monkeypatch.setattr(report_scorer, 'compute_mlrg_scores', backend)
    result = score_mimic_cxr_report_mlrg(ds, decisions(ds, [REPORT]))
    assert seen == [([REPORT, REPORT], [REPORT, ''])]
    assert result['aggregate']['mlrg_raw_corpus_score'] == 0.5
    assert result['aggregate']['complete_submission_score'] == 0
    assert result['aggregate']['mlrg_metrics']['F1RadGraph'] == 0.5
    assert 'case_mean_score' not in result['aggregate']


def test_mlrg_rejects_missing_or_nonfinite_metrics(tmp_path, monkeypatch):
    ds = fixture(tmp_path)
    ds.eval_config.rules = {'mlrg': {k: str(tmp_path) for k in
        ['mlrg_repo_path', 'chexbert_path', 'bert_path', 'radgraph_path']}}
    monkeypatch.delenv('AMA_CXR_MLRG_CONFIG', raising=False)
    monkeypatch.setattr(report_scorer, 'compute_mlrg_scores', lambda *args: {'BLEU': float('nan')})
    with pytest.raises(ValueError, match='incomplete/non-finite'):
        score_mimic_cxr_report_mlrg(ds, decisions(ds, [REPORT, REPORT]))
