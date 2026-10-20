"""Synthetic offline scoring and upstream-parity checks; no clinical data/API."""
import hashlib
import json
from pathlib import Path

import pytest

from ama.cxr_report import SCORER, score_mimic_cxr_report
from ama.data import load_dataset
from ama._vendor.automedbench_report import report_scorer

REPORT = 'The lungs are clear. No pleural effusion or pneumothorax is identified.'
NO_LABEL_REPORT = 'The study shows a stable appearance with otherwise unremarkable features.'


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


def test_vendored_source_hashes_and_modifications_are_recorded():
    root = Path(report_scorer.__file__).parent
    source = json.loads((root / 'SOURCE.json').read_text())
    assert source['revision'] == 'a4e64e5502c728cee744edce2b596000fcd60fdd'
    for name, metadata in source['files'].items():
        expected = metadata.get('vendored_sha256', metadata['sha256'])
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_perfect_and_negation_pairs_match_upstream(tmp_path):
    ds = fixture(tmp_path)
    result = score_mimic_cxr_report(ds, decisions(ds, [REPORT, REPORT]))
    assert result['aggregate']['case_mean_score'] == 1
    assert result['aggregate']['complete_submission_score'] == 1
    negative = report_scorer.score_report_pair(REPORT, 'Pleural effusion is present. The cardiac silhouette is normal in size.')
    result = score_mimic_cxr_report(ds, decisions(ds, [REPORT, 'Pleural effusion is present. The cardiac silhouette is normal in size.']))
    assert result['per_episode']['case-1']['turns'][0]['observation_f1'] == negative['observation_f1']
    assert result['per_episode']['case-1']['turns'][0]['rouge_l_f1'] == negative['report_similarity']
    assert report_scorer.rouge_l_f1('A B C', 'A C') == 0.8


@pytest.mark.parametrize('failure', ['missing', 'parser_error', 'short', 'non_ascii', 'wrong_type'])
@pytest.mark.parametrize('reference', [REPORT, NO_LABEL_REPORT])
def test_invalid_cases_are_zero_and_keep_denominator(tmp_path, failure, reference):
    # Empty rule-label sets must not give an invalid prediction a perfect F1.
    ds = fixture(tmp_path, references=(reference, reference))
    rows = decisions(ds, [reference, reference])
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
