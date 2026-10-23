import pytest
from scripts.audit_regression_data import audit


def config(entries):
    return {'schema': 'ama-regression-site-draft-v1', 'benchmarks': entries}


def test_disabled_does_not_require_data():
    result = audit(config({'rocov2': {'enabled': False, 'sample': {'dataset': '/missing'}}}))
    assert result['benchmarks'] == [{'benchmark': 'rocov2', 'status': 'disabled'}]
    assert result['acceptance'] == 'not_run'


def test_missing_scope_blocks_instead_of_using_other_suite():
    result = audit({**config({'pathvqa': {'enabled': True, 'sample': {'dataset': '/missing'}}}),
                    'suite': 'full'})
    assert result['benchmarks'][0]['category'] == 'scope_not_configured'


@pytest.mark.parametrize('entries', [ {'typo': {'enabled': True}},
    {'pathvqa': {'enabled': 'false'}}, {}])
def test_invalid_selection(entries):
    with pytest.raises(ValueError):
        audit(config(entries))


def test_invalid_dataset_does_not_leak_paths(tmp_path):
    missing = str(tmp_path / 'private-patient-id')
    result = audit(config({'pathvqa': {'enabled': True, 'sample': {
        'dataset': missing, 'lock': missing}}}))
    assert result['benchmarks'][0]['status'] == 'blocked'
    assert 'private-patient-id' not in str(result)


def test_lock_failure_is_not_success(monkeypatch, tmp_path):
    import scripts.audit_regression_data as module
    monkeypatch.setattr(module, 'validate_dataset', lambda path: [])
    monkeypatch.setattr(module, 'load_dataset', lambda *a, **k: object())
    def drift(*args):
        raise ValueError('sensitive path and case id')
    monkeypatch.setattr(module, 'prepare', drift)
    result = audit(config({'pathvqa': {'enabled': True, 'sample': {
        'dataset': str(tmp_path), 'lock': str(tmp_path / 'lock')}}}))
    assert result['benchmarks'][0]['category'] == 'data_or_lock_invalid'
    assert 'sensitive' not in str(result)
