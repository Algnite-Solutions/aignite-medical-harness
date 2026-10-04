import json
import urllib.error
import urllib.request

import pytest

from scripts.qwen35_open100_budgeted import (Ledger, BudgetModel, BudgetStop,
    ALIAS, INPUT_NCNY, OUTPUT_NCNY, MAX_INPUT, LIMIT_NCNY)
from test_model_client import Response


def test_crash_reservation_persists_and_success_refunds(tmp_path):
    path = tmp_path / 'budget.json'
    ledger = Ledger(path)
    index = ledger.reserve(4096, {'phase': 'synthetic'})
    reserved = MAX_INPUT * INPUT_NCNY + 4096 * OUTPUT_NCNY
    assert Ledger(path).data['accounted_ncny'] == reserved
    ledger.settle(index, {'prompt_tokens': 100, 'completion_tokens': 20}, {'model': 'fixture'})
    assert Ledger(path).data['accounted_ncny'] == 100 * INPUT_NCNY + 20 * OUTPUT_NCNY


def test_unknown_and_invalid_usage_keep_reservation(tmp_path):
    ledger = Ledger(tmp_path / 'budget.json')
    index = ledger.reserve(4096, {})
    reserved = ledger.data['accounted_ncny']
    ledger.settle(index, None, None, 'connection failure')
    assert ledger.data['accounted_ncny'] == reserved
    second = ledger.reserve(4096, {})
    ledger.settle(second, {'prompt_tokens': 1, 'completion_tokens': 4097}, {})
    assert ledger.data['accounted_ncny'] == 2 * reserved
    assert ledger.data['requests'][second]['usage_contract_violation']


def test_budget_stops_before_network(monkeypatch, tmp_path):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-test')
    ledger = Ledger(tmp_path / 'budget.json')
    ledger.data['accounted_ncny'] = LIMIT_NCNY
    ledger.save()
    config = json.loads(open('ama.json').read())['models'][ALIAS]
    model = BudgetModel(ALIAS, config, ledger)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('budget stop must not send'))
    with pytest.raises(BudgetStop):
        model.complete([])
    assert len(ledger.data['requests']) == 0
    assert model.stop_reason


def test_every_transport_attempt_reserved_and_retries_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-test')
    monkeypatch.setattr('scripts.qwen35_open100_budgeted.time.sleep', lambda *_: None)
    ledger = Ledger(tmp_path / 'budget.json')
    config = json.loads(open('ama.json').read())['models'][ALIAS]
    model = BudgetModel(ALIAS, config, ledger)
    sent = []
    def fail(request, timeout):
        sent.append(1)
        assert len(ledger.data['requests']) == len(sent)
        assert ledger.data['requests'][-1]['status'] == 'reserved'
        raise urllib.error.HTTPError('https://example.test', 429, 'limited', {}, None)
    monkeypatch.setattr(urllib.request, 'urlopen', fail)
    with pytest.raises(Exception, match='HTTP 429'):
        model.complete([{'role': 'user', 'content': 'synthetic'}])
    assert len(sent) == 3 and len(ledger.data['requests']) == 3
    assert ledger.data['accounted_ncny'] == 3 * (MAX_INPUT * INPUT_NCNY + 4096 * OUTPUT_NCNY)
    assert model.stop_reason


def test_successful_request_refunds_reported_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-test')
    monkeypatch.setattr('scripts.qwen35_open100_budgeted.time.sleep', lambda *_: None)
    ledger = Ledger(tmp_path / 'budget.json')
    config = json.loads(open('ama.json').read())['models'][ALIAS]
    model = BudgetModel(ALIAS, config, ledger)
    def respond(request, timeout):
        assert ledger.data['requests'][-1]['status'] == 'reserved'
        return Response(json.dumps({'model': config['model'], 'choices': [
            {'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 50, 'completion_tokens': 1}}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', respond)
    assert model.complete([{'role': 'user', 'content': 'synthetic'}])['content'] == 'OK'
    assert ledger.data['accounted_ncny'] == 50 * INPUT_NCNY + OUTPUT_NCNY
    assert ledger.data['requests'][0]['response_metadata']['model'] == config['model']
