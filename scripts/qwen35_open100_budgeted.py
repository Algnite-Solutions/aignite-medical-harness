"""One authorized PR13 open100 pass with a persisted CNY 35 request ledger."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

from ama.cli import _eval_run, _load_dotenv
from ama.data import load_dataset
from ama.model import Model, ModelError
from ama.recorder import write_json
from ama.runner import execute
from scripts.check_qwen35_open100 import check
from scripts.mimic_cdm_batch import _decision, merge_batch

ALIAS = 'qwen36-35b-a3b-bailian'
WIRE_MODEL = 'qwen3.6-35b-a3b'
MAX_INPUT = 260096
INPUT_NCNY = 1800
OUTPUT_NCNY = 10800
LIMIT_NCNY = 35_000_000_000
SOURCE = 'https://help.aliyun.com/zh/model-studio/qwen3-6-35b-a3b'


class BudgetStop(ModelError):
    pass


class Ledger:
    """Reserve a whole model-context charge before EACH HTTP attempt.

    Unknown usage, failed requests and crash-in-flight keep the full reservation.
    Successful reported usage can refund unused reservation. Integer nanoyuan
    avoid cumulative floating-point budget drift. Caller holds one process lock.
    """
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {
            'limit_ncny': LIMIT_NCNY, 'accounted_ncny': 0, 'requests': [],
            'price_source': SOURCE, 'input_ncny_per_token': INPUT_NCNY,
            'output_ncny_per_token': OUTPUT_NCNY, 'max_input_tokens': MAX_INPUT,
        }
        if self.data['limit_ncny'] != LIMIT_NCNY:
            raise ValueError('saved budget differs')
        self.save()

    def save(self):
        write_json(self.path, self.data)

    def reserve(self, max_tokens: int, context: dict) -> int:
        reserve = MAX_INPUT * INPUT_NCNY + max_tokens * OUTPUT_NCNY
        if self.data['accounted_ncny'] + reserve > LIMIT_NCNY:
            raise BudgetStop('CNY 35 budget: insufficient room for the next full-context reservation')
        self.data['accounted_ncny'] += reserve
        self.data['requests'].append({**context, 'at': time.time(), 'status': 'reserved',
            'reserved_ncny': reserve, 'charged_ncny': reserve, 'max_output_tokens': max_tokens})
        self.save()
        return len(self.data['requests']) - 1

    def settle(self, index: int, usage, metadata, error=None):
        row = self.data['requests'][index]
        row.update(status='finished', usage=usage, response_metadata=metadata,
                   error=error, usage_known=False)
        if isinstance(usage, dict):
            prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
            reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            if (type(prompt) is int and type(completion) is int and
                    0 <= prompt <= MAX_INPUT and 0 <= completion <= row['max_output_tokens'] and
                    reasoning == 0):
                charge = prompt * INPUT_NCNY + completion * OUTPUT_NCNY
                self.data['accounted_ncny'] -= row['charged_ncny'] - charge
                row.update(charged_ncny=charge, usage_known=True)
            else:
                row['usage_contract_violation'] = True
        self.save()


class BudgetModel(Model):
    def __init__(self, name, config, ledger, timeout=120):
        super().__init__(name, config, timeout)
        if (config['model'] != WIRE_MODEL or config.get('enable_thinking') is not False or
                config['base_url'] != 'https://dashscope.aliyuncs.com/compatible-mode/v1' or
                not config.get('require_response_model') or not 1 <= config.get('max_tokens', 0) <= 4096):
            raise ValueError('budgeted run requires frozen Beijing Qwen35 non-thinking config')
        self.ledger, self.context, self.stop_reason = ledger, {}, None

    def complete(self, history, tools=None):
        for attempt in range(3):
            # Model resets per-call telemetry; reset here too when budget stops before HTTP.
            self.last_usage = self.last_response_metadata = None
            try:
                index = self.ledger.reserve(self.config['max_tokens'], {**self.context, 'transport_attempt': attempt + 1})
            except BudgetStop as exc:
                self.stop_reason = str(exc)
                raise
            error = None
            retryable = False
            try:
                time.sleep(3)
                reply = super().complete(history, tools)
            except ModelError as exc:
                error = str(exc)
                retryable = exc.http_status in {429, 500, 502, 503, 504} or error.startswith('model connection failed:')
                if not retryable or attempt == 2:
                    self.stop_reason = error
                    raise
            finally:
                self.ledger.settle(index, self.last_usage, self.last_response_metadata, error)
            if self.ledger.data['requests'][index].get('usage_contract_violation'):
                self.stop_reason = 'provider token usage violated the reserved non-thinking contract'
                raise BudgetStop(self.stop_reason)
            if error is None:
                return reply
            if retryable:
                time.sleep(min(5 * 2 ** attempt, 20))
        raise AssertionError('unreachable')


def summary(root, batches, ledger):
    report = {'views': {}, 'budget_limit_cny': 35, 'accounted_cny': ledger.data['accounted_ncny'] / 1e9,
        'http_attempts': len(ledger.data['requests']), 'known_prompt_tokens': 0,
        'known_completion_tokens': 0, 'unknown_usage_attempts': 0, 'paired': {}}
    for req in ledger.data['requests']:
        if req.get('usage_known'):
            report['known_prompt_tokens'] += req['usage']['prompt_tokens']
            report['known_completion_tokens'] += req['usage']['completion_tokens']
        else:
            report['unknown_usage_attempts'] += 1
    report['known_usage_cost_cny'] = (report['known_prompt_tokens'] * INPUT_NCNY +
        report['known_completion_tokens'] * OUTPUT_NCNY) / 1e9
    per = {}
    for view, batch in batches.items():
        rows = [_decision(Path(run)) for run in batch['runs'].values()]
        counts = {}
        for row in rows:
            counts[row['termination']] = counts.get(row['termination'], 0) + 1
        value = {'attempted_cases': len(rows), 'completed_cases': counts.get('completed', 0),
            'terminations': counts, 'model_calls': sum(row['model_calls'] for row in rows),
            'failures': [{'episode_id': r['episode_id'], 'termination': r['termination'], 'error': r['error']}
                for r in rows if r['termination'] != 'completed']}
        metrics_path = root / view / 'merged' / 'metrics.json'
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())['scored']
            value['scores'] = metrics['aggregate']
            per[view] = metrics['per_episode']
        report['views'][view] = value
    for left, right in [('interactive', 'hpi'), ('full_info', 'interactive'), ('full_info', 'hpi')]:
        if left in per and right in per:
            pairs = {'both_correct': 0, 'left_only': 0, 'right_only': 0, 'neither': 0}
            for case_id in per[left]:
                a, b = bool(per[left][case_id]['correct_auto']), bool(per[right][case_id]['correct_auto'])
                key = 'both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'neither'
                pairs[key] += 1
            report['paired'][left + '_vs_' + right] = pairs
    write_json(root / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, default=Path('runs/reviewer-open100'))
    args = parser.parse_args()
    os.umask(0o077)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / 'cohort_check.json', check(args.cohort, Path('runs/mimic-cdm-open-v1-100-seed42')))
        _load_dotenv()
        if not os.environ.get('DASHSCOPE_API_KEY'):
            _load_dotenv(Path('/Users/bran/Documents/Projects/medical model training/.env'))
        config = json.loads(Path('ama.json').read_text())['models'][ALIAS]
        ledger = Ledger(root / 'budget.json')
        model = BudgetModel(ALIAS, config, ledger)
        write_json(root / 'execution_config.json', {'model': ALIAS, 'model_config': config,
            'timeout': 120, 'max_calls': 24, 'delay': 3, 'transport_attempts_max': 3,
            'cases_per_view': 100, 'views': ['hpi', 'interactive', 'full_info'], 'budget_cny': 35,
            'authorization': 'User explicitly authorized reviewer-open100, three views once, Bailian Qwen35, CNY35; relayed from source thread 01a0ff97-fd24-7be1-9647-e5f92768eb59.',
            'price_source': SOURCE})
        if not (root / 'synthetic_smoke.json').exists():
            smoke_config = dict(config, max_tokens=64)
            smoke = BudgetModel(ALIAS, smoke_config, ledger)
            smoke.context = {'phase': 'synthetic_smoke'}
            reply = smoke.complete([{'role': 'user', 'content': 'Synthetic connectivity test only. Reply with the word OK.'}])
            write_json(root / 'synthetic_smoke.json', {'response_metadata': smoke.last_response_metadata,
                'usage': smoke.last_usage, 'assistant_response_present': isinstance(reply.get('content'), str)})
            print('Synthetic smoke verified exact model identity.', flush=True)
        batches = {}
        try:
            for view in ['hpi', 'interactive', 'full_info']:
                folder = args.cohort / ('mimic_cdm_open_' + view)
                ids = [ep.id for ep in load_dataset(folder).episodes]
                out = root / view
                out.mkdir(exist_ok=True)
                record = out / 'batch.json'
                tools = Path('src/ama/importers/mimic_cdm_tools.py').resolve() if view == 'interactive' else None
                settings = {'dataset_dir': str(folder.resolve()), 'episode_ids': ids, 'model': ALIAS,
                    'model_config': config, 'timeout': 120, 'max_calls': 24,
                    'tools': str(tools) if tools else None, 'delay': 3}
                if record.exists():
                    batch = json.loads(record.read_text())
                    if any(batch.get(k) != v for k, v in settings.items()):
                        raise ValueError('resume settings differ')
                else:
                    batch = {**settings, 'status': 'running', 'runs': {}, 'attempts': {}, 'started_at': time.time()}
                batches[view] = batch
                write_json(record, batch)
                for case_id in ids:
                    # One encounter per case; never automatically resubmit failed answers.
                    if case_id in batch['runs']:
                        continue
                    model.context = {'phase': 'clinical', 'view': view, 'episode_id': case_id}
                    with (root / 'runner.log').open('a') as log, contextlib.redirect_stdout(log):
                        run = execute(folder, model, episode_ids=[case_id], runs_root=out / 'shards',
                            tools_path=tools, max_calls=24)
                    batch['runs'][case_id] = str(run)
                    batch['attempts'][case_id] = [str(run)]
                    write_json(record, batch)
                    rep = summary(root, batches, ledger)
                    print(json.dumps({'view': view, 'attempted': len(batch['runs']),
                        'completed': rep['views'][view]['completed_cases'],
                        'accounted_cny': rep['accounted_cny'], 'http_attempts': rep['http_attempts']}), flush=True)
                    if _decision(run)['termination'] == 'interrupted':
                        model.stop_reason = 'interrupted'
                    if model.stop_reason:
                        batch.update(status='stopped', stop_reason=model.stop_reason)
                        write_json(record, batch)
                        raise BudgetStop(model.stop_reason)
                with (root / 'runner.log').open('a') as log, contextlib.redirect_stdout(log):
                    merged = merge_batch(out, batch)
                completed = all(_decision(Path(p))['termination'] == 'completed' for p in batch['runs'].values())
                batch.update(status='completed' if completed else 'completed_with_errors', merged_run=str(merged), finished_at=time.time())
                write_json(record, batch)
                summary(root, batches, ledger)
        except (BudgetStop, KeyboardInterrupt) as exc:
            write_json(root / 'stop.json', {'reason': str(exc) or 'interrupted', 'at': time.time()})
            print('Execution stopped: ' + (str(exc) or 'interrupted'), flush=True)
            return 2
        finally:
            summary(root, batches, ledger)
        print('Completed all 300 encounters; summary.json saved.', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
