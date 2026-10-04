"""Local evaluation of the completed single-pass PR13 baseline. No model calls."""
from __future__ import annotations

import collections
import argparse
import json
import math
import os
from pathlib import Path

ROOT = None
REPO = None
VIEWS = ('hpi', 'interactive', 'full_info')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(n, total):
    z = 1.959963984540054
    p = n / total
    d = 1 + z*z/total
    c = (p + z*z/(2*total))/d
    h = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [c-h, c+h]


def main():
    global ROOT, REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    ROOT, REPO = args.run.resolve(), args.repository.resolve()
    os.umask(0o077)
    output = {'protocol': 'PR13 fixed open100, three views once, hosted Bailian Qwen35',
        'status': 'completed', 'cases': 100, 'encounters': 300, 'views': {}, 'paired': {},
        'prior_model_comparisons': {}, 'limitations': [
            'Automatic diagnosis mapping is not blinded clinical adjudication.',
            'The fixed balanced cohort is exploratory and overlaps previously examined cases.',
            'View changes bundle different instructions and evidence access; differences do not isolate a mechanism.',
            'One generation per view cannot estimate five-repeat consistency AUC.',
            'The unchanged task does not request confidence; no missing confidence or logprob is imputed.',
            'Hosted API identity is provider provenance, not checkpoint attestation or an on-premise reproduction.',
            'Wilson intervals are descriptive binomial intervals and do not establish population representativeness.',
            'Unique admission IDs do not certify unique patients; between-episode independence is unverified.',
        ]}
    per = {}
    decisions = {}
    private_errors = []
    model_ids = set()
    finish_reasons = collections.Counter()
    for view in VIEWS:
        merged = ROOT / view / 'merged'
        metrics = read(merged / 'metrics.json')['scored']
        manifest = read(merged / 'manifest.json')
        messages = read(merged / 'messages.json')
        ids = manifest['episode_ids']
        assert len(ids) == 100 and len(set(ids)) == 100
        per[view] = metrics['per_episode']
        decisions[view] = {r['episode_id']: r for r in rows(merged / 'decisions.jsonl')}
        assert list(decisions[view]) == ids
        calls = [r for r in rows(merged / 'diagnostics.jsonl') if r['kind'] == 'model_call']
        aggregate = metrics['aggregate']
        for c in calls:
            meta = c.get('response_metadata') or {}
            if meta.get('model'):
                model_ids.add(meta['model'])
            finish_reasons[meta.get('finish_reason')] += 1
        error_types = collections.Counter()
        tool_counts = collections.Counter()
        confidence_count = 0
        episodes_with_tools = 0
        missing_diagnosis = 0
        abstentions = 0
        answer_types = collections.Counter()
        for case_id in ids:
            score = per[view][case_id]
            row = decisions[view][case_id]
            answer = (row.get('decision') or {}).get('answer')
            diagnosis_text = score.get('diagnosis_text')
            answer_types[type(answer).__name__] += 1
            if isinstance(answer, dict) and type(answer.get('confidence')) in (int, float):
                confidence_count += 1
            episodes_with_tools += any(m.get('tool_calls') for m in messages[case_id] if m.get('role') == 'assistant')
            if row['termination'] == 'completed':
                abstentions += answer is None
                missing_diagnosis += answer is not None and (not isinstance(diagnosis_text, str) or not diagnosis_text.strip())
            for message in messages[case_id]:
                if message.get('role') == 'assistant':
                    for call in message.get('tool_calls') or []:
                        tool_counts[(call.get('function') or {}).get('name', 'unknown')] += 1
            if not score['correct_auto']:
                category = ('run_or_parse_failure' if row['termination'] != 'completed' else
                    'explicit_abstention' if answer is None else
                    'missing_diagnosis_field' if score['mapping'] == 'missing' else
                    'unmapped_or_ambiguous_diagnosis' if score.get('predicted_pathology') is None else
                    'mapped_to_different_source_group')
                error_types[category] += 1
                private_errors.append({'view': view, 'episode_id': case_id, 'category': category,
                    'score': score, 'termination': row['termination'], 'error': row['error'],
                    'citations': (row.get('decision') or {}).get('citations', []),
                    'reasoning_summary': (row.get('decision') or {}).get('reasoning_summary'),
                    'model_calls': row['model_calls'],
                    'retrieved_tools': [c.get('function', {}).get('name')
                        for m in messages[case_id] if m.get('role') == 'assistant'
                        for c in m.get('tool_calls') or []]})
        n = aggregate['diagnosis_accuracy_auto']['num']
        output['views'][view] = {
            'correct': n, 'total': 100, 'accuracy_auto': n/100,
            'descriptive_wilson_95_ci': wilson(n, 100),
            'completed': aggregate['completion']['num'],
            'mapped': aggregate['diagnosis_mapped']['num'],
            'needs_review': aggregate['needs_review'],
            'per_pathology': aggregate['per_pathology'],
            'output_quality': aggregate['output_quality'],
            'error_types': dict(error_types), 'tool_counts': dict(tool_counts),
            'model_calls': len(calls), 'tool_calls': sum(tool_counts.values()),
            'numeric_confidence_present': confidence_count,
            'episodes_with_tools': episodes_with_tools,
            'answer_value_types': dict(answer_types),
            'explicit_abstention_cases': abstentions, 'missing_diagnosis_cases': missing_diagnosis,
            'terminations': dict(collections.Counter(r['termination'] for r in decisions[view].values())),
            'usage': {key: sum((r.get('usage') or {}).get(key, 0) for r in decisions[view].values())
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
            'model_call_failures': sum(bool(c.get('error')) for c in calls),
        }
    assert all(list(per[v]) == list(per['hpi']) for v in VIEWS)
    for left, right in [('interactive', 'hpi'), ('full_info', 'interactive'), ('full_info', 'hpi')]:
        c = collections.Counter()
        for case_id in per[left]:
            a, b = bool(per[left][case_id]['correct_auto']), bool(per[right][case_id]['correct_auto'])
            c['both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'neither'] += 1
        output['paired'][left + '_vs_' + right] = {k: c[k] for k in ['both_correct', 'left_only', 'right_only', 'neither']}
    for model in ['antangelmed2', 'deepseek-v4-flash', 'glm-5.3-flash']:
        comparisons = {}
        for view in VIEWS:
            oldroot = REPO / 'runs/mimic-cdm-open-v1-100-live' / model / view / 'merged'
            oldmanifest = read(oldroot / 'manifest.json')
            assert oldmanifest['episode_ids'] == list(per[view])
            old = read(oldroot / 'metrics.json')['scored']['per_episode']
            c = collections.Counter()
            for case_id in per[view]:
                a, b = bool(per[view][case_id]['correct_auto']), bool(old[case_id]['correct_auto'])
                c['both_correct' if a and b else 'qwen_only' if a else 'prior_only' if b else 'neither'] += 1
            comparisons[view] = {k: c[k] for k in ['both_correct', 'qwen_only', 'prior_only', 'neither']}
            comparisons[view]['prior_correct'] = sum(bool(x['correct_auto']) for x in old.values())
        output['prior_model_comparisons'][model] = comparisons
    ledger = read(ROOT / 'budget.json')
    output['cost'] = {'limit_cny': 35, 'accounted_cny': ledger['accounted_ncny']/1e9,
        'http_attempts': len(ledger['requests']),
        'failed_attempts': sum(bool(r.get('error')) for r in ledger['requests']),
        'usage_unknown_attempts': sum(not r.get('usage_known', False) for r in ledger['requests']),
        'known_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in ledger['requests'] if r.get('usage_known')),
        'known_completion_tokens': sum(r['usage']['completion_tokens'] for r in ledger['requests'] if r.get('usage_known')),
        'price_source': ledger['price_source']}
    output['returned_models'] = sorted(model_ids)
    output['finish_reasons'] = dict(finish_reasons)
    scan = {
        'Simpson paradox': 'Checked aggregate and four source-group directions; no reversal observed.',
        'Ecological fallacy': 'Admission-level paired summaries; no patient-specific reliability inference.',
        'Berkson selection': 'CAUTION: selected hospitalized abdominal cases, not a general clinical population.',
        'Collider bias': 'No adjusted regression or collider controls used; no causal claims made.',
        'Base rate neglect': 'CAUTION: 25 cases per group is artificial balance, not disease prevalence.',
        'Regression to mean': 'No training pre/post claim or extreme-score selection for this run.',
        'Survivorship bias': 'All 100 cases per view retained, including the invalid decision.',
        'Look elsewhere': 'All three view scores and paired transitions reported; no significance testing.',
        'Forking paths': 'NOTE: reused exploratory cohort; frozen scope, one encounter each, no answer replacement.',
        'Correlation versus causation': 'CAUTION: view instructions and evidence access differ together; no isolated mechanism inference.',
        'Reverse causality': 'No directional inference from tool use versus accuracy; descriptive observations only.',
    }
    for left, right in [('interactive', 'hpi'), ('full_info', 'interactive'), ('full_info', 'hpi')]:
        overall = output['views'][left]['correct'] - output['views'][right]['correct']
        differences = [output['views'][left]['per_pathology'][g]['correct'] -
            output['views'][right]['per_pathology'][g]['correct']
            for g in output['views'][left]['per_pathology']]
        if overall and all(d * overall < 0 for d in differences):
            scan['Simpson paradox'] = 'FLAG: aggregate/subgroup reversal observed; investigate.'
    output['fallacy_scan'] = {'checked': '11/11', 'findings': scan}
    output['interpretation'] = 'A hosted same-cohort model baseline; neither robust reproduction nor rejection of the paper AUC observation.'
    (ROOT / 'analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    private = ROOT / 'private'
    private.mkdir(exist_ok=True, mode=0o700)
    (private / 'errors.json').write_text(json.dumps(private_errors, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Qwen3.6-35B-A3B: 100-case three-view baseline', '', '## Material Passport', '',
        '- Status: completed single-pass hosted baseline; not an AUC reproduction.',
        '- Cohort: existing PR13 reviewer-open100, seed 42, 25 cases per source group.',
        '- Configuration: temperature 0, thinking disabled, max_tokens 4096, max_calls 24, timeout 120 s.',
        '- Endpoint model returned: ' + ', '.join(output['returned_models']), '',
        '| View | Automatic correct | Completed | Unmapped / needs review | Model calls |',
        '|---|---:|---:|---:|---:|']
    for v, a in output['views'].items():
        lines.append(f"| {v} | {a['correct']}/100 | {a['completed']}/100 | {a['needs_review']} | {a['model_calls']} |")
    lines += ['', 'Scores use the unchanged PR13 automatic source-group mapper. Unmapped answers require blinded clinical review; they are not confirmed clinical mistakes.', '',
        '## Paired changes', '', '| Left versus right | Both correct | Left only | Right only | Neither |', '|---|---:|---:|---:|---:|']
    for name, a in output['paired'].items():
        lines.append(f"| {name} | {a['both_correct']} | {a['left_only']} | {a['right_only']} | {a['neither']} |")
    lines += ['', '## Remaining errors', '', '| View | Different mapped group | Unmapped or ambiguous | Explicit abstention | Missing diagnosis | Run / parse failure |', '|---|---:|---:|---:|---:|---:|']
    for v, a in output['views'].items():
        e = a['error_types']
        lines.append(f"| {v} | {e.get('mapped_to_different_source_group', 0)} | {e.get('unmapped_or_ambiguous_diagnosis', 0)} | {e.get('explicit_abstention', 0)} | {e.get('missing_diagnosis_field', 0)} | {e.get('run_or_parse_failure', 0)} |")
    lines += ['', 'Private error records include case IDs, mapped reference/prediction, cited evidence and retrieved tools in `private/errors.json`. No case text was sent to an analysis model.', '',
        '## Existing same-cohort baselines', '', '| Model | HPI | Interactive | Full info |', '|---|---:|---:|---:|']
    for model, comps in output['prior_model_comparisons'].items():
        lines.append('| ' + model + ' | ' + ' | '.join(str(comps[v]['prior_correct'])+'/100' for v in VIEWS) + ' |')
    lines += ['', '## Tool access and answers', '']
    for v, a in output['views'].items():
        lines.append(f"- {v}: {a['episodes_with_tools']} cases used tools; {a['explicit_abstention_cases']} explicit abstentions; {a['missing_diagnosis_cases']} other valid-envelope answers with no scorer-readable diagnosis. String answers are retained by the unchanged scorer; answer value types: {a['answer_value_types']}.")
    cost = output['cost']
    lines += ['', 'These are archived model deployments on matched cases, not randomized evidence of training benefit. Full pairwise tables are in `analysis.json`.', '',
        '## Usage and failures', '',
        f"- Accounted cost: CNY {cost['accounted_cny']:.6f}, against the CNY35 ceiling; includes the synthetic check.",
        f"- HTTP attempts: {cost['http_attempts']}; failed transport attempts: {cost['failed_attempts']}; unknown-usage attempts: {cost['usage_unknown_attempts']}.",
        f"- Reported input tokens: {cost['known_prompt_tokens']}; output tokens: {cost['known_completion_tokens']}.",
        '- Pricing: [official Beijing model prices](' + cost['price_source'] + '). Provider billing is authoritative; cached discounts are not assumed.', '',
        '## Limits on interpretation', '']
    lines += ['- '+x for x in output['limitations']]
    lines += ['', '## Statistical interpretation check', '', 'ARS fallacy coverage: 11/11 checked. These checks do not certify clinical reliability.', '']
    lines += ['- ' + name + ': ' + finding for name, finding in output['fallacy_scan']['findings'].items()]
    (ROOT / 'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'views': {v: a['correct'] for v, a in output['views'].items()}, 'cost': output['cost'], 'report': str(ROOT/'report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
