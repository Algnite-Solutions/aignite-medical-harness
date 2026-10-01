"""Compare two completed model batches on the same MIMIC cases, without API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(root):
    batch = json.loads((root / 'batch.json').read_text())
    if batch['status'] != 'completed':
        raise ValueError(f'Batch is not completed: {root}')
    metrics = json.loads((root / 'metrics.json').read_text())
    rows = [row for p in root.glob('shards/*/decisions.jsonl') for row in read_lines(p)]
    counts, errors, repeated, tool_cases = Counter(), 0, 0, 0
    for p in root.glob('shards/*/messages.json'):
        for messages in json.loads(p.read_text()).values():
            seen = set()
            used = False
            for message in messages:
                for call in message.get('tool_calls') or []:
                    function = call['function']
                    counts[function['name']] += 1
                    used = True
                    signature = (function['name'], json.dumps(json.loads(function['arguments']), sort_keys=True))
                    repeated += signature in seen
                    seen.add(signature)
                if message['role'] == 'tool':
                    try:
                        result = json.loads(message['content'])
                    except (ValueError, TypeError):
                        continue
                    errors += isinstance(result, dict) and 'error' in result
            tool_cases += used
    retries = read_lines(root / 'transport_retries.jsonl') if (root / 'transport_retries.jsonl').exists() else []
    aggregate = metrics['scored']['aggregate']
    correct, n = aggregate['diagnosis_accuracy']['num'], batch['count']
    z = 1.959963984540054
    proportion = correct / n
    center = (proportion + z*z/(2*n)) / (1+z*z/n)
    margin = z * math.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n)) / (1+z*z/n)
    details = {
        'model':batch['model'], 'served_model':batch['model_config']['model'],
        'diagnosis_accuracy':aggregate['diagnosis_accuracy'], 'accuracy_wilson_95':[center-margin,center+margin],
        'completion':aggregate['completion'],
        'abstentions':sum(r.get('decision') is not None and r['decision'].get('answer') is None for r in rows),
        'output_quality':aggregate['output_quality'],
        'per_pathology':aggregate['per_pathology'], 'terminations':metrics['terminations'],
        'logical_model_calls':sum(r['model_calls'] for r in rows),
        'total_reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in rows),
        'median_episode_seconds':statistics.median(r['duration_ms'] for r in rows)/1000,
        'tool_calls':dict(counts), 'tool_calls_total':sum(counts.values()),
        'cases_using_tools':tool_cases, 'tool_errors':errors, 'repeated_tool_calls':repeated,
        'http_429_attempts':len(retries),
        'errors':[{'episode_id':r['episode_id'],'termination':r['termination'],'error':r.get('error')}
                  for r in rows if r['termination'] != 'completed'],
    }
    return batch, metrics['scored']['per_episode'], details


def compare(left, right, out):
    a, ap, ad = load(left)
    b, bp, bd = load(right)
    for field in ('episode_ids','variant','max_calls'):
        if a.get(field) != b.get(field):
            raise ValueError(f'Unmatched {field}')
    for filename in ('dataset.json','episodes.jsonl','targets.jsonl','eval.json','instructions.txt','cases.jsonl','lab_mapping.json'):
        pa, pb = left/'dataset'/filename, right/'dataset'/filename
        if pa.exists() != pb.exists() or (pa.exists() and pa.read_bytes() != pb.read_bytes()):
            raise ValueError(f'Unmatched dataset input: {filename}')
    counts = Counter((ap[i]['correct'],bp[i]['correct']) for i in a['episode_ids'])
    wins, losses = counts[(True,False)], counts[(False,True)]
    discordant = wins+losses
    p = min(1.0, 2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.0
    disagreements = [{'episode_id':i,'reference':ap[i]['reference_pathology'],
                       a['model']:ap[i]['predicted_pathology'],b['model']:bp[i]['predicted_pathology']}
                     for i in a['episode_ids'] if ap[i]['predicted_pathology'] != bp[i]['predicted_pathology']]
    result = {'left':ad,'right':bd,'paired':{'both_correct':counts[(True,True)],'left_only_correct':wins,
              'right_only_correct':losses,'both_wrong':counts[(False,False)],'exact_mcnemar_p':p},'disagreements':disagreements}
    shared = [i for i in a['episode_ids'] if ap[i]['termination'] == bp[i]['termination'] == 'completed']
    result['both_completed'] = {'cases':len(shared), 'left_correct':sum(ap[i]['correct'] for i in shared),
                                'right_correct':sum(bp[i]['correct'] for i in shared)}
    out.mkdir(parents=True,exist_ok=True)
    historical_path = out / 'historical_comparison.json'
    if historical_path.exists():
        result['historical'] = json.loads(historical_path.read_text())
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    def rate(value):
        return f"{value['num']}/{value['den']} ({value['value']:.1%})"
    lines=['# AntAngelMed2 vs DeepSeek interactive evaluation','',
           f"Same {a['count']} cases, temperature 0, one worker per model, 2-second request delay, 24-call cap.",
           'Dataset inputs and case order were verified equal. Both batches were rerun with the current harness.',
           '',f"| Metric | {a['model']} | {b['model']} |",'|---|---:|---:|']
    for label,key in [('Diagnosis accuracy','diagnosis_accuracy'),('Parsed decisions (including abstentions)','completion')]:
        lines.append(f'| {label} | {rate(ad[key])} | {rate(bd[key])} |')
    for label,key in [('Strict JSON','strict_output_compliance'),('Valid citation IDs','valid_citations'),('Summary present','summary_present')]:
        lines.append(f"| {label} | {rate(ad['output_quality'][key])} | {rate(bd['output_quality'][key])} |")
    lines.append(f"| Transport failures | {ad['terminations'].get('api_error',0)} | {bd['terminations'].get('api_error',0)} |")
    for label,key in [('Abstentions','abstentions'),('Model calls (excluding retried attempts)','logical_model_calls'),('Reported tokens','total_reported_tokens'),('Median episode seconds (including backoff)','median_episode_seconds'),('Tool calls','tool_calls_total'),('Cases using tools','cases_using_tools'),('Tool errors','tool_errors'),('Repeated identical tool calls','repeated_tool_calls'),('HTTP 429 attempts','http_429_attempts')]:
        lines.append(f'| {label} | {ad[key]} | {bd[key]} |')
    lines += ['', '## Diagnosis breakdown','',f"| Diagnosis | {a['model']} | {b['model']} |",'|---|---:|---:|']
    for label in ad['per_pathology']:
        x,y=ad['per_pathology'][label],bd['per_pathology'][label]
        lines.append(f"| {label} | {x['correct']}/{x['total']} | {y['correct']}/{y['total']} |")
    lines += ['', '## Paired outcomes','',
              f"Both correct: {counts[(True,True)]}; AntAngel only: {wins}; DeepSeek only: {losses}; both wrong: {counts[(False,False)]}.",
              f'Two-sided exact McNemar p = {p:.4g}. This is a 100-case sample, not a claim about all cases.',
              f"On the {len(shared)} cases completed by both models: AntAngel {result['both_completed']['left_correct']} correct; DeepSeek {result['both_completed']['right_correct']} correct. This subset includes abstentions and may be selected by transport success.",
              '', '## Setup and interpretation','',
              '- Native AntAngel automatic tool calling returned HTTP 400: server auto-tool-choice/parser flags were missing.',
              '- Its verified compatibility mode sends tools with tool_choice=none and parses the native tool_call/arg_key/arg_value tags locally. Raw output is retained; no extra prompts or repair calls.',
              '- DeepSeek uses native structured tool calls. The served model names are recorded in comparison.json and config.json.',
              '- DeepSeek requests enable_thinking=false; AntAngel retains its endpoint default. This compares these configured deployments, not identical reasoning modes.',
              '- Accuracy includes transport and parsing failures. Citation checks only validate released IDs; summary checks only establish presence. Treatment quality is not scored.',
              '- The gateway explicitly reported TPM (tokens-per-minute) quota exhaustion in the saved synthetic probe. Its exact allowance is unknown.',
              '- 429 retries, shared endpoint load, and observed host pauses confound latency. Reported tokens exclude unreported failed-request usage.',
              '- Historical scores are not pooled into these fresh batches. Pilots are excluded from the 100-case scores.', '']
    if 'historical' in result:
        h = result['historical']
        paired = h['both_completed']
        lines += ['## Historical DeepSeek comparison', '',
                  'The historical run used the same 100 case IDs in the same order, identical dataset files and the same actual system prompt.',
                  'Its original batch scored 82/100 with 15 transport failures. The published 83/100 includes one successful retry (episode 27738411), leaving 14 failures.',
                  f"Among the {paired['cases']} cases with parsed Decisions in both AntAngel and the retry-adjusted historical run, AntAngel got {paired['antangel_correct']} correct and historical DeepSeek got {paired['deepseek_correct']} correct.",
                  'This completed-only subset can be biased. The historical timeout was 60 seconds versus 120 seconds now, and serving conditions differ. The apparent 90% versus 83% gap is not a clean model-quality advantage.', '']
    (out/'review.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left',type=Path);parser.add_argument('right',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();compare(args.left,args.right,args.out)
