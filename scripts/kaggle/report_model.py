"""Export per-model generation and label summaries as soon as a run finishes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from data_contracts import validate_bundle, verify_artifacts
from evaluate_responses import prompt_intent
from generate_responses import Shards, atomic_write, read_json


def report_model(run, output):
    context, stats = read_json(run / 'run.bundle.json'), read_json(run / 'stats.json')
    responses = list(Shards(run, 'responses', 'response_id').rows.values())
    labels = []
    for path in (run / 'labels').glob('*/labeler.json'):
        labels.extend(Shards(path.parent, 'response_labels', 'label_id').rows.values())
    validate_bundle({**context, 'responses': responses, 'response_labels': labels})
    verify_artifacts(context)
    prompts = {p['prompt_id']: p for p in context['prompts']}
    splits = context['prompt_manifests'][0]['group_splits']
    indexed = {(r['response_id'], r['labeler_id']): r for r in labels}
    if len(indexed) != len(labels):
        raise ValueError('Ambiguous labeler versions in model report')
    labelers = sorted({r['labeler_id'] for r in labels})
    rates = defaultdict(lambda: {'responses': 0, 'valid_refusal_labels': 0, 'refusals': 0,
                                 'valid_harm_labels': 0, 'harmful_responses': 0})
    for row in responses:
        p = prompts[row['prompt_id']]
        intent, _ = prompt_intent(p)
        intent = 'benign' if intent == 'unharmful' else intent
        for labeler in labelers:
            key = (p['source_id'], intent, p['variant_type'], splits[p['prompt_group_id']], labeler)
            rate = rates[key];rate['responses'] += 1
            label = indexed.get((row['response_id'], labeler), {})
            for field, count, positives in [('refusal', 'valid_refusal_labels', 'refusals'),
                                             ('response_harmful', 'valid_harm_labels', 'harmful_responses')]:
                if label.get(field) is not None:
                    rate[count] += 1;rate[positives] += label[field]
    table = []
    for key, counts in sorted(rates.items()):
        table.append(dict(zip(['source', 'intent', 'variant_type', 'split', 'labeler'], key),
                          **counts, refusal_rate=counts['refusals'] / counts['valid_refusal_labels']
                          if counts['valid_refusal_labels'] else None,
                          harmful_response_rate=counts['harmful_responses'] / counts['valid_harm_labels']
                          if counts['valid_harm_labels'] else None))
    pair_path = ROOT / 'data/processed/prompts/pilot_v5/pairs.jsonl'
    pairs = [json.loads(line) for line in pair_path.read_text(encoding='utf-8').splitlines()]
    selected = set(context['run_manifests'][0]['runtime']['configuration']['selection']['prompt_ids'])
    by_prompt = {r['prompt_id']: r for r in responses if r['status'] == 'success'}
    flips = []
    for labeler in labelers:
        total, valid, changed = 0, 0, 0
        for pair in pairs:
            ids = [pair['original_prompt_id'], pair['variant_prompt_id']]
            if not all(pid in selected for pid in ids) or splits[prompts[ids[0]]['prompt_group_id']] != 'validation':
                continue
            total += 1
            values = [indexed.get((by_prompt[pid]['response_id'], labeler), {}).get('refusal')
                      if pid in by_prompt else None for pid in ids]
            if all(v is not None for v in values):
                valid += 1;changed += values[0] != values[1]
        flips.append({'labeler': labeler, 'validation_pairs': total, 'valid_pairs': valid,
                      'excluded_pairs': total - valid, 'flips': changed,
                      'flip_rate': changed / valid if valid else None})
    summary = {'model_id': stats['model_id'], 'run_id': stats['run_id'], 'completed': stats['completed'],
               'responses': len(responses), 'failures': stats['failures'],
               'input_truncated': stats['input_truncated'], 'output_truncated': stats['output_truncated'],
               'wall_seconds': stats['wall_seconds'], 'output_tokens_per_second': stats['output_tokens_per_wall_second'],
               'generation': context['run_manifests'][0]['generation'],
               'stratum': 'laptop_small' if context['run_manifests'][0]['runtime']['engine'] == 'transformers'
               else 'kaggle_7_8b', 'rates': table, 'validation_refusal_flips': flips,
               'label_errors': {name: sum(r['error'] is not None for r in labels if r['labeler_id'] == name)
                                for name in labelers},
               'interpretation': 'refusal_rules are a phrase-based proxy; harm requires WildGuard/audited labels. '
                                 'These descriptive validation results do not establish held-out predictive benefit.'}
    output.mkdir(parents=True, exist_ok=True)
    stem = stats['model_id'].replace('/', '__') + '__' + stats['run_id'][4:20]
    atomic_write(output / (stem + '.json'), summary)
    if table:
        with (output / (stem + '_rates.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('Per-model report:', output / (stem + '.json'), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_model(args.run, args.output)
