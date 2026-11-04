"""Summarize completed runs and compare truncation by observed refusal label."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
from run_artifacts import ROOT, discover_runs, intent, load_run


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def describe(run):
    context = run['context']
    manifest = context['run_manifests'][0]
    model_id = manifest['runtime']['configuration']['model_id']
    model = next(m for m in context['model_manifests'][0]['models'] if m['model_id'] == model_id)
    family = model['family']
    family = next((name for name in ['Llama', 'Qwen', 'Mistral', 'SmolLM2']
                   if family.startswith(name)), family)
    size = re.search(r'(\d+(?:\.\d+)?)[Bb](?:-|$)', model_id)
    return {'run_id': manifest['run_id'], 'model': model_id, 'family': family,
            'size_b': float(size.group(1)) if size else None,
            'backend': manifest['runtime']['engine'],
            'max_new_tokens': manifest['generation']['max_new_tokens'],
            'selection_limit': manifest['runtime']['configuration']['selection']['limit']}


def summarize_run(run):
    meta = describe(run)
    prompts = {p['prompt_id']: p for p in run['context']['prompts']}
    splits = run['context']['prompt_manifests'][0]['group_splits']
    labels = {(r['response_id'], r['labeler_id']): r for r in run['labels']}
    if len(labels) != len(run['labels']):
        raise ValueError('Multiple label revisions require a separate behavior table')
    labelers = sorted({r['labeler_id'] for r in run['labels']}) or ['unlabeled']
    groups = defaultdict(list)
    truncation = defaultdict(list)
    for response in run['responses']:
        prompt = prompts[response['prompt_id']]
        split = splits[prompt['prompt_group_id']]
        for labeler in labelers:
            label = labels.get((response['response_id'], labeler), {})
            refused = label.get('refusal') if not label.get('error') else None
            key = (split, prompt['source_id'], intent(prompt), prompt['variant_type'], labeler)
            groups[key].append((response, refused))
            truncation[(split, labeler)].append((response, refused))
    rows = []
    for key, values in sorted(groups.items()):
        valid = [(r, label) for r, label in values if r['status'] == 'success' and label is not None]
        successful = [r for r, _ in values if r['status'] == 'success']
        rows.append({**meta, **dict(zip(['split', 'source', 'intent', 'variant_type', 'label_source'], key)),
                     'n': len(values), 'n_failed': len(values) - len(successful), 'n_valid_labels': len(valid),
                     'refusal_rate': rate(sum(label for _, label in valid), len(valid)),
                     'truncation_rate': rate(sum(r['output_truncated'] for r in successful), len(successful)),
                     'median_output_tokens': statistics.median(r['output_tokens'] for r in successful)
                     if successful else None})
    associations = []
    for (split, labeler), values in sorted(truncation.items()):
        valid = [(r, label) for r, label in values if r['status'] == 'success' and label is not None]
        refused = [r for r, label in valid if label]
        nonrefused = [r for r, label in valid if not label]
        truncated = [(r, label) for r, label in valid if r['output_truncated']]
        uncut = [(r, label) for r, label in valid if not r['output_truncated']]
        associations.append({**meta, 'split': split, 'label_source': labeler, 'n_valid': len(valid),
                             'n_refused': len(refused), 'n_nonrefused': len(nonrefused),
                             'n_truncated': len(truncated),
                             'truncation_given_refused': rate(sum(r['output_truncated'] for r in refused), len(refused)),
                             'truncation_given_nonrefused': rate(sum(r['output_truncated'] for r in nonrefused), len(nonrefused)),
                             'refusal_given_truncated': rate(sum(label for _, label in truncated), len(truncated)),
                             'refusal_given_uncut': rate(sum(label for _, label in uncut), len(uncut))})
    return rows, associations


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build(run_folders, output, validated_runs=None):
    output.mkdir(parents=True, exist_ok=True)
    rows, associations, manifest = [], [], []
    for folder in run_folders:
        print('Validating behavior input: ' + str(folder), flush=True)
        run = (validated_runs or {}).get(Path(folder).resolve())
        if run is None:
            run = load_run(folder)
        if not run['stats'].get('completed'):
            continue
        table, cut = summarize_run(run)
        rows.extend(table)
        associations.extend(cut)
        manifest.append({**describe(run), 'folder': str(folder), 'responses': len(run['responses']),
                         'labels': len(run['labels'])})
    write_csv(output / 'behavior_summary.csv', rows)
    write_csv(output / 'truncation_by_refusal.csv', associations)
    notes = ['# Behavior and response truncation', '',
             'Rows are separated by run, split, source, intent, variant and evaluator. Refusal rates use valid labels;',
             'truncation rates and output-token medians use successful responses. Smoke runs retain their run ID and',
             'selection limit. Replicas of the same run are counted once. No live-run output is included.', '',
             'The laptop validation runs use a 256-token cap; the two Kaggle runs use 512. These tables describe',
             'each recorded run, not a controlled estimate of model size or family effects. Compare source, intent',
             'and variant strata rather than treating aggregate refusal as a safety score.', '',
             '| Model | Split | Label source | Refused n | Other n | Cut off among refused | Cut off among other |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in associations:
        if 'zephyr' not in row['model'].lower():
            continue
        fmt = lambda x: f'{100*x:.1f}%' if x is not None else 'unavailable'
        notes.append(f"| {row['model']} | {row['split']} | {row['label_source']} | {row['n_refused']} | "
                     f"{row['n_nonrefused']} | {fmt(row['truncation_given_refused'])} | "
                     f"{fmt(row['truncation_given_nonrefused'])} |")
    has_wg = any(row['label_source'] == 'wildguard' and row['n_valid'] for row in associations)
    notes.extend(['', 'Cut-off status is associated with the observed refusal decision; this comparison does not identify',
                  'label errors. Short refusals and long compliant answers can create the same association. A cut-off',
                  'answer can also omit a later refusal or harmful continuation. The unseen continuation is unavailable.',
                  '', 'WildGuard labels are included where present.' if has_wg else
                  'No valid WildGuard labels are available in these completed local runs yet.',
                  'The current phrase-rule results therefore do not measure WildGuard misclassification. Recompute this',
                  'table after ingestion and compare the human audit on cut-off and uncut answers. Keep the registered',
                  'generation caps fixed; report the existing cut-offs and evaluate excluding them as a sensitivity check.',
                  'Excluding cut-off responses changes the response mix; it does not correct the missing continuations.'])
    (output / 'truncation_note.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    (output / 'behavior_sources.json').write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(), 'runs': manifest}, indent=2) + '\n')
    print(f'Wrote behavior tables for {len(manifest)} finished runs ({len(rows)} grouped rows)', flush=True)
    return rows, associations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses-root', type=Path, default=ROOT / 'data/processed/responses')
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.runs or discover_runs(args.responses_root), args.output)
