"""Sample validation responses and score refusal_rules against a blind review.

The draw writes prompts and responses only. Rule labels stay in a separate key
file and are joined after the review.

Usage:
    python python/rule_label_check.py draw --output-dir DIR
    python python/rule_label_check.py score --key DIR/key.json --labels labels.csv --output report.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = (
    'mistralai/Mistral-7B-Instruct-v0.1',
    'HuggingFaceH4/zephyr-7b-beta',
)
SEED = 20261104


def draw_sample(frame, seed=SEED, size=100, max_positives=30, models=None):
    """All rule refusals up to max_positives, then rule non-refusals, per model."""
    rng = np.random.default_rng(seed)
    validation = frame[frame['split'].eq('validation')].copy()
    pieces = []
    for model_id in (MODELS if models is None else models):
        part = validation[validation['model_id'].eq(model_id)]
        rules = part['refusal_rules'].fillna(False).astype(bool)
        positives = part[rules]
        negatives = part[~rules]
        n_pos = min(max_positives, len(positives))
        pos_at = rng.choice(len(positives), size=n_pos, replace=False)
        pos = positives.iloc[np.sort(pos_at)].copy()
        n_neg = size - n_pos
        if n_neg > len(negatives):
            raise ValueError('Not enough rule non-refusals to fill the sample')
        neg_at = rng.choice(len(negatives), size=n_neg, replace=False)
        neg = negatives.iloc[np.sort(neg_at)].copy()
        pos['stratum'] = 'rule_refused'
        neg['stratum'] = 'rule_not_refused'
        pos['stratum_n'] = int(len(positives))
        neg['stratum_n'] = int(len(negatives))
        pos['sample_n'] = int(n_pos)
        neg['sample_n'] = int(n_neg)
        pieces.append(pos)
        pieces.append(neg)
    sample = pd.concat(pieces, ignore_index=True)
    sample['weight'] = sample['stratum_n'] / sample['sample_n']
    return sample


def load_response_text(roots, wanted):
    """Read response text for the given ids from responses-*.jsonl shards."""
    found = {}
    remaining = set(wanted)
    for root in roots:
        for path in sorted(Path(root).rglob('responses-*.jsonl')):
            if not remaining:
                break
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    response_id = row.get('response_id')
                    if response_id in remaining and row.get('status') == 'success':
                        found[response_id] = row.get('text') or ''
                        remaining.remove(response_id)
    if remaining:
        raise ValueError('Missing response text for %d ids' % len(remaining))
    return found


def write_blind_chunks(sample, texts, prompts, directory, per_file=8, prompt_tail=900):
    directory = Path(directory)
    chunk_dir = directory / 'chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in sample.itertuples(index=False):
        prompt = prompts.get(row.prompt_id, '')
        rows.append({
            'response_id': row.response_id,
            'model_id': row.model_id,
            'prompt_tail': prompt[-prompt_tail:],
            'response': texts[row.response_id],
        })
    for index in range(0, len(rows), per_file):
        chunk = rows[index:index + per_file]
        lines = []
        for number, row in enumerate(chunk, start=index + 1):
            lines.append('--- %d %s %s' % (number, row['model_id'].split('/')[-1], row['response_id']))
            lines.append('PROMPT_TAIL:')
            lines.append(row['prompt_tail'])
            lines.append('RESPONSE:')
            lines.append(row['response'])
            lines.append('')
        (chunk_dir / ('%03d.txt' % (index // per_file))).write_text('\n'.join(lines), encoding='utf-8')
    return len(rows)


def rule_metrics(rows):
    """Precision on the rule-positive stratum. Recall uses sampling weights."""
    frame = pd.DataFrame(rows)
    scored = frame[frame['agent_refusal'].notna()].copy()
    scored['agent_refusal'] = scored['agent_refusal'].astype(bool)
    scored['refusal_rules'] = scored['refusal_rules'].astype(bool)
    by_model = {}
    for model_id, part in scored.groupby('model_id'):
        by_model[model_id] = _one_model(part)
    return {'labeler': 'llm_review', 'by_model': by_model, 'scored_rows': int(len(scored)),
            'declined': int(frame['agent_refusal'].isna().sum())}


def _one_model(part):
    positive = part[part['refusal_rules']]
    negative = part[~part['refusal_rules']]
    true_positive = int(positive['agent_refusal'].sum())
    false_positive = int((~positive['agent_refusal']).sum())
    false_negative = int(negative['agent_refusal'].sum())
    precision = true_positive / len(positive) if len(positive) else None
    weight_tp = float(positive.loc[positive['agent_refusal'], 'weight'].sum())
    weight_fn = float(negative.loc[negative['agent_refusal'], 'weight'].sum())
    recall = weight_tp / (weight_tp + weight_fn) if (weight_tp + weight_fn) else None
    population = float(part['weight'].sum())
    estimated_refusals = weight_tp + weight_fn
    return {
        'n': int(len(part)),
        'rule_refused_in_sample': int(len(positive)),
        'rule_not_refused_in_sample': int(len(negative)),
        'agent_refusal_given_rule': true_positive,
        'agent_compliance_given_rule': false_positive,
        'agent_refusal_given_not_rule': false_negative,
        'precision': precision,
        'recall': recall,
        'estimated_refusal_rate': estimated_refusals / population if population else None,
        'estimated_refusals': estimated_refusals,
        'population': population,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    draw = commands.add_parser('draw')
    draw.add_argument('--flat', type=Path, nargs='+', required=True)
    draw.add_argument('--models', nargs='+')
    draw.add_argument('--prompts', type=Path, required=True)
    draw.add_argument('--responses', type=Path, nargs='+', required=True)
    draw.add_argument('--output-dir', type=Path, required=True)
    draw.add_argument('--seed', type=int, default=SEED)
    score = commands.add_parser('score')
    score.add_argument('--key', type=Path, required=True)
    score.add_argument('--labels', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--summary', type=Path)
    args = parser.parse_args()
    if args.command == 'draw':
        frame = pd.concat([pd.read_parquet(path) for path in args.flat], ignore_index=True)
        sample = draw_sample(frame, args.seed, models=args.models)
        prompts = {}
        with args.prompts.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    prompts[row['prompt_id']] = row['messages'][-1]['content']
        texts = load_response_text(args.responses, set(sample['response_id']))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        key_columns = [
            'response_id', 'model_id', 'prompt_id', 'variant_type', 'stratum',
            'refusal_rules', 'weight', 'stratum_n', 'sample_n',
        ]
        key = sample[key_columns].copy()
        key['refusal_rules'] = key['refusal_rules'].astype(bool)
        key.to_json(args.output_dir / 'key.json', orient='records')
        n = write_blind_chunks(sample, texts, prompts, args.output_dir)
        print(json.dumps({'rows': n, 'seed': args.seed, 'dir': str(args.output_dir)}))
        return
    key = pd.read_json(args.key)
    labels = pd.read_csv(args.labels)
    merged = key.merge(labels[['response_id', 'agent_refusal', 'agent_note']], on='response_id', how='left')
    note = merged['agent_note'].fillna('').astype(str)
    missing = merged['agent_refusal'].isna() & ~note.str.lower().eq('declined')
    if bool(missing.any()):
        raise ValueError('Unlabeled rows: %d' % int(missing.sum()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    summary = rule_metrics(merged.to_dict('records'))
    summary_path = args.summary or args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
