"""Pair original and variant responses and define refusal-flip outcomes.

Usage:
    python python/paired_outcomes.py --flat data/processed/analysis/layer2_flat_fallback/responses.parquet
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def read_jsonl_shards(folder, prefix):
    rows = []
    for path in sorted(Path(folder).glob(prefix + '-*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_table(pilot_dir):
    path = Path(pilot_dir) / 'prompts.jsonl'
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels = row.get('source_labels') or {}
        text = row['messages'][-1]['content']
        rows.append({
            'prompt_id': row['prompt_id'],
            'group_id': row['prompt_group_id'],
            'source': row['source_id'],
            'intent': labels.get('prompt_intent'),
            'variant_type': row['variant_type'],
            'prompt_chars': len(text),
            'prompt_text': text,
        })
    return pd.DataFrame(rows)


def outcomes_from_frame(frame, label_column):
    """One row per variant that shares a group with exactly one original.

    `label_column` is `refusal` for WildGuard or `refusal_rules` for the rule labeler.
    """
    work = frame.copy()
    if label_column not in work.columns:
        raise ValueError('Missing label column ' + label_column)
    work['refusal_label'] = work[label_column]
    failed = 0
    if 'status' in work.columns:
        failed = int((work['status'] != 'success').sum())
        work = work[work['status'] == 'success']
    exclusions = {
        'missing_label': int(work['refusal_label'].isna().sum()),
        'failed_response': failed,
    }
    work = work[work['refusal_label'].notna()].copy()
    if 'split' not in work.columns:
        work['split'] = 'validation'
    originals = work[work['variant_type'] == 'original'].copy()
    variants = work[work['variant_type'] != 'original'].copy()
    counts = originals.groupby('group_id').size()
    good_groups = set(counts[counts == 1].index)
    exclusions['groups_without_one_original'] = int((counts != 1).sum())
    exclusions['variants_without_original'] = int((~variants['group_id'].isin(good_groups)).sum())
    variants = variants[variants['group_id'].isin(good_groups)].copy()
    originals = originals[originals['group_id'].isin(good_groups)].copy()
    keep = [
        'model_id', 'group_id', 'source', 'intent', 'split', 'prompt_id', 'response_id',
        'refusal_label', 'response_harmful', 'output_tokens', 'text_chars', 'vector',
    ]
    for column in keep:
        if column not in originals.columns:
            originals[column] = pd.NA
        if column not in variants.columns:
            variants[column] = pd.NA
    left = originals[keep].rename(columns={
        'prompt_id': 'original_prompt_id',
        'response_id': 'original_response_id',
        'refusal_label': 'original_refusal',
        'response_harmful': 'original_harmful',
        'output_tokens': 'original_output_tokens',
        'text_chars': 'original_text_chars',
        'vector': 'original_vector',
    })
    right_cols = [
        'group_id', 'variant_type', 'prompt_id', 'response_id', 'refusal_label', 'response_harmful',
    ]
    right = variants[right_cols].rename(columns={
        'prompt_id': 'variant_prompt_id',
        'response_id': 'variant_response_id',
        'refusal_label': 'variant_refusal',
        'response_harmful': 'variant_harmful',
    })
    paired = left.merge(right, on='group_id', how='inner')
    paired['refusal_flip'] = paired['original_refusal'].astype(bool) != paired['variant_refusal'].astype(bool)
    paired['flip_direction'] = 'none'
    refused = paired['original_refusal'].astype(bool) & ~paired['variant_refusal'].astype(bool)
    complied = ~paired['original_refusal'].astype(bool) & paired['variant_refusal'].astype(bool)
    paired.loc[refused, 'flip_direction'] = 'refused_to_complied'
    paired.loc[complied, 'flip_direction'] = 'complied_to_refused'
    orig_harm = paired['original_harmful']
    var_harm = paired['variant_harmful']
    paired['harmful_transition'] = pd.NA
    known = orig_harm.notna() & var_harm.notna()
    paired.loc[known, 'harmful_transition'] = (
        ~orig_harm[known].astype(bool) & var_harm[known].astype(bool)
    )
    return paired.reset_index(drop=True), exclusions


def load_flat(path, label_column):
    frame = pd.read_parquet(path)
    return outcomes_from_frame(frame, label_column)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flat', type=Path, required=True)
    parser.add_argument('--label-column', default='refusal_rules')
    args = parser.parse_args()
    paired, exclusions = load_flat(args.flat, args.label_column)
    print(json.dumps({
        'pairs': int(len(paired)),
        'refusal_flip_rate': float(paired['refusal_flip'].mean()) if len(paired) else None,
        'exclusions': exclusions,
    }, indent=2))


if __name__ == '__main__':
    main()
