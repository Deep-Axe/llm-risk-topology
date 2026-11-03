"""Flatten generated responses, labels, prompt metadata and MiniLM embeddings into one Parquet file for HDFS.

Run: python python/stage_layer2.py --runs RUN_FOLDER [RUN_FOLDER ...] --output data/processed/analysis/layer2_flat
Each run folder holds responses-*.jsonl and labels/<labeler>-<rev>/response_labels-*.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINILM = ('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41')


def read_shards(folder, pattern):
    rows = []
    for path in sorted(folder.glob(pattern)):
        with path.open(encoding='utf-8') as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--prompts', type=Path, default=ROOT / 'data/processed/prompts/pilot_v5')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/analysis/layer2_flat')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    bundle = json.loads((args.prompts / 'experiment.bundle.json').read_text(encoding='utf-8'))
    splits = bundle['prompt_manifests'][0]['group_splits']
    prompts = {}
    with (args.prompts / 'prompts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            p = json.loads(line)
            labels = p['source_labels']
            intent = labels.get('prompt_intent') or ('unharmful' if labels.get('label') == 'safe' else 'harmful')
            prompts[p['prompt_id']] = {'split': splits[p['prompt_group_id']], 'group_id': p['prompt_group_id'],
                                       'source': p['source_id'].split('/')[-1], 'intent': intent,
                                       'variant_type': p['variant_type'], 'prompt_text': p['messages'][-1]['content']}
    rows = []
    for run in args.runs:
        responses = read_shards(run, 'responses-*.jsonl')
        labels = {}
        for folder in sorted((run / 'labels').glob('*')):
            labeler = folder.name.split('-')[0]
            for row in read_shards(folder, 'response_labels-*.jsonl'):
                labels.setdefault(row['response_id'], {})[labeler] = row
        for r in responses:
            if r['status'] != 'success':
                continue
            wildguard = labels.get(r['response_id'], {}).get('wildguard', {})
            rules = labels.get(r['response_id'], {}).get('refusal_rules', {})
            rows.append({'response_id': r['response_id'], 'model_id': r['model_id'], 'prompt_id': r['prompt_id'],
                         **{k: v for k, v in prompts[r['prompt_id']].items() if k != 'prompt_text'},
                         'output_tokens': r['output_tokens'], 'output_truncated': r['output_truncated'],
                         'refusal': wildguard.get('refusal'), 'response_harmful': wildguard.get('response_harmful'),
                         'prompt_harmful': wildguard.get('prompt_harmful'), 'refusal_rules': rules.get('refusal'),
                         'text': r['text']})
    frame = pd.DataFrame(rows)
    for column in ['refusal', 'response_harmful', 'prompt_harmful', 'refusal_rules']:
        frame[column] = frame[column].astype('boolean')
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM[0], revision=MINILM[1])
    vectors = model.encode(frame['text'].tolist(), batch_size=args.batch_size, normalize_embeddings=True,
                           show_progress_bar=True)
    frame['vector'] = [v.astype('float32').tolist() for v in vectors]
    frame['text_chars'] = frame['text'].str.len()
    args.output.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=['text']).to_parquet(args.output / 'responses.parquet', index=False)
    summary = frame.groupby('model_id').agg(responses=('response_id', 'size'),
                                            wildguard_refusal=('refusal', 'mean'),
                                            rules_refusal=('refusal_rules', 'mean')).reset_index()
    print(summary.to_string(index=False))
    print('Wrote', args.output / 'responses.parquet', len(frame), 'rows')


if __name__ == '__main__':
    main()
