"""Boundary diagnostics on WildGuard labels at the setting WildGuard development data selected.

Exploratory. Reuses the checks in diagnose_null.py with k=20 and Fiedler magnitude.

Usage:
    python python/diagnose_wildguard.py --flat PATH/responses.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import diagnose_null as dn
from paired_outcomes import load_pair_table, outcomes_from_pairs, prompt_table
from run_layer2 import _attach_prompts

K = 20
BOUNDARY = 'fiedler_magnitude'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flat', type=Path, required=True, help='Staged Mistral and Zephyr responses with WildGuard labels')
    parser.add_argument('--output', type=Path, default=dn.ROOT / 'report/pilot/boundary_diagnosis_wildguard.json')
    args = parser.parse_args()
    dn.K, dn.BOUNDARY = K, BOUNDARY
    frame = pd.read_parquet(args.flat)
    prompts = prompt_table(dn.PILOT)
    pairs = load_pair_table(dn.PILOT)
    results, by_model = [], {}
    for model_id, part in frame.groupby('model_id', sort=True):
        paired, _ = outcomes_from_pairs(part, pairs, 'refusal')
        paired = _attach_prompts(paired, prompts)
        distance, _ = dn.prompt_distance(paired, prompts, dn.CACHE)
        result = dn.diagnose_model(paired, distance)
        results.append(result)
        by_model[str(model_id)] = {key: value for key, value in result.items() if key != 'scores'}
        print(model_id, by_model[str(model_id)]['boundary_auroc'], flush=True)
    payload = {'exploratory': True, 'label': 'wildguard', 'k': K, 'boundary': BOUNDARY,
               'by_model': by_model, 'pooled': dn._pooled(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
