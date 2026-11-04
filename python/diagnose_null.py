"""Explain the registered null at the already chosen boundary settings.

Uses k=15 and spectral neighbor disagreement. No setting search. MiniLM, if a
prompt vector is missing, runs on CPU so a laptop generation job keeps the GPU.

Usage:
    python python/diagnose_null.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from embed import embed_texts
from multimodel import _align, _people, feature_matrices
from paired_outcomes import load_pair_table, outcomes_from_pairs, prompt_table
from predictors import fit_predict
from run_layer2 import _attach_prompts

K = 15
BOUNDARY = 'spectral_neighbor_disagreement'
ROOT = Path(r'C:\Users\ahuja\Desktop\minor_big_data')
FLAT = ROOT / 'data/processed/analysis/layer2_flat_v7/responses.parquet'
PILOT = ROOT / 'data/processed/prompts/pilot_v5'
CACHE = ROOT / 'data/processed/analysis/embeddings/prompts.parquet'
FIGURE = ROOT / 'report/figures/layer2/boundary_vs_flip.png'
OUTPUT = ROOT / 'report/pilot/boundary_diagnosis.json'


def prompt_distance(paired, prompts, cache_path):
    ids = pd.unique(np.concatenate([
        paired['original_prompt_id'].to_numpy(), paired['variant_prompt_id'].to_numpy(),
    ]))
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=['prompt_id', 'vector'])
    lookup = {prompt_id: np.asarray(vector, dtype=np.float64)
              for prompt_id, vector in zip(cached['prompt_id'], cached['vector'])}
    missing = [prompt_id for prompt_id in ids if prompt_id not in lookup]
    if missing:
        text_of = prompts.set_index('prompt_id')['prompt_text']
        vectors = embed_texts([text_of.loc[prompt_id] for prompt_id in missing], device='cpu')
        for prompt_id, vector in zip(missing, vectors):
            lookup[prompt_id] = vector
    distance = []
    for original_id, variant_id in zip(paired['original_prompt_id'], paired['variant_prompt_id']):
        distance.append(float(1.0 - np.dot(lookup[original_id], lookup[variant_id])))
    return np.asarray(distance, dtype=np.float64), len(missing)


def _pr(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, scores))


def _auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _matrices(paired, distance):
    paired = paired.reset_index(drop=True)
    distance = np.asarray(distance, dtype=np.float64)
    development = paired['split'].eq('development')
    validation = paired['split'].eq('validation')
    dev = paired.loc[development].reset_index(drop=True)
    val = paired.loc[validation].reset_index(drop=True)
    dev_distance = distance[development.to_numpy()]
    val_distance = distance[validation.to_numpy()]
    graph_rows = pd.concat([dev, val], ignore_index=True)
    graph_people = _people(graph_rows)
    library = _people(dev)
    base_dev, full_dev, _ = feature_matrices(dev, library, _people(dev), dev_distance, K, BOUNDARY)
    base_val, full_val, _ = feature_matrices(val, library, graph_people, val_distance, K, BOUNDARY)
    full_dev, full_val = _align(full_dev, full_val)
    base_dev, base_val = _align(base_dev, base_val)
    return dev, val, base_dev, base_val, full_dev, full_val


def _fit_pr(train_x, train_y, test_x, test_y):
    scores = fit_predict(train_x, train_y, test_x)
    return _pr(test_y, scores), scores


def diagnose_model(paired, distance):
    dev, val, base_dev, base_val, full_dev, full_val = _matrices(paired, distance)
    dev_y = dev['refusal_flip'].astype(int)
    val_y = val['refusal_flip'].astype(int).to_numpy()
    refusal_only, _ = _fit_pr(
        base_dev[['original_refusal']], dev_y, base_val[['original_refusal']], val_y,
    )
    without_columns = [column for column in base_dev.columns if column != 'original_refusal']
    without, _ = _fit_pr(base_dev[without_columns], dev_y, base_val[without_columns], val_y)
    baseline, _ = _fit_pr(base_dev, dev_y, base_val, val_y)
    boundary = full_val[BOUNDARY].to_numpy()
    refused = val['original_refusal'].astype(bool).to_numpy()
    strata = {}
    for name, mask in (('original_refused', refused), ('original_complied', ~refused)):
        events = int(val_y[mask].sum())
        row = {
            'pairs': int(mask.sum()),
            'flips': events,
            'mean_boundary': float(boundary[mask].mean()) if mask.any() else None,
        }
        if events < 10:
            row['separation'] = 'too few events'
        else:
            row['boundary_auroc'] = _auroc(val_y[mask], boundary[mask])
            row['separation'] = 'auroc'
        strata[name] = row
    quartiles = _quartiles(val_y, boundary)
    return {
        'development_pairs': int(len(dev)),
        'validation_pairs': int(len(val)),
        'validation_flips': int(val_y.sum()),
        'pr_auc_original_refusal_only': refusal_only,
        'pr_auc_b0_without_original_refusal': without,
        'pr_auc_b0': baseline,
        'boundary_auroc': _auroc(val_y, boundary),
        'boundary_zero_fraction': float(np.mean(boundary == 0)),
        'chance_auroc': 0.5,
        'strata': strata,
        'quartiles': quartiles,
        'scores': {
            'labels': val_y,
            'refusal_only': fit_predict(
                base_dev[['original_refusal']], dev_y, base_val[['original_refusal']],
            ),
            'without': fit_predict(base_dev[without_columns], dev_y, base_val[without_columns]),
            'b0': fit_predict(base_dev, dev_y, base_val),
            'boundary': boundary,
        },
    }


def _quartiles(labels, scores):
    order = pd.Series(scores).rank(method='first')
    bins = pd.qcut(order, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    rows = []
    for name in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = bins.eq(name).to_numpy()
        rows.append({
            'quartile': name,
            'pairs': int(mask.sum()),
            'flips': int(labels[mask].sum()),
            'flip_rate': float(labels[mask].mean()) if mask.any() else None,
            'score_min': float(scores[mask].min()) if mask.any() else None,
            'score_max': float(scores[mask].max()) if mask.any() else None,
        })
    return rows


def _figure(by_model, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(by_model)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        rows = by_model[name]['quartiles']
        rates = [row['flip_rate'] for row in rows]
        axis.bar([row['quartile'] for row in rows], rates, color='#3b6ea5')
        for index, row in enumerate(rows):
            axis.text(index, row['flip_rate'] + 0.005, '%d/%d' % (row['flips'], row['pairs']),
                      ha='center', va='bottom', fontsize=8)
        axis.set_title(name.split('/')[-1])
        axis.set_xlabel('Boundary-score quartile')
        axis.set_ylabel('Flip rate')
    fig.suptitle('Exploratory: flip rate by spectral-neighbor-disagreement quartile, k=15')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pooled(results):
    labels = np.concatenate([row['scores']['labels'] for row in results])
    pooled = {'validation_pairs': int(labels.size), 'validation_flips': int(labels.sum())}
    for name in ('refusal_only', 'without', 'b0'):
        scores = np.concatenate([row['scores'][name] for row in results])
        pooled['pr_auc_' + name] = _pr(labels, scores)
    boundary = np.concatenate([row['scores']['boundary'] for row in results])
    pooled['boundary_auroc'] = _auroc(labels, boundary)
    pooled['flips_original_refused'] = int(sum(
        row['strata']['original_refused']['flips'] for row in results
    ))
    pooled['flips_original_complied'] = int(sum(
        row['strata']['original_complied']['flips'] for row in results
    ))
    return pooled


def main():
    frame = pd.read_parquet(FLAT)
    prompts = prompt_table(PILOT)
    pairs = load_pair_table(PILOT)
    prepared = []
    for model_id, part in frame.groupby('model_id', sort=True):
        paired, _ = outcomes_from_pairs(part, pairs, 'refusal_rules')
        paired = _attach_prompts(paired, prompts)
        distance, missing = prompt_distance(paired, prompts, CACHE)
        prepared.append((str(model_id), paired, distance, missing))
    by_model = {}
    score_rows = []
    for model_id, paired, distance, missing in prepared:
        result = diagnose_model(paired, distance)
        result['prompts_embedded_on_cpu'] = missing
        public = {key: value for key, value in result.items() if key != 'scores'}
        by_model[model_id] = public
        score_rows.append(result)
        print(model_id, public['pr_auc_b0'], public['boundary_auroc'], flush=True)
    payload = {
        'exploratory': True,
        'registered_settings': {'k': K, 'boundary': BOUNDARY},
        'note': 'Fixed settings from the Mistral and Zephyr development selection. No reselection.',
        'by_model': by_model,
        'pooled_validation_scores': _pooled(score_rows),
        'figure': str(FIGURE),
    }
    _figure({model_id: {'quartiles': by_model[model_id]['quartiles']} for model_id in by_model}, FIGURE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print('wrote', OUTPUT)


if __name__ == '__main__':
    main()
