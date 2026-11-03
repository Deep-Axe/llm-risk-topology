"""Run the response-graph predictors and write the report figures.

Usage:
    python python/run_layer2.py --flat data/processed/analysis/layer2_flat_fallback/responses.parquet
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from embed import embed_texts, unit_rows, write_embeddings
from paired_outcomes import load_flat, prompt_table
from predictors import (
    _disagreement_against,
    design_columns,
    metric_block,
    oof_predictions,
    spectral_scores,
)
from response_graph import graph_spectrum


def _attach_prompts(paired, prompts):
    lengths = prompts.set_index('prompt_id')['prompt_chars']
    texts = prompts.set_index('prompt_id')['prompt_text']
    paired = paired.copy()
    paired['original_prompt_chars'] = lengths.reindex(paired['original_prompt_id']).to_numpy()
    paired['variant_prompt_chars'] = lengths.reindex(paired['variant_prompt_id']).to_numpy()
    paired['original_prompt_text'] = texts.reindex(paired['original_prompt_id']).to_numpy()
    paired['variant_prompt_text'] = texts.reindex(paired['variant_prompt_id']).to_numpy()
    return paired


def _prompt_distances(paired):
    ids = pd.unique(np.concatenate([
        paired['original_prompt_id'].to_numpy(), paired['variant_prompt_id'].to_numpy(),
    ]))
    text_of = {}
    for prompt_id, text in zip(paired['original_prompt_id'], paired['original_prompt_text']):
        text_of[prompt_id] = text
    for prompt_id, text in zip(paired['variant_prompt_id'], paired['variant_prompt_text']):
        text_of[prompt_id] = text
    vectors = embed_texts([text_of[prompt_id] for prompt_id in ids])
    lookup = {prompt_id: vector for prompt_id, vector in zip(ids, vectors)}
    distance = []
    for original_id, variant_id in zip(paired['original_prompt_id'], paired['variant_prompt_id']):
        distance.append(float(1.0 - np.dot(lookup[original_id], lookup[variant_id])))
    frame = pd.DataFrame({
        'response_id': '',
        'prompt_id': ids,
        'model_id': 'prompt',
        'vector': list(vectors),
    })
    return np.asarray(distance, dtype=np.float64), frame


def _original_matrix(paired):
    originals = paired.drop_duplicates('original_response_id')
    vectors = unit_rows(np.vstack(originals['original_vector'].to_numpy()))
    labels = originals['original_refusal'].astype(bool).to_numpy()
    index = {response_id: at for at, response_id in enumerate(originals['original_response_id'])}
    return originals, vectors, labels, index


def _fold_disagreement(paired, vectors, labels, index, groups, k, space):
    """Neighbor label disagreement using training-fold originals only."""
    from sklearn.model_selection import GroupKFold
    scores = np.zeros(len(paired), dtype=np.float64)
    row_groups = paired['group_id'].to_numpy()
    splitter = GroupKFold(n_splits=min(5, len(np.unique(row_groups))))
    dummy = np.zeros(len(paired))
    for train_at, test_at in splitter.split(dummy, dummy, row_groups):
        train_ids = paired.iloc[train_at]['original_response_id'].unique()
        train_pos = np.array([index[response_id] for response_id in train_ids])
        library = space[train_pos]
        library_labels = labels[train_pos]
        query_pos = np.array([index[response_id] for response_id in paired.iloc[test_at]['original_response_id']])
        query = space[query_pos]
        query_labels = labels[query_pos]
        # A query that is also in the library must not count itself.
        rates = _disagreement_against(query, query_labels, library, library_labels, k)
        scores[test_at] = rates
    return scores


def _feature_frame(paired, extra):
    base = pd.DataFrame({
        'original_refusal': paired['original_refusal'].astype(float).to_numpy(),
        'original_response_chars': paired['original_text_chars'].astype(float).to_numpy(),
        'original_prompt_chars': paired['original_prompt_chars'].astype(float).to_numpy(),
        'variant_prompt_chars': paired['variant_prompt_chars'].astype(float).to_numpy(),
        'prompt_distance': extra['prompt_distance'],
        'raw_neighbor_disagreement': extra['raw_neighbor_disagreement'],
        'source': paired['source'].astype(str).to_numpy(),
        'intent': paired['intent'].fillna('missing').astype(str).to_numpy(),
        'variant_type': paired['variant_type'].astype(str).to_numpy(),
    })
    numeric = [
        'original_refusal', 'original_response_chars', 'original_prompt_chars',
        'variant_prompt_chars', 'prompt_distance', 'raw_neighbor_disagreement',
    ]
    categorical = ['source', 'intent', 'variant_type']
    return design_columns(base, numeric, categorical)


def _rates(paired):
    grouped = paired.groupby(['model_id', 'source', 'intent'], dropna=False)['original_refusal']
    refusal = grouped.mean().reset_index().rename(columns={'original_refusal': 'refusal_rate'})
    refusal['n'] = grouped.size().to_numpy()
    flips = paired.groupby(['model_id', 'variant_type'])['refusal_flip']
    flip = flips.mean().reset_index().rename(columns={'refusal_flip': 'flip_rate'})
    flip['n'] = flips.size().to_numpy()
    return refusal, flip


def _save_figures(folder, paired, spectrum, labels, baseline, full, delta, lower, upper, model_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    folder.mkdir(parents=True, exist_ok=True)
    refusal, flip = _rates(paired)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    labels_x = [f'{row.source}\n{row.intent}' for row in refusal.itertuples()]
    axis.bar(labels_x, refusal['refusal_rate'])
    axis.set_ylim(0, 1)
    axis.set_ylabel('Refusal rate')
    axis.set_xlabel('Prompt source and intent')
    axis.set_title(f'How often {model_name} refuses, by prompt source and intent')
    fig.tight_layout()
    fig.savefig(folder / 'refusal_by_source_intent.png', dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar(flip['variant_type'], flip['flip_rate'])
    axis.set_ylim(0, 1)
    axis.set_ylabel('Refusal-flip rate')
    axis.set_xlabel('Variant type')
    axis.set_title(f'How often refusal changes from the original prompt ({model_name})')
    fig.tight_layout()
    fig.savefig(folder / 'flip_by_variant.png', dpi=150)
    plt.close(fig)

    vectors = spectrum['eigenvectors']
    coords = vectors[:, 1:3] if vectors.shape[1] > 2 else vectors[:, :2]
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    refused = np.asarray(labels, dtype=bool)
    axis.scatter(coords[~refused, 0], coords[~refused, 1], s=12, label='Did not refuse', c='C0')
    axis.scatter(coords[refused, 0], coords[refused, 1], s=12, label='Refused', c='C1')
    axis.set_xlabel('Second eigenvector')
    axis.set_ylabel('Third eigenvector')
    axis.set_title(f'Spectral map of {model_name.split("/")[-1]} responses,\ncolored by refusal')
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / 'spectral_embedding.png', dpi=150)
    plt.close(fig)

    values = spectrum['eigenvalues']
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(np.arange(1, len(values) + 1), values, marker='o')
    gap_at = spectrum['eigengap_index']
    axis.axvline(gap_at + 1.5, color='C1', linestyle='--', label='Largest gap')
    axis.set_xlabel('Eigenvalue rank (smallest first)')
    axis.set_ylabel('Eigenvalue')
    axis.set_title(f'Response-graph eigenvalues for {model_name}')
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / 'eigenvalues.png', dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 5))
    for score, name in ((baseline, 'Baselines'), (full, 'Baselines + boundary')):
        precision, recall, _ = precision_recall_curve(paired['refusal_flip'].astype(int), score)
        axis.plot(recall, precision, label=name)
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(
        'Validation precision-recall\nÎ”PR-AUC %.3f (95%% interval %.3f to %.3f)'
        % (delta, lower, upper)
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / 'pr_curve.png', dpi=150)
    plt.close(fig)


def run(flat, pilot, overlap_path, figure_dir, results_path, embedding_dir, k_values, n_boot, seed):
    paired, exclusions = load_flat(flat, 'refusal_rules')
    prompts = prompt_table(pilot)
    paired = _attach_prompts(paired, prompts)
    prompt_distance, prompt_embed = _prompt_distances(paired)
    originals, vectors, refusal, index = _original_matrix(paired)
    raw_disagree = _fold_disagreement(paired, vectors, refusal, index, paired['group_id'], k_values[0], vectors)
    extra = {
        'prompt_distance': prompt_distance,
        'raw_neighbor_disagreement': raw_disagree,
    }
    baseline_x = _feature_frame(paired, extra)
    labels = paired['refusal_flip'].astype(int).to_numpy()
    groups = paired['group_id'].to_numpy()
    baseline_score = oof_predictions(baseline_x, labels, groups)
    spectra = {}
    candidates = {}
    for k in k_values:
        spectrum = graph_spectrum(vectors, k)
        scores = spectral_scores(spectrum['eigenvalues'], spectrum['eigenvectors'], spectrum['eigengap_index'])
        spectra[k] = spectrum
        for name in ('fiedler_magnitude', 'spectral_margin'):
            column = pd.DataFrame({name: scores[name][paired['original_response_id'].map(index).to_numpy()]})
            full_x = pd.concat([baseline_x, column], axis=1)
            full_score = oof_predictions(full_x, labels, groups)
            block = metric_block(labels, baseline_score, full_score, groups, n_boot, seed)
            candidates[(k, name)] = {'metrics': block, 'score': full_score, 'boundary': scores[name]}
        spectral_space = unit_rows(spectrum['eigenvectors'])
        spectral_disagree = _fold_disagreement(
            paired, vectors, refusal, index, paired['group_id'], k, spectral_space,
        )
        column = pd.DataFrame({'spectral_neighbor_disagreement': spectral_disagree})
        full_x = pd.concat([baseline_x, column], axis=1)
        full_score = oof_predictions(full_x, labels, groups)
        block = metric_block(labels, baseline_score, full_score, groups, n_boot, seed)
        candidates[(k, 'spectral_neighbor_disagreement')] = {
            'metrics': block, 'score': full_score, 'boundary': spectral_disagree,
        }
    best_key = max(candidates, key=lambda key: candidates[key]['metrics']['full_pr_auc'])
    best = candidates[best_key]
    chosen_k, chosen_name = best_key
    spectrum = spectra[chosen_k]
    overlap = set()
    if overlap_path and Path(overlap_path).exists():
        text = Path(overlap_path).read_text(encoding='utf-8')
        overlap = {line.strip() for line in text.splitlines() if line.strip()}
    keep = ~paired['group_id'].isin(overlap)
    overlap_metrics = None
    if int(keep.sum()) > 20 and paired.loc[keep, 'refusal_flip'].nunique() > 1:
        overlap_metrics = metric_block(
            labels[keep.to_numpy()], baseline_score[keep.to_numpy()], best['score'][keep.to_numpy()],
            groups[keep.to_numpy()], n_boot, seed,
        )
    refusal_table, flip_table = _rates(paired)
    model_name = str(paired['model_id'].iloc[0])
    _save_figures(
        Path(figure_dir), paired, spectrum, refusal, baseline_score, best['score'],
        best['metrics']['delta_pr_auc'], best['metrics']['delta_lower'], best['metrics']['delta_upper'],
        model_name,
    )
    embedding_dir = Path(embedding_dir)
    response_embed = pd.DataFrame({
        'response_id': originals['original_response_id'].to_numpy(),
        'prompt_id': originals['original_prompt_id'].to_numpy(),
        'model_id': originals['model_id'].to_numpy(),
        'vector': list(vectors),
    })
    write_embeddings(embedding_dir / 'responses.parquet', response_embed)
    write_embeddings(embedding_dir / 'prompts.parquet', prompt_embed)
    results = {
        'fit': '5-fold GroupKFold on validation groups. This run has no development split.',
        'graph_scope': 'k-NN graph uses original responses only. Variant responses are not nodes.',
        'label_source': 'refusal_rules',
        'model_id': model_name,
        'n_pairs': int(len(paired)),
        'n_originals': int(len(originals)),
        'refusal_flip_rate': float(paired['refusal_flip'].mean()),
        'flip_direction_counts': paired['flip_direction'].value_counts().to_dict(),
        'exclusions': exclusions,
        'chosen_k': int(chosen_k),
        'chosen_boundary': chosen_name,
        'components': spectrum['n_components'],
        'eigengap': spectrum['eigengap'],
        'eigengap_index': int(spectrum['eigengap_index']),
        'eigenvalues': [float(value) for value in spectrum['eigenvalues']],
        'metrics': best['metrics'],
        'metrics_without_overlap_groups': overlap_metrics,
        'n_pairs_in_overlap_groups': int((~keep).sum()),
        'variant_refusal_rate': paired.groupby(
            ['source', 'intent', 'variant_type'], dropna=False,
        )['variant_refusal'].agg(['mean', 'size']).reset_index().to_dict(orient='records'),
        'candidate_pr_auc': {
            '%d:%s' % (k, name): candidates[(k, name)]['metrics']['full_pr_auc']
            for k, name in candidates
        },
        'refusal_by_source_intent': refusal_table.to_dict(orient='records'),
        'flip_by_variant': flip_table.to_dict(orient='records'),
        'baseline_only_pr_auc': float(best['metrics']['baseline_pr_auc']),
    }
    # exclusions were not returned through _attach; reload count is in the caller if needed
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({
        'pairs': results['n_pairs'],
        'flip_rate': results['refusal_flip_rate'],
        'chosen': results['chosen_boundary'],
        'k': results['chosen_k'],
        'delta': results['metrics']['delta_pr_auc'],
        'interval': [results['metrics']['delta_lower'], results['metrics']['delta_upper']],
    }, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flat', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('data/processed/prompts/pilot_v5'))
    parser.add_argument(
        '--overlap-groups', type=Path,
        default=Path(
            r'C:\Users\ahuja\Desktop\minor_big_data\report\evaluator'
            r'\wildguardmix_train_overlap_groups_v5.txt'
        ),
    )
    parser.add_argument(
        '--figures', type=Path,
        default=Path(r'C:\Users\ahuja\Desktop\minor_big_data\report\figures\layer2'),
    )
    parser.add_argument(
        '--results', type=Path,
        default=Path(r'C:\Users\ahuja\Desktop\minor_big_data\report\pilot\layer2_results.json'),
    )
    parser.add_argument('--embeddings', type=Path, default=Path('data/processed/analysis/embeddings'))
    parser.add_argument('--boot', type=int, default=199)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    run(args.flat, args.pilot, args.overlap_groups, args.figures, args.results, args.embeddings,
        [10, 15, 20], args.boot, args.seed)


if __name__ == '__main__':
    main()
