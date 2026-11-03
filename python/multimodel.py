"""Score one response graph per model and pool the validation deltas.

Usage:
    python python/run_layer2.py --flat a.parquet b.parquet
"""
import numpy as np
import pandas as pd

from embed import unit_rows
from predictors import fit_predict, metric_block, oof_predictions, spectral_scores
from response_graph import graph_spectrum


BOUNDARY_NAMES = ('fiedler_magnitude', 'spectral_margin', 'spectral_neighbor_disagreement')
KNOWN_MODELS = {
    'Qwen/Qwen2.5-0.5B-Instruct': ('Qwen', '0.5B'),
    'Qwen/Qwen2.5-1.5B-Instruct': ('Qwen', '1.5B'),
    'Qwen/Qwen2.5-7B-Instruct': ('Qwen', '7B'),
    'meta-llama/Llama-3.2-1B-Instruct': ('Llama', '1B'),
    'meta-llama/Llama-3.1-8B-Instruct': ('Llama', '8B'),
    'NousResearch/Hermes-3-Llama-3.1-8B': ('Llama', '8B'),
    'dphn/dolphin-2.9.4-llama3.1-8b': ('Llama', '8B'),
    'allenai/Llama-3.1-Tulu-3-8B': ('Llama', '8B'),
    'mistralai/Mistral-7B-Instruct-v0.1': ('Mistral', '7B'),
    'HuggingFaceH4/zephyr-7b-beta': ('Mistral', '7B'),
    'teknium/OpenHermes-2.5-Mistral-7B': ('Mistral', '7B'),
}


def family_size(model_id):
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]
    return ('unknown', 'unknown')


def library_disagreement(query_ids, query_vectors, query_labels, library_ids, library_vectors, library_labels, k):
    """Label disagreement against a library. A query id that is in the library is not its own neighbor."""
    n_library = library_vectors.shape[0]
    if n_library < 2:
        return np.zeros(len(query_ids), dtype=np.float64)
    similarity = query_vectors @ library_vectors.T
    position = {response_id: index for index, response_id in enumerate(library_ids)}
    for row, response_id in enumerate(query_ids):
        if response_id in position:
            similarity[row, position[response_id]] = -np.inf
    usable = min(int(k), n_library - 1)
    usable = max(usable, 1)
    neighbors = np.argpartition(-similarity, usable - 1, axis=1)[:, :usable]
    disagreed = library_labels[neighbors] != query_labels[:, None]
    return disagreed.mean(axis=1)


def _people(paired):
    from run_layer2 import _original_matrix
    originals, vectors, labels, index = _original_matrix(paired)
    return {
        'frame': originals,
        'ids': originals['original_response_id'].to_numpy(),
        'vectors': vectors,
        'labels': labels,
        'index': index,
    }


def feature_matrices(query, library, graph_people, distance, k, boundary):
    """Baseline matrix and, if requested, the same matrix plus one boundary column."""
    raw = library_disagreement(
        query['original_response_id'].to_numpy(),
        graph_people['vectors'][query['original_response_id'].map(graph_people['index']).to_numpy()],
        graph_people['labels'][query['original_response_id'].map(graph_people['index']).to_numpy()],
        library['ids'], library['vectors'], library['labels'], k,
    )
    from run_layer2 import _feature_frame
    work = query.copy()
    extra = {'prompt_distance': np.asarray(distance, dtype=np.float64), 'raw_neighbor_disagreement': raw}
    baseline = _feature_frame(work, extra)
    if boundary is None:
        return baseline, None, None
    spectrum = graph_spectrum(graph_people['vectors'], k)
    if boundary == 'spectral_neighbor_disagreement':
        space = unit_rows(spectrum['eigenvectors'])
        column_values = library_disagreement(
            query['original_response_id'].to_numpy(),
            space[query['original_response_id'].map(graph_people['index']).to_numpy()],
            graph_people['labels'][query['original_response_id'].map(graph_people['index']).to_numpy()],
            library['ids'], space[library['frame']['original_response_id'].map(graph_people['index']).to_numpy()],
            library['labels'], k,
        )
    else:
        scores = spectral_scores(spectrum['eigenvalues'], spectrum['eigenvectors'], spectrum['eigengap_index'])
        located = query['original_response_id'].map(graph_people['index']).to_numpy()
        column_values = scores[boundary][located]
    column = pd.DataFrame({boundary: column_values}, index=baseline.index)
    full = pd.concat([baseline, column], axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    return baseline, full, spectrum


def _oof_delta(paired, distance, k, boundary, n_boot, seed):
    people = _people(paired)
    baseline, full, spectrum = feature_matrices(paired, people, people, distance, k, boundary)
    labels = paired['refusal_flip'].astype(int).to_numpy()
    groups = paired['group_id'].to_numpy()
    base_score = oof_predictions(baseline, labels, groups)
    full_score = oof_predictions(full, labels, groups) if full is not None else base_score
    block = metric_block(labels, base_score, full_score, groups, n_boot, seed)
    return block, base_score, full_score, spectrum


def select_boundary(paired, distance, k_values, n_boot, seed):
    """Choose k and the boundary column by out-of-fold PR-AUC. Caller decides which rows."""
    best = None
    for k in k_values:
        for name in BOUNDARY_NAMES:
            block, _, _, _ = _oof_delta(paired, distance, k, name, n_boot, seed)
            if best is None or block['full_pr_auc'] > best[0]:
                best = (block['full_pr_auc'], k, name)
    return best[1], best[2], best[0]


def score_model(paired, distance, k_values, n_boot, seed):
    """Return validation scores for one model and the protocol name."""
    paired = paired.reset_index(drop=True)
    distance = np.asarray(distance, dtype=np.float64)
    development = paired['split'].eq('development')
    validation = paired['split'].eq('validation') if 'validation' in set(paired['split']) else ~development
    if int(development.sum()) >= 40 and int(validation.sum()) >= 15:
        dev = paired.loc[development].reset_index(drop=True)
        val = paired.loc[validation].reset_index(drop=True)
        dev_distance = distance[development.to_numpy()]
        val_distance = distance[validation.to_numpy()]
        chosen_k, chosen_name, selection_pr = select_boundary(dev, dev_distance, k_values, n_boot, seed)
        graph_rows = pd.concat([dev, val], ignore_index=True)
        graph_people = _people(graph_rows)
        library = _people(dev)
        base_dev, full_dev, _ = feature_matrices(dev, library, _people(dev), dev_distance, chosen_k, chosen_name)
        base_val, full_val, spectrum = feature_matrices(
            val, library, graph_people, val_distance, chosen_k, chosen_name,
        )
        # Column sets can differ if a category is missing from one split.
        full_dev, full_val = _align(full_dev, full_val)
        base_dev, base_val = _align(base_dev, base_val)
        labels = dev['refusal_flip'].astype(int).to_numpy()
        base_score = fit_predict(base_dev, labels, base_val)
        full_score = fit_predict(full_dev, labels, full_val)
        block = metric_block(val['refusal_flip'].astype(int), base_score, full_score, val['group_id'], n_boot, seed)
        return {
            'protocol': 'development_then_validation',
            'chosen_k': int(chosen_k),
            'chosen_boundary': chosen_name,
            'selection_pr_auc': float(selection_pr),
            'metrics': block,
            'baseline_score': base_score,
            'full_score': full_score,
            'scored': val,
            'spectrum': spectrum,
            'graph_people': graph_people,
        }
    scored = paired.loc[validation].reset_index(drop=True) if validation.any() else paired
    scored_distance = distance[validation.to_numpy()] if validation.any() else distance
    best = None
    chosen = None
    spectrum = None
    for k in k_values:
        for name in BOUNDARY_NAMES:
            block, base_score, full_score, spec = _oof_delta(scored, scored_distance, k, name, n_boot, seed)
            if best is None or block['full_pr_auc'] > best['metrics']['full_pr_auc']:
                best = {
                    'protocol': 'group_kfold_validation',
                    'chosen_k': int(k),
                    'chosen_boundary': name,
                    'selection_pr_auc': float(block['full_pr_auc']),
                    'metrics': block,
                    'baseline_score': base_score,
                    'full_score': full_score,
                    'scored': scored,
                    'spectrum': spec,
                    'graph_people': _people(scored),
                }
    return best


def _align(left, right):
    columns = sorted(set(left.columns) | set(right.columns))
    def pad(frame):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            extra = pd.DataFrame(0.0, index=frame.index, columns=missing)
            frame = pd.concat([frame, extra], axis=1)
        return frame[columns]
    return pad(left), pad(right)


def variant_grams(texts, n=8):
    import re
    import unicodedata
    collected = []
    for text in texts:
        cleaned = unicodedata.normalize('NFKC', str(text)).lower()
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        words = cleaned.split()
        if len(words) < n:
            collected.append(set())
        else:
            collected.append({tuple(words[i:i + n]) for i in range(len(words) - n + 1)})
    return collected


def template_keep_mask(scored, training):
    """Keep pairs whose variant shares no word 8-gram with another group's training variant."""
    blocked = {}
    for group, grams in zip(training['group_id'], variant_grams(training['variant_prompt_text'])):
        blocked.setdefault(group, set()).update(grams)
    all_grams = set()
    for grams in blocked.values():
        all_grams |= grams
    keep = []
    query_grams = variant_grams(scored['variant_prompt_text'])
    for group, grams in zip(scored['group_id'], query_grams):
        others = all_grams - blocked.get(group, set())
        keep.append(not grams or grams.isdisjoint(others))
    return np.asarray(keep, dtype=bool)


def pooled_metrics(model_results, n_boot, seed):
    frames = []
    for result in model_results:
        scored = result['scored'].copy()
        scored['baseline_score'] = result['baseline_score']
        scored['full_score'] = result['full_score']
        frames.append(scored)
    stacked = pd.concat(frames, ignore_index=True)
    block = metric_block(
        stacked['refusal_flip'].astype(int),
        stacked['baseline_score'].to_numpy(),
        stacked['full_score'].to_numpy(),
        stacked['group_id'].to_numpy(),
        n_boot,
        seed,
    )
    block['n_pairs'] = int(len(stacked))
    block['n_models'] = int(stacked['model_id'].nunique())
    return block
