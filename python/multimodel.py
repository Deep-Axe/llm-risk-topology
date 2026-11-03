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
DEVELOPMENT_MODELS = (
    'mistralai/Mistral-7B-Instruct-v0.1',
    'HuggingFaceH4/zephyr-7b-beta',
)
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


def _point_block(labels, base_score, full_score):
    from sklearn.metrics import average_precision_score
    labels = np.asarray(labels, dtype=int)
    base = float(average_precision_score(labels, base_score))
    full = float(average_precision_score(labels, full_score))
    return {
        'prevalence': float(labels.mean()) if labels.size else None,
        'baseline_pr_auc': base,
        'full_pr_auc': full,
        'delta_pr_auc': full - base,
        'delta_lower': None,
        'delta_upper': None,
    }


def _oof_delta(paired, distance, k, boundary, n_boot, seed):
    people = _people(paired)
    baseline, full, spectrum = feature_matrices(paired, people, people, distance, k, boundary)
    labels = paired['refusal_flip'].astype(int).to_numpy()
    groups = paired['group_id'].to_numpy()
    base_score = oof_predictions(baseline, labels, groups)
    full_score = oof_predictions(full, labels, groups) if full is not None else base_score
    if n_boot:
        block = metric_block(labels, base_score, full_score, groups, n_boot, seed)
    else:
        block = _point_block(labels, base_score, full_score)
    return block, base_score, full_score, spectrum


def select_boundary(paired, distance, k_values, seed):
    """Choose k and the boundary column by out-of-fold PR-AUC. No bootstrap."""
    best = None
    for k in k_values:
        for name in BOUNDARY_NAMES:
            block, _, _, _ = _oof_delta(paired, distance, k, name, 0, seed)
            if best is None or block['full_pr_auc'] > best[0]:
                best = (block['full_pr_auc'], k, name)
    return best[1], best[2], best[0]


def development_settings(prepared, k_values, seed):
    """One (k, score) from the pooled development pairs of Mistral and Zephyr."""
    pieces = []
    sources = []
    for model_id, paired, distance in prepared:
        if model_id not in DEVELOPMENT_MODELS:
            continue
        development = paired['split'].eq('development')
        if int(development.sum()) < 40:
            continue
        part = paired.loc[development].reset_index(drop=True)
        part['_distance'] = np.asarray(distance)[development.to_numpy()]
        pieces.append(part)
        sources.append(model_id)
    if not pieces:
        return None
    pooled = pd.concat(pieces, ignore_index=True)
    chosen_k, chosen_name, selection_pr = select_boundary(
        pooled.drop(columns=['_distance']), pooled['_distance'].to_numpy(), k_values, seed,
    )
    return {
        'k': int(chosen_k),
        'boundary': chosen_name,
        'selection_pr_auc': float(selection_pr),
        'models': sources,
    }


def score_model(paired, distance, k_values, n_boot, seed, fixed=None):
    """Return validation scores for one model and the protocol name.

    `fixed` is (k, boundary name). Validation-only models use it instead of reselecting.
    """
    paired = paired.reset_index(drop=True)
    distance = np.asarray(distance, dtype=np.float64)
    development = paired['split'].eq('development')
    validation = paired['split'].eq('validation') if 'validation' in set(paired['split']) else ~development
    if int(development.sum()) >= 40 and int(validation.sum()) >= 15:
        dev = paired.loc[development].reset_index(drop=True)
        val = paired.loc[validation].reset_index(drop=True)
        dev_distance = distance[development.to_numpy()]
        val_distance = distance[validation.to_numpy()]
        if fixed is None:
            chosen_k, chosen_name, selection_pr = select_boundary(dev, dev_distance, k_values, seed)
        else:
            chosen_k, chosen_name = fixed
            selection_pr = None
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
            'exploratory_selection': False,
            'chosen_k': int(chosen_k),
            'chosen_boundary': chosen_name,
            'selection_pr_auc': None if selection_pr is None else float(selection_pr),
            'metrics': block,
            'baseline_score': base_score,
            'full_score': full_score,
            'scored': val,
            'spectrum': spectrum,
            'graph_people': graph_people,
        }
    scored = paired.loc[validation].reset_index(drop=True) if validation.any() else paired
    scored_distance = distance[validation.to_numpy()] if validation.any() else distance
    if fixed is None:
        best_pr = -1.0
        saved = None
        for k in k_values:
            for name in BOUNDARY_NAMES:
                block, base_score, full_score, spec = _oof_delta(scored, scored_distance, k, name, 0, seed)
                if block['full_pr_auc'] > best_pr:
                    best_pr = block['full_pr_auc']
                    saved = (k, name, base_score, full_score, spec, block['full_pr_auc'])
        chosen_k, chosen_name, base_score, full_score, spec, selection_pr = saved
        exploratory = True
    else:
        chosen_k, chosen_name = fixed
        _, base_score, full_score, spec = _oof_delta(
            scored, scored_distance, chosen_k, chosen_name, 0, seed,
        )
        selection_pr = None
        exploratory = False
    labels = scored['refusal_flip'].astype(int).to_numpy()
    block = metric_block(labels, base_score, full_score, scored['group_id'], n_boot, seed)
    return {
        'protocol': 'group_kfold_validation',
        'exploratory_selection': exploratory,
        'chosen_k': int(chosen_k),
        'chosen_boundary': chosen_name,
        'selection_pr_auc': None if selection_pr is None else float(selection_pr),
        'metrics': block,
        'baseline_score': base_score,
        'full_score': full_score,
        'scored': scored,
        'spectrum': spec,
        'graph_people': _people(scored),
    }


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


def template_fold_mask(scored, n_splits=5):
    """Drop test-fold pairs that share an 8-gram with a training-fold variant.

    Returns None when no training fold has an 8-gram to compare against.
    """
    from sklearn.model_selection import GroupKFold
    groups = scored['group_id'].to_numpy()
    n_groups = len(pd.unique(groups))
    if n_groups < 2:
        return None
    grams = variant_grams(scored['variant_prompt_text'])
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    dummy = np.zeros(len(scored))
    keep = np.ones(len(scored), dtype=bool)
    comparable = False
    for train_at, test_at in splitter.split(dummy, groups, groups):
        blocked = set()
        for index in train_at:
            blocked |= grams[index]
        if not blocked:
            continue
        comparable = True
        for index in test_at:
            if grams[index] and not grams[index].isdisjoint(blocked):
                keep[index] = False
    if not comparable:
        return None
    return keep


def template_summary(scored, paired, baseline_score, full_score, n_boot, seed):
    """Development variants are the reference when they exist. Otherwise each CV fold is."""
    development = paired['split'].eq('development')
    if int(development.sum()) > 0:
        training = paired.loc[development, ['group_id', 'variant_prompt_text']]
        mask = template_keep_mask(scored, training)
        if mask is None or not any(variant_grams(training['variant_prompt_text'])):
            return {'status': 'n/a', 'reason': 'no development 8-grams'}
    else:
        mask = template_fold_mask(scored)
        if mask is None:
            return {'status': 'n/a', 'reason': 'no training-fold 8-grams'}
    kept = int(mask.sum())
    dropped = int((~mask).sum())
    summary = {'status': 'compared', 'n_kept': kept, 'n_dropped': dropped, 'n_pairs': kept}
    if dropped == 0:
        summary['delta_pr_auc'] = None
        summary['note'] = 'no shared templates with the training variants'
        return summary
    if kept >= 20 and scored.loc[mask, 'refusal_flip'].nunique() > 1:
        block = metric_block(
            scored.loc[mask, 'refusal_flip'].astype(int),
            baseline_score[mask], full_score[mask], scored.loc[mask, 'group_id'], n_boot, seed,
        )
        summary.update(block)
    return summary


def template_keep_mask(scored, training):
    """Keep pairs whose variant shares no word 8-gram with another group's training variant."""
    blocked = {}
    for group, grams in zip(training['group_id'], variant_grams(training['variant_prompt_text'])):
        blocked.setdefault(group, set()).update(grams)
    keep = []
    query_grams = variant_grams(scored['variant_prompt_text'])
    for group, grams in zip(scored['group_id'], query_grams):
        others = set()
        for other_group, other_grams in blocked.items():
            if other_group != group:
                others |= other_grams
        keep.append(not grams or grams.isdisjoint(others))
    return np.asarray(keep, dtype=bool)


def _tex(value):
    return str(value).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def _num(value):
    if value is None:
        return '---'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '---'
    if not np.isfinite(number):
        return '---'
    return '%.3f' % number


def _protocol_code(row):
    label = 'WG' if row['label_source'] == 'wildguard' else 'rules'
    protocol = 'dev' if row['protocol'] == 'development_then_validation' else 'cv'
    return '%s/%s' % (label, protocol)


def _flip_by(rows):
    return '; '.join('%s %s' % (row['model'].split('/')[-1], _num(row['flip_rate'])) for row in rows)


def write_tables(directory, rows, confirmatory, exploratory, n_responses):
    """Three booktabs files. The lead wraps the tables in floats."""
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    flips = [row['flip_rate'] for row in rows]
    pooled = confirmatory if confirmatory is not None else exploratory
    numbers = [
        '\\newcommand{\\LtwoPooledDelta}{%s}' % _num(None if pooled is None else pooled['delta_pr_auc']),
        '\\newcommand{\\LtwoPooledLower}{%s}' % _num(None if pooled is None else pooled['delta_lower']),
        '\\newcommand{\\LtwoPooledUpper}{%s}' % _num(None if pooled is None else pooled['delta_upper']),
        '\\newcommand{\\LtwoModels}{%d}' % len(rows),
        '\\newcommand{\\LtwoPairs}{%d}' % sum(row['n_pairs'] for row in rows),
        '\\newcommand{\\LtwoResponses}{%d}' % int(n_responses),
        '\\newcommand{\\LtwoFlipMin}{%s}' % _num(min(flips) if flips else None),
        '\\newcommand{\\LtwoFlipMax}{%s}' % _num(max(flips) if flips else None),
        '\\newcommand{\\LtwoFlipBy}{%s}' % _tex(_flip_by(rows)),
        '\\newcommand{\\LtwoPrimaryModels}{Mistral-7B-Instruct-v0.1, Zephyr-7b-beta}',
        '',
    ]
    (directory / 'layer2_numbers.tex').write_text('\n'.join(numbers), encoding='utf-8')
    lines = [
        '\\begin{tabular}{llrrrrl}',
        '\\toprule',
        'Model & Code & Pairs & Flip & B0 & B1 & $\\Delta$ (95\\% CI) \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append('%s & %s & %d & %s & %s & %s & %s [%s, %s] \\\\' % (
            _tex(row['model'].split('/')[-1]), _protocol_code(row), row['n_pairs'], _num(row['flip_rate']),
            _num(row['baseline_pr_auc']), _num(row['full_pr_auc']), _num(row['delta']),
            _num(row['lower']), _num(row['upper']),
        ))
    if confirmatory is not None:
        lines.append('\\midrule')
        lines.append('Mistral + Zephyr & dev & %d &  & %s & %s & %s [%s, %s] \\\\' % (
            confirmatory['n_pairs'], _num(confirmatory['baseline_pr_auc']), _num(confirmatory['full_pr_auc']),
            _num(confirmatory['delta_pr_auc']), _num(confirmatory['delta_lower']), _num(confirmatory['delta_upper']),
        ))
    if exploratory is not None:
        lines.append('All other models & cv & %d &  & %s & %s & %s [%s, %s] \\\\' % (
            exploratory['n_pairs'], _num(exploratory['baseline_pr_auc']), _num(exploratory['full_pr_auc']),
            _num(exploratory['delta_pr_auc']), _num(exploratory['delta_lower']), _num(exploratory['delta_upper']),
        ))
    lines.extend(['\\bottomrule', '\\end{tabular}', ''])
    (directory / 'layer2_main_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    sens = [
        '\\begin{tabular}{llrl}',
        '\\toprule',
        'Check & Model & Pairs & $\\Delta$ (95\\% CI) \\\\',
        '\\midrule',
    ]
    for row in rows:
        name = _tex(row['model'].split('/')[-1])
        for check, block in (
            ('overlap excluded', row.get('without_overlap')),
            ('other label', row.get('other_label')),
            ('template held out', row.get('template_held_out')),
        ):
            if not block:
                continue
            if block.get('status') == 'n/a':
                sens.append('%s & %s & n/a & n/a \\\\' % (check, name))
                continue
            delta = block.get('delta', block.get('delta_pr_auc'))
            lower = block.get('lower', block.get('delta_lower'))
            upper = block.get('upper', block.get('delta_upper'))
            if 'n_kept' in block:
                pair_cell = '%d kept, %d dropped' % (block['n_kept'], block['n_dropped'])
            else:
                pair_cell = str(block.get('n_pairs', '---'))
            if delta is None:
                sens.append('%s & %s & %s & n/a \\\\' % (check, name, pair_cell))
            else:
                sens.append('%s & %s & %s & %s [%s, %s] \\\\' % (
                    check, name, pair_cell, _num(delta), _num(lower), _num(upper),
                ))
        for item in row.get('k_sensitivity') or []:
            sens.append('k=%d & %s & %s & %s [%s, %s] \\\\' % (
                item['k'], name, item.get('n_pairs', '---'), _num(item['delta']),
                _num(item['lower']), _num(item['upper']),
            ))
        harm = row.get('harmful_transition')
        if harm and harm.get('metrics'):
            sens.append('harmful transition & %s & %d & %s [%s, %s] \\\\' % (
                name, harm['n_pairs'], _num(harm['metrics']['delta_pr_auc']),
                _num(harm['metrics']['delta_lower']), _num(harm['metrics']['delta_upper']),
            ))
    sens.extend(['\\bottomrule', '\\end{tabular}', ''])
    (directory / 'layer2_sensitivity_table.tex').write_text('\n'.join(sens), encoding='utf-8')
    old = directory / 'layer2_tables.tex'
    if old.exists():
        old.unlink()


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
