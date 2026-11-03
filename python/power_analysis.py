"""Power of the test-set ΔPR-AUC comparison for refusal flips.

A baseline score and an added score are noisy readings of one latent trait.
Logistic regression is fit on a development split. ΔPR-AUC and a prompt-group
bootstrap interval are computed on the test split. Later predictor code can
import `average_precision` and `delta_pr_auc_interval`.

Usage:
    python python/power_analysis.py --output data/processed/analysis/threshold
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

LOW_WITHIN = 0.15
LOW_CROSS = 0.35
HIGH_WITHIN = 0.45
HIGH_CROSS = 0.75
# Latent correlations, not correlations of the binary labels.
# Cross-checkpoint correlation is the same pair on two models.
# Within-group correlation is two pairs in one group.
# The same pair is more dependent across models than two pairs are.


def average_precision(y, scores):
    """Mean precision at each positive label, after sorting by score."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind='mergesort')
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    if n_pos <= 0.0:
        return np.nan
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y_sorted.size + 1, dtype=np.float64)
    return float(precision[y_sorted > 0.0].sum() / n_pos)


def index_groups(group_ids):
    """Row positions of each group. group_ids are arbitrary hashable ints."""
    group_ids = np.asarray(group_ids)
    order = np.argsort(group_ids, kind='mergesort')
    sorted_ids = group_ids[order]
    if sorted_ids.size == 0:
        return np.array([], dtype=int), order, np.array([], dtype=int), np.array([], dtype=int)
    change = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, sorted_ids.size]
    ids = sorted_ids[starts]
    return ids, order, starts, ends


def resample_group_rows(group_ids, rng):
    """Sample groups with replacement and keep every row of each draw."""
    ids, order, starts, ends = index_groups(group_ids)
    pick = rng.integers(0, ids.size, size=ids.size)
    chunks = [order[starts[i]:ends[i]] for i in pick]
    return np.concatenate(chunks), ids[pick]


def _percentile(values, q):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return np.nan
    pos = (values.size - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return float(values[lo] * (1.0 - weight) + values[hi] * weight)


def delta_pr_auc_interval(y, score_base, score_full, group_ids, n_boot, rng, alpha=0.05):
    """Point ΔPR-AUC and an equal-tailed group bootstrap interval.

    The scores are already the test-split predictions. Groups are resampled;
    both scores use the same draws. The model is not refit.
    """
    point_base = average_precision(y, score_base)
    point_full = average_precision(y, score_full)
    point = point_full - point_base
    ids, order, starts, ends = index_groups(group_ids)
    boots = np.empty(n_boot, dtype=np.float64)
    pick = rng.integers(0, ids.size, size=(n_boot, ids.size))
    for b in range(n_boot):
        chunks = [order[starts[i]:ends[i]] for i in pick[b]]
        idx = np.concatenate(chunks)
        base = average_precision(y[idx], score_base[idx])
        full = average_precision(y[idx], score_full[idx])
        boots[b] = full - base
    valid = boots[np.isfinite(boots)]
    lower = _percentile(valid, alpha / 2.0)
    upper = _percentile(valid, 1.0 - alpha / 2.0)
    return float(point), float(lower), float(upper)


def load_pair_structure(pilot_dir):
    """Pairs-per-group sizes for groups that contain a linked pair."""
    pilot_dir = Path(pilot_dir)
    prompts = [
        json.loads(line)
        for line in (pilot_dir / 'prompts.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    pairs = [
        json.loads(line)
        for line in (pilot_dir / 'pairs.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    bundle = json.loads((pilot_dir / 'experiment.bundle.json').read_text(encoding='utf-8'))
    splits = bundle['prompt_manifests'][0]['group_splits']
    group_of = {row['prompt_id']: row['prompt_group_id'] for row in prompts}
    counts = {}
    split_of = {}
    for pair in pairs:
        group = group_of[pair['original_prompt_id']]
        other = group_of[pair['variant_prompt_id']]
        if group != other:
            raise ValueError('Pair endpoints are in different groups')
        counts[group] = counts.get(group, 0) + 1
        split_of[group] = splits[group]
    all_sizes = np.array(sorted(counts.values()), dtype=int)
    test_sizes = np.array(
        sorted(n for group, n in counts.items() if split_of[group] == 'test'),
        dtype=int,
    )
    dev_groups = sum(1 for group in counts if split_of[group] == 'development')
    return {
        'all_sizes': all_sizes,
        'test_sizes': test_sizes,
        'dev_groups': int(dev_groups),
        'test_groups': int(test_sizes.size),
        'dev_per_test': float(dev_groups) / float(test_sizes.size),
    }


def _latent_bank(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    e0 = rng.normal(size=n)
    ea = rng.normal(size=n)
    return z, e0, ea


def _combined_scores(z, e0, ea, r0, ra):
    s0 = r0 * z + np.sqrt(1.0 - r0 * r0) * e0
    if ra <= 0.0:
        return s0, s0.copy()
    sadd = ra * z + np.sqrt(1.0 - ra * ra) * ea
    cov = r0 * ra
    denom = 1.0 - cov * cov
    w0 = (r0 - cov * ra) / denom
    wa = (ra - cov * r0) / denom
    return s0, w0 * s0 + wa * sadd


def correlation_for_auroc(prevalence, target, n=250000, seed=1103):
    """Smallest latent correlation whose marginal score reaches the AUROC."""
    from sklearn.metrics import roc_auc_score
    z, e0, ea = _latent_bank(n, seed)
    y = (z > _threshold(prevalence)).astype(np.float64)
    lo, hi = 0.0, 0.999
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        s0, _ = _combined_scores(z, e0, ea, mid, 0.0)
        auc = roc_auc_score(y, s0)
        if auc < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def added_correlation_for_delta(prevalence, r0, target_delta, n=250000, seed=1103):
    """Added-score correlation that lifts population PR-AUC by target_delta.

    Returns (correlation, achieved delta, feasible). The combined score is the
    linear score with the same weights as the conditional mean of the latent.
    """
    if target_delta <= 0.0:
        return 0.0, 0.0, True
    z, e0, ea = _latent_bank(n, seed)
    y = (z > _threshold(prevalence)).astype(np.float64)
    s0, _ = _combined_scores(z, e0, ea, r0, 0.0)
    base = average_precision(y, s0)
    lo, hi = 0.0, 0.995
    achieved = 0.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        _, s1 = _combined_scores(z, e0, ea, r0, mid)
        achieved = average_precision(y, s1) - base
        if achieved < target_delta:
            lo = mid
        else:
            hi = mid
    ra = 0.5 * (lo + hi)
    _, s1 = _combined_scores(z, e0, ea, r0, ra)
    achieved = float(average_precision(y, s1) - base)
    return float(ra), achieved, bool(achieved + 1e-3 >= target_delta)


def _threshold(prevalence):
    from statistics import NormalDist
    probability = min(max(float(prevalence), 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(1.0 - probability)


def calibrate(prevalence, baseline_auroc, true_delta, n=250000, seed=1103, r0=None):
    if r0 is None:
        r0 = correlation_for_auroc(prevalence, baseline_auroc, n=n, seed=seed)
    ra, achieved, feasible = added_correlation_for_delta(
        prevalence, r0, true_delta, n=n, seed=seed,
    )
    from sklearn.metrics import roc_auc_score
    z, e0, ea = _latent_bank(n, seed)
    y = (z > _threshold(prevalence)).astype(np.float64)
    s0, s1 = _combined_scores(z, e0, ea, r0, ra)
    return {
        'r0': float(r0),
        'ra': float(ra),
        'achieved_delta': float(achieved),
        'feasible': feasible,
        'measured_auroc': float(roc_auc_score(y, s0)),
        'baseline_pr_auc': float(average_precision(y, s0)),
        'full_pr_auc': float(average_precision(y, s1)),
    }


def _sample_sizes(pool, n_groups, rng, exact):
    if exact and n_groups == pool.size:
        return pool.copy()
    draw = rng.integers(0, pool.size, size=n_groups)
    return pool[draw]


def draw_split(group_sizes, n_checkpoints, prevalence, rho_within, rho_cross, r0, ra, rng):
    """One split: rows are (checkpoint, pair), nested in prompt groups."""
    sizes = np.asarray(group_sizes, dtype=int)
    n_groups = int(sizes.size)
    n_pairs = int(sizes.sum())
    pair_group = np.repeat(np.arange(n_groups, dtype=int), sizes)
    group_factor = rng.normal(size=n_groups)
    pair_factor = rng.normal(size=n_pairs)
    checkpoint_noise = rng.normal(size=(n_checkpoints, n_pairs))
    shared = (
        np.sqrt(rho_within) * group_factor[pair_group]
        + np.sqrt(rho_cross - rho_within) * pair_factor
    )
    latent = shared + np.sqrt(1.0 - rho_cross) * checkpoint_noise
    y = (latent > _threshold(prevalence)).astype(np.float64)
    e0 = rng.normal(size=(n_checkpoints, n_pairs))
    ea = rng.normal(size=(n_checkpoints, n_pairs))
    s0 = r0 * latent + np.sqrt(1.0 - r0 * r0) * e0
    sadd = ra * latent + np.sqrt(1.0 - ra * ra) * ea
    return {
        'y': y.ravel(),
        's0': s0.ravel(),
        'sadd': sadd.ravel(),
        'group_ids': np.tile(pair_group, n_checkpoints),
        'checkpoint': np.repeat(np.arange(n_checkpoints, dtype=int), n_pairs),
    }


def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _fit_probabilities(x_dev, y_dev, x_test):
    """Unpenalized logistic regression by Newton steps. Returns P(y=1)."""
    x_dev = np.asarray(x_dev, dtype=np.float64)
    y = np.asarray(y_dev, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        rate = float(y.mean()) if y.size else 0.0
        return np.full(x_test.shape[0], rate)
    design = np.column_stack([np.ones(x_dev.shape[0]), x_dev])
    beta = np.zeros(design.shape[1])
    for _ in range(25):
        prob = _sigmoid(design @ beta)
        weight = np.maximum(prob * (1.0 - prob), 1e-6)
        working = design @ beta + (y - prob) / weight
        weighted = design * weight[:, None]
        try:
            step = np.linalg.solve(design.T @ weighted, weighted.T @ working)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(step - beta)) < 1e-8:
            beta = step
            break
        beta = step
    test_design = np.column_stack([np.ones(x_test.shape[0]), x_test])
    return _sigmoid(test_design @ beta)


def evaluate_split(dev, test, n_boot, rng):
    """Fit B0 and B1 on development and score the test split."""
    y_dev = dev['y']
    p0 = _fit_probabilities(dev['s0'].reshape(-1, 1), y_dev, test['s0'].reshape(-1, 1))
    x1_dev = np.column_stack([dev['s0'], dev['sadd']])
    x1_test = np.column_stack([test['s0'], test['sadd']])
    p1 = _fit_probabilities(x1_dev, y_dev, x1_test)
    point, lower, upper = delta_pr_auc_interval(
        test['y'], p0, p1, test['group_ids'], n_boot, rng,
    )
    n_checkpoints = int(test['checkpoint'].max()) + 1
    signs = []
    for checkpoint in range(n_checkpoints):
        mask = test['checkpoint'] == checkpoint
        if test['y'][mask].sum() <= 0.0:
            signs.append(0.0)
            continue
        delta = average_precision(test['y'][mask], p1[mask]) - average_precision(test['y'][mask], p0[mask])
        signs.append(float(delta > 0.0))
    return point, lower, upper, signs


def one_replicate(structure, n_test_groups, n_checkpoints, prevalence, rho_within, rho_cross,
                  r0, ra, n_boot, rng):
    exact = n_test_groups == structure['test_groups']
    test_sizes = _sample_sizes(structure['test_sizes'] if exact else structure['all_sizes'],
                               n_test_groups, rng, exact)
    n_dev_groups = int(round(n_test_groups * structure['dev_per_test']))
    dev_sizes = _sample_sizes(structure['all_sizes'], n_dev_groups, rng, False)
    dev = draw_split(dev_sizes, n_checkpoints, prevalence, rho_within, rho_cross, r0, ra, rng)
    test = draw_split(test_sizes, n_checkpoints, prevalence, rho_within, rho_cross, r0, ra, rng)
    return evaluate_split(dev, test, n_boot, rng)


def _empty_summary(spec, feasible):
    return {
        'test_groups': spec['test_groups'],
        'checkpoints': spec['checkpoints'],
        'prevalence': spec['prevalence'],
        'baseline_auroc': spec['baseline_auroc'],
        'true_delta': spec['true_delta'],
        'rho_within': spec['rho_within'],
        'rho_cross': spec['rho_cross'],
        'dependence': spec['dependence'],
        'cohort': spec['cohort'],
        'feasible': feasible,
        'reps': 0,
        'n_boot': spec['n_boot'],
        'power_ci': np.nan,
        'rate_point_ge_005': np.nan,
        'power_rule': np.nan,
        'mean_delta': np.nan,
        'std_delta': np.nan,
        'mean_lower': np.nan,
        'rate_ckpt_positive': np.nan,
        'rate_six_of_eight': np.nan,
    }


def run_cell(spec):
    """Monte Carlo one grid cell. `spec` includes calibrated r0 and ra."""
    if not spec['feasible']:
        return _empty_summary(spec, False)
    rng = np.random.default_rng(spec['seed'])
    points = []
    lowers = []
    ckpt_rates = []
    six = []
    structure = {
        'all_sizes': np.asarray(spec['all_sizes'], dtype=int),
        'test_sizes': np.asarray(spec['test_sizes'], dtype=int),
        'test_groups': int(spec['pilot_test_groups']),
        'dev_per_test': float(spec['dev_per_test']),
    }
    for _ in range(spec['reps']):
        point, lower, _, signs = one_replicate(
            structure, spec['test_groups'], spec['checkpoints'], spec['prevalence'],
            spec['rho_within'], spec['rho_cross'], spec['r0'], spec['ra'], spec['n_boot'], rng,
        )
        if not np.isfinite(point) or not np.isfinite(lower):
            continue
        points.append(point)
        lowers.append(lower)
        if signs:
            ckpt_rates.append(float(np.mean(signs)))
            if spec['checkpoints'] >= 8:
                six.append(float(np.sum(signs) >= 6.0))
    points = np.asarray(points, dtype=np.float64)
    lowers = np.asarray(lowers, dtype=np.float64)
    row = _empty_summary(spec, True)
    row['reps'] = int(points.size)
    if points.size == 0:
        return row
    row['power_ci'] = float(np.mean(lowers > 0.0))
    row['rate_point_ge_005'] = float(np.mean(points >= 0.05))
    row['power_rule'] = float(np.mean((points >= 0.05) & (lowers > 0.0)))
    row['mean_delta'] = float(points.mean())
    row['std_delta'] = float(points.std(ddof=1)) if points.size > 1 else 0.0
    row['mean_lower'] = float(lowers.mean())
    row['rate_ckpt_positive'] = float(np.mean(ckpt_rates)) if ckpt_rates else np.nan
    row['rate_six_of_eight'] = float(np.mean(six)) if six else np.nan
    return row


def decision_specs(structure, calibration, reps, n_boot, seed):
    """Full prevalence × size × checkpoint grid at AUROC 0.75 and low dependence.

    AUROC and high dependence are slices. The full cross of 2,250 cells is not run.
    """
    groups = [88, 150, 300, 500, 800]
    checkpoints = [2, 8, 25]
    prevalences = [0.05, 0.1, 0.2, 0.35, 0.5]
    deltas = [0.0, 0.02, 0.05, 0.08, 0.12]
    specs = []

    def add(n_groups, n_ckpt, prevalence, auroc, delta, dependence):
        if dependence == 'low':
            rho_within, rho_cross = LOW_WITHIN, LOW_CROSS
        else:
            rho_within, rho_cross = HIGH_WITHIN, HIGH_CROSS
        key = (prevalence, auroc, delta)
        cal = calibration[key]
        specs.append({
            'test_groups': n_groups,
            'checkpoints': n_ckpt,
            'prevalence': prevalence,
            'baseline_auroc': auroc,
            'true_delta': delta,
            'dependence': dependence,
            'rho_within': rho_within,
            'rho_cross': rho_cross,
            'r0': cal['r0'],
            'ra': cal['ra'],
            'feasible': cal['feasible'],
            'reps': reps,
            'n_boot': n_boot,
            'seed': seed + len(specs) * 10007,
            'all_sizes': structure['all_sizes'].tolist(),
            'test_sizes': structure['test_sizes'].tolist(),
            'pilot_test_groups': structure['test_groups'],
            'dev_per_test': structure['dev_per_test'],
            'cohort': 'pilot_v4',
        })

    for n_groups in groups:
        for n_ckpt in checkpoints:
            for prevalence in prevalences:
                for delta in deltas:
                    add(n_groups, n_ckpt, prevalence, 0.75, delta, 'low')
    for n_groups in groups:
        for prevalence in (0.1, 0.2):
            for delta in deltas:
                add(n_groups, 8, prevalence, 0.75, delta, 'high')
    for auroc in (0.60, 0.85):
        for n_groups in (88, 300):
            for prevalence in (0.1, 0.2):
                for delta in deltas:
                    add(n_groups, 8, prevalence, auroc, delta, 'low')
    return specs


def exact_cohort_specs(structure, calibration, reps, n_boot, seed, cohort):
    """One test-set size: that cohort's real test groups, low dependence, AUROC 0.75."""
    specs = []
    for n_ckpt in (2, 8, 25):
        for prevalence in (0.05, 0.1, 0.2, 0.35, 0.5):
            for delta in (0.0, 0.02, 0.05, 0.08, 0.12):
                cal = calibration[(prevalence, 0.75, delta)]
                specs.append({
                    'test_groups': structure['test_groups'],
                    'checkpoints': n_ckpt,
                    'prevalence': prevalence,
                    'baseline_auroc': 0.75,
                    'true_delta': delta,
                    'dependence': 'low',
                    'rho_within': LOW_WITHIN,
                    'rho_cross': LOW_CROSS,
                    'r0': cal['r0'],
                    'ra': cal['ra'],
                    'feasible': cal['feasible'],
                    'reps': reps,
                    'n_boot': n_boot,
                    'seed': seed + 500000 + len(specs) * 10007,
                    'all_sizes': structure['all_sizes'].tolist(),
                    'test_sizes': structure['test_sizes'].tolist(),
                    'pilot_test_groups': structure['test_groups'],
                    'dev_per_test': structure['dev_per_test'],
                    'cohort': cohort,
                })
    return specs


def minimum_detectable(rows, power_key='power_ci', power=0.80):
    """Smallest grid Δ with power at least `power`. None if none qualify."""
    usable = [
        row for row in rows
        if row['feasible'] and row['true_delta'] > 0.0 and np.isfinite(row[power_key])
    ]
    if not usable:
        return None
    keys = {
        (row['cohort'], row['test_groups'], row['checkpoints'], row['prevalence'],
         row['baseline_auroc'], row['dependence'])
        for row in usable
    }
    found = []
    for key in sorted(keys):
        cells = [row for row in usable if (
            row['cohort'], row['test_groups'], row['checkpoints'], row['prevalence'],
            row['baseline_auroc'], row['dependence'],
        ) == key]
        cells.sort(key=lambda row: row['true_delta'])
        hit = next((row['true_delta'] for row in cells if row[power_key] >= power), None)
        found.append({
            'cohort': key[0],
            'test_groups': key[1],
            'checkpoints': key[2],
            'prevalence': key[3],
            'baseline_auroc': key[4],
            'dependence': key[5],
            'mdd': hit,
            'power_key': power_key,
        })
    return found


def operating_characteristics(prevalence, r0, ra, n=300000, seed=7):
    """Precision at recall 0.5, and true flips in the top 10% of rows."""
    z, e0, ea = _latent_bank(n, seed)
    y = (z > _threshold(prevalence)).astype(np.float64)
    s0, s1 = _combined_scores(z, e0, ea, r0, ra)

    def at_recall(scores, recall=0.5):
        order = np.argsort(-scores, kind='mergesort')
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        target = recall * y_sorted.sum()
        idx = int(np.searchsorted(tp, target, side='left'))
        idx = min(idx, y_sorted.size - 1)
        return float(tp[idx] / float(idx + 1))

    def top_rate(scores, frac=0.1):
        order = np.argsort(-scores, kind='mergesort')
        k = max(1, int(round(frac * y.size)))
        return float(y[order[:k]].sum() / k)

    base_top = top_rate(s0)
    full_top = top_rate(s1)
    return {
        'prevalence': prevalence,
        'precision_at_recall_0.5_base': at_recall(s0),
        'precision_at_recall_0.5_full': at_recall(s1),
        'top10_precision_base': base_top,
        'top10_precision_full': full_top,
        'extra_true_flips_per_1000': (full_top - base_top) * 100.0,
    }


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_mdd(path, mdd_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for axis, prevalence in zip(axes, (0.1, 0.2)):
        for checkpoints, color in ((2, 'C0'), (8, 'C1'), (25, 'C2')):
            subset = [
                row for row in mdd_rows
                if row['power_key'] == 'power_ci'
                and row['dependence'] == 'low'
                and row['baseline_auroc'] == 0.75
                and row['prevalence'] == prevalence
                and row['checkpoints'] == checkpoints
                and row['test_groups'] in (88, 150, 300, 500, 800)
            ]
            subset.sort(key=lambda row: row['test_groups'])
            xs = [row['test_groups'] for row in subset]
            ys = [np.nan if row['mdd'] is None else row['mdd'] for row in subset]
            axis.plot(xs, ys, marker='o', color=color, label='%d checkpoints' % checkpoints)
        axis.axhline(0.05, color='0.4', linewidth=0.8, linestyle='--')
        axis.set_title('Prevalence %.2f' % prevalence)
        axis.set_xlabel('Test pair groups')
        axis.set_ylabel('Smallest grid Δ with 80% CI power')
        axis.set_xticks([88, 150, 300, 500, 800])
        axis.set_ylim(0.0, 0.14)
    axes[0].legend(frameon=False)
    fig.suptitle('Baseline AUROC 0.75, low within-group dependence')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_calibration(seed):
    prevalences = [0.05, 0.1, 0.2, 0.35, 0.5]
    aurocs = [0.60, 0.75, 0.85]
    deltas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12]
    table = {}
    r0_cache = {}
    for prevalence in prevalences:
        for auroc in aurocs:
            r0_cache[(prevalence, auroc)] = correlation_for_auroc(prevalence, auroc, seed=seed)
            for delta in deltas:
                table[(prevalence, auroc, delta)] = calibrate(
                    prevalence, auroc, delta, seed=seed, r0=r0_cache[(prevalence, auroc)],
                )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('data/processed/prompts/pilot_v4'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/analysis/threshold'))
    parser.add_argument('--reps', type=int, default=200)
    parser.add_argument('--boot', type=int, default=199)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    structure = load_pair_structure(args.pilot)
    calibration = build_calibration(args.seed)
    specs = decision_specs(structure, calibration, args.reps, args.boot, args.seed)
    extra = args.pilot.parent / 'pilot_v5'
    if args.pilot.name == 'pilot_v4' and (extra / 'pairs.jsonl').exists():
        specs.extend(exact_cohort_specs(
            load_pair_structure(extra), calibration, args.reps, args.boot, args.seed, 'pilot_v5',
        ))
    args.output.mkdir(parents=True, exist_ok=True)
    cal_rows = []
    for (prevalence, auroc, delta), row in sorted(calibration.items()):
        item = {
            'prevalence': prevalence,
            'target_auroc': auroc,
            'true_delta': delta,
            'r0': row['r0'],
            'ra': row['ra'],
            'achieved_delta': row['achieved_delta'],
            'feasible': row['feasible'],
            'measured_auroc': row['measured_auroc'],
            'baseline_pr_auc': row['baseline_pr_auc'],
            'full_pr_auc': row['full_pr_auc'],
        }
        cal_rows.append(item)
    _write_csv(
        args.output / 'calibration.csv',
        cal_rows,
        ['prevalence', 'target_auroc', 'true_delta', 'r0', 'ra', 'achieved_delta', 'feasible',
         'measured_auroc', 'baseline_pr_auc', 'full_pr_auc'],
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print('cells scheduled %d' % len(specs), flush=True)
    results = []
    fields = list(_empty_summary(specs[0], True).keys())
    out_csv = args.output / 'cells.csv'

    def store(row, index, total):
        results.append(row)
        if index % 10 == 0 or index == total:
            _write_csv(out_csv, results, fields)
            print('cells %d/%d' % (index, total), flush=True)

    if args.jobs == 1:
        for index, spec in enumerate(specs, start=1):
            store(run_cell(spec), index, len(specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_cell, spec) for spec in specs]
            for index, future in enumerate(as_completed(futures), start=1):
                store(future.result(), index, len(futures))
    mdd_ci = minimum_detectable(results, 'power_ci') or []
    mdd_rule = minimum_detectable(results, 'power_rule') or []
    _write_csv(
        args.output / 'minimum_detectable.csv',
        mdd_ci + mdd_rule,
        ['cohort', 'test_groups', 'checkpoints', 'prevalence', 'baseline_auroc',
         'dependence', 'mdd', 'power_key'],
    )
    op_rows = []
    for prevalence in (0.1, 0.2, 0.35):
        for delta in (0.02, 0.05, 0.10):
            cal = calibration[(prevalence, 0.75, delta)]
            if not cal['feasible']:
                continue
            item = operating_characteristics(prevalence, cal['r0'], cal['ra'], seed=args.seed)
            item['true_delta'] = delta
            item['baseline_auroc'] = 0.75
            item['baseline_pr_auc'] = cal['baseline_pr_auc']
            item['full_pr_auc'] = cal['full_pr_auc']
            op_rows.append(item)
    _write_csv(
        args.output / 'operating_points.csv',
        op_rows,
        ['prevalence', 'baseline_auroc', 'true_delta', 'baseline_pr_auc', 'full_pr_auc',
         'precision_at_recall_0.5_base', 'precision_at_recall_0.5_full',
         'top10_precision_base', 'top10_precision_full', 'extra_true_flips_per_1000'],
    )
    _plot_mdd(args.output / 'mdd_vs_groups.png', mdd_ci)
    design = {
        'reps': args.reps,
        'n_boot': args.boot,
        'seed': args.seed,
        'pilot_test_groups': structure['test_groups'],
        'pilot_dev_groups': structure['dev_groups'],
        'dev_per_test': structure['dev_per_test'],
        'pair_group_size_counts': {
            str(int(size)): int(np.sum(structure['all_sizes'] == size))
            for size in np.unique(structure['all_sizes'])
        },
        'dependence': {
            'low': {'rho_within': LOW_WITHIN, 'rho_cross': LOW_CROSS},
            'high': {'rho_within': HIGH_WITHIN, 'rho_cross': HIGH_CROSS},
        },
        'cells': len(specs),
        'monte_carlo_se_at_power_0.8': float(np.sqrt(0.16 / args.reps)),
    }
    (args.output / 'design.json').write_text(json.dumps(design, indent=2) + '\n', encoding='utf-8')
    print('wrote %s' % args.output)


if __name__ == '__main__':
    main()
