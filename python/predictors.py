"""Baseline and spectral-boundary predictors for refusal flips.

Fit is logistic regression. With a development split, it is fit there and scored
on validation. With validation only, scores are out-of-fold GroupKFold predictions.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from power_analysis import delta_pr_auc_interval


def spectral_scores(eigenvalues, eigenvectors, eigengap_index):
    """Fiedler magnitude and a spectral k-means margin. Higher margin is deeper inside a cluster."""
    fiedler_at = 1 if eigenvalues.shape[0] > 1 else 0
    fiedler = np.abs(eigenvectors[:, fiedler_at])
    n_clusters = max(2, min(int(eigengap_index) + 1, eigenvectors.shape[1], eigenvectors.shape[0] - 1))
    coords = eigenvectors[:, :n_clusters]
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(coords)
    margins = np.empty(coords.shape[0], dtype=np.float64)
    centroids = np.vstack([coords[labels == cluster].mean(axis=0) for cluster in range(n_clusters)])
    for index, point in enumerate(coords):
        distance = np.linalg.norm(centroids - point, axis=1)
        own = distance[labels[index]]
        others = np.delete(distance, labels[index])
        margins[index] = float(others.min() - own) if others.size else 0.0
    return {'fiedler_magnitude': fiedler, 'spectral_margin': margins, 'cluster': labels}


def _disagreement_against(query, query_labels, library, library_labels, k):
    if library.shape[0] < 2:
        return np.zeros(query.shape[0], dtype=np.float64)
    k = min(int(k), library.shape[0] - 1 if query is library else library.shape[0])
    k = max(k, 1)
    similarity = query @ library.T
    if query is library:
        np.fill_diagonal(similarity, -np.inf)
        k = min(k, library.shape[0] - 1)
    neighbors = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    disagreed = library_labels[neighbors] != query_labels[:, None]
    return disagreed.mean(axis=1)


def design_columns(frame, numeric, categorical):
    pieces = [frame[numeric].astype(float)]
    for name in categorical:
        dummies = pd.get_dummies(frame[name].fillna('missing').astype(str), prefix=name)
        pieces.append(dummies)
    return pd.concat(pieces, axis=1)


def _fit_predict(train_x, train_y, test_x):
    if np.unique(train_y).size < 2:
        rate = float(np.mean(train_y)) if len(train_y) else 0.0
        return np.full(test_x.shape[0], rate)
    model = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(test_x)[:, 1]


def _scale(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def oof_predictions(features, labels, groups, n_splits=5):
    """Out-of-fold probabilities. `groups` keeps a prompt group inside one fold."""
    predictions = np.zeros(len(labels), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    matrix = features.to_numpy(dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    for train_at, test_at in splitter.split(matrix, labels, groups):
        train_x, test_x = _scale(matrix[train_at], matrix[test_at])
        predictions[test_at] = _fit_predict(train_x, labels[train_at], test_x)
    return predictions


def metric_block(labels, baseline_score, full_score, groups, n_boot, seed):
    labels = np.asarray(labels, dtype=int)
    point, lower, upper = delta_pr_auc_interval(
        labels.astype(float), baseline_score, full_score, np.asarray(groups), n_boot,
        np.random.default_rng(seed),
    )
    return {
        'prevalence': float(labels.mean()),
        'baseline_pr_auc': float(average_precision_score(labels, baseline_score)),
        'full_pr_auc': float(average_precision_score(labels, full_score)),
        'baseline_auroc': float(roc_auc_score(labels, baseline_score)) if np.unique(labels).size > 1 else None,
        'full_auroc': float(roc_auc_score(labels, full_score)) if np.unique(labels).size > 1 else None,
        'delta_pr_auc': point,
        'delta_lower': lower,
        'delta_upper': upper,
    }
