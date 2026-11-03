"""Cosine k-NN graph and the smallest eigenvectors of its normalized Laplacian.

Usage: imported by python/run_layer2.py. A 25-node check compares eigsh with eigh.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from embed import unit_rows


def knn_adjacency(vectors, k):
    """Symmetric union k-NN. Edge weight is cosine similarity."""
    points = unit_rows(vectors)
    n = points.shape[0]
    if n < 3:
        raise ValueError('Need at least 3 responses to build a graph')
    k = min(int(k), n - 1)
    similarity = points @ points.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = neighbors.ravel()
    weights = similarity[rows, cols]
    matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    return matrix.maximum(matrix.T)


def normalized_laplacian(adjacency):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    scaled = sparse.diags(scale) @ adjacency @ sparse.diags(scale)
    return sparse.eye(adjacency.shape[0], format='csr') - scaled


def smallest_eigenpairs(laplacian, n_components, dense=False):
    n = laplacian.shape[0]
    n_components = min(int(n_components), n - 2)
    if dense or n < 40:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        return values[:n_components], vectors[:, :n_components]
    values, vectors = eigsh(laplacian.asfptype(), k=n_components, which='SA')
    order = np.argsort(values)
    return values[order], vectors[:, order]


def graph_spectrum(vectors, k, n_components=12):
    adjacency = knn_adjacency(vectors, k)
    n_components_graph, labels = connected_components(adjacency, directed=False)
    laplacian = normalized_laplacian(adjacency)
    values, vectors = smallest_eigenpairs(laplacian, n_components)
    gaps = np.diff(values)
    gap_at = int(np.argmax(gaps)) if gaps.size else 0
    return {
        'adjacency': adjacency,
        'n_components': int(n_components_graph),
        'component_labels': labels,
        'eigenvalues': values,
        'eigenvectors': vectors,
        'eigengap': float(gaps[gap_at]) if gaps.size else 0.0,
        'eigengap_index': gap_at,
        'n_clusters': int(gap_at + 1),
    }
