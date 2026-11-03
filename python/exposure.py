"""Expected descendant exposure E_q on the declared lineage graph (reference implementation).

Scenario: a defect present in a parent passes to each declared child independently with probability q.
E_q(v) is the expected number of distinct descendants of v that receive it (v itself excluded). q=1 gives
deterministic descendant reach. Each edge's draw comes from a hash of (seed, replicate, parent, child),
so the Spark job (spark/scala/exposure.scala) samples the same edges and must give identical results.

Run: python python/exposure.py --edges EDGES.jsonl --output OUT.csv [--replicates 200] [--seed 42]
EDGES.jsonl holds metadata_edges contract rows; --snapshot DIR reads a collected snapshot instead.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

Q_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
RELATIONS = ('fine_tune', 'adapter', 'merge', 'quantization', 'other')


def edge_uniform(seed, replicate, parent, child):
    """Uniform [0, 1) value from the first 52 bits of sha256('seed|replicate|parent|child')."""
    text = f'{seed}|{replicate}|{parent}|{child}'
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:13], 16) / 2**52


def read_edges(path, relations=RELATIONS):
    pairs = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row['relation'] in relations:
                    pairs.add((row['parent_model_id'], row['child_model_id']))
    return sorted(pairs)


def snapshot_edges(directory, relations=RELATIONS):
    from extract_lineage import extract

    pairs = set()
    with (Path(directory) / 'models.tsv').open(encoding='utf-8') as stream:
        for line in stream:
            child, _, payload = line.rstrip('\n').split('\t', 2)
            for status, _, parent, relation, *_ in extract(child, json.loads(payload)):
                if status == 'valid' and relation in relations:
                    pairs.add((parent, child))
    return sorted(pairs)


def topological_children_first(children):
    """Order nodes so every child comes before its parents; raises on a cycle."""
    nodes = set(children) | {c for kids in children.values() for c in kids}
    pending = {v: len(children.get(v, ())) for v in nodes}
    parents = defaultdict(list)
    for parent, kids in children.items():
        for child in kids:
            parents[child].append(parent)
    ready = sorted(v for v, n in pending.items() if n == 0)
    order = []
    while ready:
        node = ready.pop()
        order.append(node)
        for parent in parents[node]:
            pending[parent] -= 1
            if pending[parent] == 0:
                ready.append(parent)
    if len(order) != len(nodes):
        raise ValueError('Lineage graph has a cycle; E_q is defined here for acyclic graphs only')
    return order


def build(edges):
    children = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    return children, topological_children_first(children)


def bottlenecks(children, order, weight):
    """For every node, map each descendant to the smallest possible largest edge weight over paths.

    A descendant is reached at probability q exactly when this value is below q.
    """
    result = {}
    for node in order:
        best = {}
        for child in children.get(node, ()):
            w = weight(node, child)
            if w < best.get(child, 2.0):
                best[child] = w
            for desc, b in result[child].items():
                value = max(w, b)
                if value < best.get(desc, 2.0):
                    best[desc] = value
        result[node] = best
    return result


def descendant_reach(children, order):
    return {v: len(d) for v, d in bottlenecks(children, order, lambda p, c: 0.0).items()}


def tree_exact(children, order, q_grid=Q_GRID):
    """Exact E_q where every descendant has exactly one path from the node; None elsewhere."""
    paths = {}
    depth_counts = {}
    for node in order:
        count = defaultdict(int)
        depths = defaultdict(int)
        for child in children.get(node, ()):
            count[child] += 1
            depths[1] += 1
            for desc, n in paths[child].items():
                count[desc] += n
            for d, n in depth_counts[child].items():
                depths[d + 1] += n
        paths[node] = count
        depth_counts[node] = depths
    exact = {}
    for node in order:
        if any(n > 1 for n in paths[node].values()):
            exact[node] = None
        else:
            exact[node] = [sum(n * q**d for d, n in depth_counts[node].items()) for q in q_grid]
    return exact


def exposure(edges, replicates=200, seed=42, q_grid=Q_GRID):
    """Monte Carlo E_q per node: mean and standard error over replicates, plus reach and tree-exact values."""
    children, order = build(edges)
    reach = descendant_reach(children, order)
    total = {v: [0.0] * len(q_grid) for v in order}
    square = {v: [0.0] * len(q_grid) for v in order}
    for r in range(replicates):
        values = bottlenecks(children, order, lambda p, c, r=r: edge_uniform(seed, r, p, c))
        for node, best in values.items():
            ranked = sorted(best.values())
            for i, q in enumerate(q_grid):
                n = bisect.bisect_left(ranked, q)
                total[node][i] += n
                square[node][i] += n * n
    exact = tree_exact(children, order, q_grid)
    rows = []
    for node in sorted(order):
        for i, q in enumerate(q_grid):
            mean = total[node][i] / replicates
            var = max(square[node][i] / replicates - mean * mean, 0.0)
            se = math.sqrt(var / (replicates - 1)) if replicates > 1 else None
            rows.append({'model_id': node, 'q': q, 'reach': reach[node], 'e_q_mean': mean, 'e_q_se': se,
                         'e_q_tree_exact': None if exact[node] is None else exact[node][i]})
    return rows


def brute_force(edges, q):
    """Exact E_q by enumerating every subset of edges. Small test graphs only."""
    if len(edges) > 16:
        raise ValueError('Brute force is limited to 16 edges')
    nodes = sorted({v for e in edges for v in e})
    result = dict.fromkeys(nodes, 0.0)
    for kept in itertools.product([False, True], repeat=len(edges)):
        prob = math.prod(q if k else 1 - q for k in kept)
        if prob == 0:
            continue
        open_edges = [e for e, k in zip(edges, kept) if k]
        children, order = build(open_edges) if open_edges else ({}, nodes)
        reach = descendant_reach(children, order) if open_edges else {}
        for node in nodes:
            result[node] += prob * reach.get(node, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--edges', type=Path, help='metadata_edges contract rows as JSONL')
    source.add_argument('--snapshot', type=Path, help='collected snapshot directory with models.tsv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--relations', default=','.join(RELATIONS))
    args = parser.parse_args()
    relations = tuple(args.relations.split(','))
    if args.replicates < 2 or any(r not in RELATIONS for r in relations):
        parser.error('Need at least 2 replicates and known relation names')
    edges = read_edges(args.edges, relations) if args.edges else snapshot_edges(args.snapshot, relations)
    rows = exposure(edges, args.replicates, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(edges)} edges, {len(rows) // len(Q_GRID)} nodes with edges, {args.replicates} replicates')


if __name__ == '__main__':
    main()
