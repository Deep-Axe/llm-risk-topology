"""Relation-aware E_q scenarios: does treating copies differently change who is most exposed?

Scenarios on the declared lineage graph, with relation = declared type, else the inferred candidate type from
python/relation_typing.py output, else 'other':
  uniform      every edge passes a defect with probability q (the reported E_q)
  copies_pass  quantization edges always pass it (same weights), other edges with probability q
  trained_only quantization edges removed: only trained or merged derivatives count
Reached at q exactly when the path bottleneck draw is below q; a copy edge contributes draw 0.

Run: python python/exposure_scenarios.py --edges EDGES.jsonl --annotations ANNOTATIONS.jsonl --output DIR
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

from compare_exposure import spearman
from exposure import build, bottlenecks, edge_uniform

Q = 0.3


def relations(edges_path, annotations_path):
    declared = {}
    with Path(edges_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            declared.setdefault((row['parent_model_id'], row['child_model_id']), set()).add(row['relation'])
    inferred = {}
    with Path(annotations_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('inferred_relation'):
                inferred[(row['parent_model_id'], row['child_model_id'])] = row['inferred_relation']
    result = {}
    for edge, kinds in declared.items():
        typed = sorted(kinds - {'other'})
        result[edge] = typed[0] if typed else inferred.get(edge, 'other')
    return result


def exposure_at_q(edges, weight, replicates, seed, q=Q):
    children, order = build(edges)
    totals = Counter()
    for r in range(replicates):
        values = bottlenecks(children, order, lambda p, c, r=r: weight(p, c, r))
        for node, best in values.items():
            totals[node] += bisect.bisect_left(sorted(best.values()), q)
    reach = {v: len(d) for v, d in bottlenecks(children, order, lambda p, c: 0.0).items()}
    return {v: totals[v] / replicates for v in reach}, reach


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    kind = relations(args.edges, args.annotations)
    edges = sorted(kind)
    uniform = lambda p, c, r: edge_uniform(args.seed, r, p, c)
    copies = lambda p, c, r: 0.0 if kind[(p, c)] == 'quantization' else edge_uniform(args.seed, r, p, c)
    trained = [e for e in edges if kind[e] != 'quantization']
    scenarios = {'uniform': exposure_at_q(edges, uniform, args.replicates, args.seed),
                 'copies_pass': exposure_at_q(edges, copies, args.replicates, args.seed),
                 'trained_only': exposure_at_q(trained, uniform, args.replicates, args.seed)}
    base, reach = scenarios['uniform']
    nodes = sorted(m for m in base if reach[m] > 0)
    top_base = sorted(nodes, key=lambda m: (-base[m], m))[:args.top]
    summary = {'q': Q, 'replicates': args.replicates, 'edges': len(edges), 'relation_counts': dict(Counter(kind.values())),
               'trained_only_edges': len(trained), 'scenarios': {}}
    for name, (values, _) in scenarios.items():
        aligned = [values.get(m, 0.0) for m in nodes]
        top = sorted(nodes, key=lambda m: (-values.get(m, 0.0), m))[:args.top]
        summary['scenarios'][name] = {
            'total_expected_reach': round(sum(values.values()), 1),
            'spearman_vs_uniform': spearman([base[m] for m in nodes], aligned),
            f'top{args.top}_overlap_with_uniform': len(set(top) & set(top_base)),
            'top10': [{'model_id': m, 'e_q': round(values.get(m, 0.0), 1), 'reach': reach[m]} for m in top[:10]]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'scenarios.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'scenarios'}, indent=2))
    for name, block in summary['scenarios'].items():
        print(name, {k: v for k, v in block.items() if k != 'top10'})
        print('   top5:', [(t['model_id'], t['e_q']) for t in block['top10'][:5]])


if __name__ == '__main__':
    main()
