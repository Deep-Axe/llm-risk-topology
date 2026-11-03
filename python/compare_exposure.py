"""Compare Spark E_q output with the Python reference and summarize E_q against descendant reach.

Run: python python/compare_exposure.py --spark SPARK.csv --reference REF.csv --output SUMMARY.json
Fails if any (model, q) row differs beyond 1e-9 or is missing on either side.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return {(row['model_id'], float(row['q'])): row for row in csv.DictReader(stream)}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spark', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    spark, ref = load(args.spark), load(args.reference)
    missing = sorted(set(ref) - set(spark))
    extra = sorted(set(spark) - set(ref))
    worst = 0.0
    reach_mismatch = 0
    for key in set(ref) & set(spark):
        a, b = spark[key], ref[key]
        reach_mismatch += int(a['reach']) != int(b['reach'])
        for field in ['e_q_mean', 'e_q_se']:
            worst = max(worst, abs(float(a[field]) - float(b[field])))
    # Monte Carlo against the exact value where the node's descendants form a tree.
    z_scores = []
    for row in ref.values():
        exact, se = row['e_q_tree_exact'], float(row['e_q_se'] or 0)
        if exact not in ('', None) and se > 0:
            z_scores.append((float(row['e_q_mean']) - float(exact)) / se)
    by_q = defaultdict(list)
    for (model, q), row in ref.items():
        if int(row['reach']) > 0:
            by_q[q].append((model, int(row['reach']), float(row['e_q_mean'])))
    rankings = []
    for q in sorted(by_q):
        rows = by_q[q]
        top_reach = {m for m, _, _ in sorted(rows, key=lambda r: (-r[1], r[0]))[:args.top]}
        top_eq = {m for m, _, _ in sorted(rows, key=lambda r: (-r[2], r[0]))[:args.top]}
        rankings.append({'q': q, 'nodes_with_descendants': len(rows),
                         'spearman_e_q_vs_reach': spearman([r[1] for r in rows], [r[2] for r in rows]),
                         f'top{args.top}_overlap_with_reach': len(top_reach & top_eq),
                         f'top{args.top}_by_e_q': [m for m, _, _ in sorted(rows, key=lambda r: (-r[2], r[0]))[:5]]})
    summary = {'rows_reference': len(ref), 'rows_spark': len(spark), 'missing_in_spark': len(missing),
               'extra_in_spark': len(extra), 'reach_mismatches': reach_mismatch,
               'max_abs_difference': worst,
               'tree_exact_checks': len(z_scores),
               'tree_exact_abs_z_over_3': sum(abs(z) > 3 for z in z_scores),
               'tree_exact_mean_z': sum(z_scores) / len(z_scores) if z_scores else None,
               'rankings': rankings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rankings'}, indent=2))
    if missing or extra or reach_mismatch or worst > 1e-9:
        raise SystemExit('Spark and Python E_q disagree')
    print('EXPOSURE_COMPARISON_OK')


if __name__ == '__main__':
    main()
