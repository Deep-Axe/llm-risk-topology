"""Lineage-graph figures and numbers for the report: E_q curves, reach vs E_q, degrees, relations, keyword flags.

Run: python python/figures_layer1.py --exposure-run report/exposure/RUN --keyword-index KEYWORD_INDEX.tsv
Writes PNGs to report/figures/layer1/ and the numbers to report/figures/layer1/layer1_numbers.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from exposure import Q_GRID, build, bottlenecks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_exposure(path):
    rows = defaultdict(dict)
    with path.open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            rows[row['model_id']][float(row['q'])] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exposure-run', type=Path, required=True)
    parser.add_argument('--keyword-index', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'report/figures/layer1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    exposure = load_exposure(args.exposure_run / 'node_exposure_python.csv')
    edges, relations = [], Counter()
    with (args.exposure_run / 'edges.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            relations[row['relation']] += 1
            edges.append((row['parent_model_id'], row['child_model_id']))
    edges = sorted(set(edges))
    numbers = {'edges': len(edges), 'relations': dict(relations)}

    # E_q curves for the eight largest roots by reach.
    top = sorted(exposure, key=lambda m: (-int(exposure[m][1.0]['reach']), m))[:8]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model in top:
        means = [float(exposure[model][q]['e_q_mean']) for q in Q_GRID]
        ses = [float(exposure[model][q]['e_q_se'] or 0) for q in Q_GRID]
        ax.plot(Q_GRID, means, marker='o', ms=3, label=model)
        ax.fill_between(Q_GRID, [m - 2 * s for m, s in zip(means, ses)], [m + 2 * s for m, s in zip(means, ses)],
                        alpha=.15)
    ax.set_xlabel('Transmission probability q per lineage edge')
    ax.set_ylabel('Expected descendants reached, E_q')
    ax.set_title('Expected downstream exposure of the largest model families')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / 'eq_curves.png')
    plt.close(fig)

    # Reach vs E_q at q = 0.3.
    pairs = [(int(r[1.0]['reach']), float(r[0.3]['e_q_mean']), m) for m, r in exposure.items()
             if int(r[1.0]['reach']) > 0]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=6, alpha=.4, color='#3b6ea5', linewidths=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    by_reach = sorted(pairs, reverse=True)
    for (reach, value, model), offset in zip([by_reach[0], by_reach[1]], [(-120, 4), (-95, -14)]):
        ax.annotate(model.split('/')[-1], (reach, value), fontsize=7, xytext=offset, textcoords='offset points',
                    arrowprops={'arrowstyle': '-', 'lw': .5, 'color': 'gray'})
    ax.set_xlabel('Descendants (deterministic reach, q = 1)')
    ax.set_ylabel('E_q at q = 0.3')
    ax.set_title(f'Reach vs expected exposure, {len(pairs):,} models')
    fig.tight_layout()
    fig.savefig(args.output / 'reach_vs_eq.png')
    plt.close(fig)

    # Out-degree distribution (number of declared children).
    children = Counter(p for p, _ in edges)
    counts = Counter(children.values())
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = sorted(counts)
    ax.scatter(xs, [counts[x] for x in xs], s=10, color='#3b6ea5')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Declared children of a model')
    ax.set_ylabel('Number of models')
    ax.set_title('Heavy-tailed fan-out in declared lineage')
    fig.tight_layout()
    fig.savefig(args.output / 'children_distribution.png')
    plt.close(fig)
    numbers['max_children'] = max(children.values())
    numbers['models_with_children'] = len(children)

    # Declared relation types.
    fig, ax = plt.subplots(figsize=(5.5, 3))
    names = [r for r, _ in relations.most_common()]
    ax.barh(names, [relations[r] for r in names], color='#3b6ea5')
    ax.invert_yaxis()
    for i, r in enumerate(names):
        ax.text(relations[r], i, f' {relations[r]:,}', va='center', fontsize=8)
    ax.set_xlabel('Declared lineage edges')
    ax.set_title('Relation types declared on model cards')
    fig.tight_layout()
    fig.savefig(args.output / 'relation_types.png')
    plt.close(fig)

    numbers['top_by_reach'] = [{'model_id': m, 'reach': int(exposure[m][1.0]['reach']),
                                'e_q_0.3': float(exposure[m][0.3]['e_q_mean'])} for m in top]

    if args.keyword_index:
        flagged = defaultdict(set)
        with args.keyword_index.open(encoding='utf-8') as stream:
            for line in stream:
                keyword, model, category, _ = line.rstrip('\n').split('\t')
                flagged[category].add(model)
        numbers['keyword_models'] = {c: len(v) for c, v in flagged.items()}
        child_map, order = build(edges)
        descendants = bottlenecks(child_map, order, lambda p, c: 0.0)
        rows = []
        for model in sorted(exposure, key=lambda m: (-int(exposure[m][1.0]['reach']), m))[:15]:
            desc = descendants.get(model, {})
            removed = sum(d in flagged['safety_removed'] for d in desc)
            quantized = sum(d in flagged['quantized'] for d in desc)
            rows.append({'model_id': model, 'descendants': len(desc), 'safety_removed': removed,
                         'quantized': quantized, 'share_safety_removed': removed / len(desc) if desc else 0})
        numbers['root_descendant_flags'] = rows
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labels = [r['model_id'].split('/')[-1] for r in rows]
        ax.barh(labels, [r['descendants'] for r in rows], color='#c9d6e8', label='All descendants')
        ax.barh(labels, [r['safety_removed'] for r in rows], color='#c0504d',
                label='Advertised as uncensored / abliterated')
        ax.invert_yaxis()
        for i, r in enumerate(rows):
            ax.text(r['descendants'], i, f" {r['share_safety_removed']:.0%}", va='center', fontsize=7)
        ax.set_xlabel('Descendants in the 50k snapshot')
        ax.set_title('Descendants that advertise removed safety training')
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        fig.tight_layout()
        fig.savefig(args.output / 'safety_removed_descendants.png')
        plt.close(fig)

    (args.output / 'layer1_numbers.json').write_text(json.dumps(numbers, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in numbers.items() if k != 'root_descendant_flags'}, indent=2))


if __name__ == '__main__':
    main()
