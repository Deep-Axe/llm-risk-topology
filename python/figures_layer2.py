"""Behavioral figures for the report: refusal before/after rewrites per model, flip rates, and lineage vs behavior.

Run: python python/figures_layer2.py --staged DIR [DIR ...] [--panel-lineage report/figures/layer1/panel_lineage.json]
Each DIR holds responses.parquet from python/stage_layer2.py. Uses WildGuard refusal where present, otherwise the
refusal_rules label, and records which one in the output numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRATA = [('wildjailbreak', 'harmful', 'adversarial', 'Harmful request,\njailbreak rewrite'),
          ('Salad-Data', 'harmful', 'adversarial', 'Harmful question,\nattack rewrite'),
          ('wildjailbreak', 'unharmful', 'adversarial', 'Benign request,\nadversarial-style rewrite')]


def short(model):
    return model.split('/')[-1].replace('-Instruct', '-Inst')


def pair_table(frame, pairs):
    lookup = frame.set_index(['model_id', 'prompt_id'])
    rows = []
    for model in sorted(frame['model_id'].unique()):
        for p in pairs:
            a, b = (model, p['original_prompt_id']), (model, p['variant_prompt_id'])
            if a in lookup.index and b in lookup.index:
                o, v = lookup.loc[a], lookup.loc[b]
                if pd.isna(o['label']) or pd.isna(v['label']):
                    continue
                rows.append({'model_id': model, 'source': p['source_id'].split('/')[-1], 'intent': o['intent'],
                             'variant_type': p['variant_type'], 'split': o['split'], 'original_refusal': bool(o['label']),
                             'variant_refusal': bool(v['label'])})
    table = pd.DataFrame(rows)
    table['flip'] = table['original_refusal'] != table['variant_refusal']
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--staged', type=Path, nargs='+', required=True)
    parser.add_argument('--pairs', type=Path, default=ROOT / 'data/processed/prompts/pilot_v5/pairs.jsonl')
    parser.add_argument('--panel-lineage', type=Path, default=ROOT / 'report/figures/layer1/panel_lineage.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'report/figures/layer2_lead')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    frame = pd.concat([pd.read_parquet(d / 'responses.parquet') for d in args.staged], ignore_index=True)
    frame = frame.drop_duplicates(['model_id', 'prompt_id'])
    frame['label_source'] = frame['refusal'].notna().map({True: 'wildguard', False: 'refusal_rules'})
    frame['label'] = frame['refusal'].astype('boolean').fillna(frame['refusal_rules'].astype('boolean'))
    pairs = [json.loads(line) for line in args.pairs.open(encoding='utf-8')]
    table = pair_table(frame, pairs)
    models = sorted(table['model_id'].unique())
    numbers = {'responses': int(len(frame)), 'models': models, 'pairs_scored': int(len(table)),
               'label_source': frame.groupby('model_id')['label_source'].agg(lambda s: s.mode()[0]).to_dict(),
               'flip_rate': table.groupby('model_id')['flip'].mean().round(3).to_dict()}

    # Refusal of originals vs rewrites, per model, for each stratum.
    fig, axes = plt.subplots(1, len(STRATA), figsize=(3.3 * len(STRATA), 3.6), sharey=True)
    width = .38
    numbers['strata'] = {}
    for ax, (source, intent, variant, title) in zip(axes, STRATA):
        sub = table[(table.source == source) & (table.intent == intent) & (table.variant_type == variant)]
        stats = sub.groupby('model_id')[['original_refusal', 'variant_refusal']].mean().reindex(models)
        xs = range(len(models))
        ax.bar([x - width / 2 for x in xs], stats['original_refusal'], width, color='#3b6ea5', label='Original')
        ax.bar([x + width / 2 for x in xs], stats['variant_refusal'], width, color='#c0504d', label='Rewrite')
        ax.set_xticks(list(xs))
        ax.set_xticklabels([short(m) for m in models], rotation=40, ha='right', fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        numbers['strata'][f'{source}/{intent}/{variant}'] = stats.round(3).reset_index().to_dict('records')
    axes[0].set_ylabel('Refusal rate')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Refusal before and after rewriting the same request', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output / 'refusal_before_after.png')
    plt.close(fig)

    # Flip rate per model.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    flips = table.groupby('model_id')['flip'].mean().reindex(models)
    ax.bar([short(m) for m in models], flips, color='#3b6ea5')
    for i, value in enumerate(flips):
        ax.text(i, value, f'{value:.0%}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Pairs where refusal changes')
    ax.set_title('How often a rewrite changes the refusal decision')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'flip_rate.png')
    plt.close(fig)

    # Lineage exposure vs jailbreak effect (exploratory; one point per model).
    if args.panel_lineage.exists():
        lineage = {r['model_id']: r for r in json.loads(args.panel_lineage.read_text(encoding='utf-8'))}
        jail = table[(table.intent == 'harmful') & (table.variant_type == 'adversarial')]
        drop = (jail.groupby('model_id')['original_refusal'].mean() - jail.groupby('model_id')['variant_refusal'].mean())
        points = [(m, lineage[m]['e_q_0.3'], drop[m]) for m in models if m in lineage and m in drop]
        numbers['lineage_vs_behavior'] = [{'model_id': m, 'e_q_0.3': e, 'refusal_drop': round(float(d), 3)}
                                          for m, e, d in points]
        if len(points) >= 2:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.scatter([p[1] for p in points], [p[2] for p in points], s=40, color='#3b6ea5')
            for m, e, d in points:
                ax.annotate(short(m), (e, d), fontsize=7, xytext=(4, 3), textcoords='offset points')
            ax.set_xlabel('Downstream exposure of the checkpoint, E_q (q = 0.3)')
            ax.set_ylabel('Refusal drop under adversarial rewrite')
            ax.set_title('Lineage position vs jailbreak sensitivity (exploratory)')
            fig.tight_layout()
            fig.savefig(args.output / 'lineage_vs_behavior.png')
            plt.close(fig)

    (args.output / 'layer2_numbers.json').write_text(json.dumps(numbers, indent=2, default=str) + '\n',
                                                     encoding='utf-8')
    print(json.dumps({k: numbers[k] for k in ['responses', 'models', 'pairs_scored', 'flip_rate']}, indent=2))


if __name__ == '__main__':
    main()
