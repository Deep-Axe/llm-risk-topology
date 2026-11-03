import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from multimodel import pooled_metrics, score_model, template_keep_mask
from paired_outcomes import outcomes_from_pairs


def _responses(model_id, n_groups, split, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(n_groups):
        base = rng.normal(size=8)
        original_refusal = bool(group % 2 == 0)
        rows.append(_row(model_id, group, 'o', 'original', split, original_refusal, base))
        variant_refusal = original_refusal if group % 3 else (not original_refusal)
        rows.append(_row(model_id, group, 'v', 'adversarial', split, variant_refusal, base + rng.normal(scale=0.1, size=8)))
    return rows


def _row(model, group, side, variant, split, refusal, vector):
    return {
        'model_id': model,
        'group_id': '%s-%d' % (split, group),
        'source': 'wildjailbreak',
        'intent': 'harmful',
        'split': split,
        'variant_type': variant,
        'prompt_id': '%s-%s-%s-%d' % (model, split, side, group),
        'response_id': 'r-%s-%s-%s-%d' % (model, split, side, group),
        'refusal_rules': refusal,
        'refusal': pd.NA,
        'response_harmful': pd.NA,
        'output_tokens': 10,
        'text_chars': 20,
        'vector': vector.tolist(),
    }


def _pairs(n_groups, split, model):
    return pd.DataFrame([
        {
            'original_prompt_id': '%s-%s-o-%d' % (model, split, group),
            'variant_prompt_id': '%s-%s-v-%d' % (model, split, group),
            'variant_type': 'adversarial',
        }
        for group in range(n_groups)
    ])


class MultimodelTest(unittest.TestCase):
    def test_pairs_file_builds_one_row_per_variant(self):
        frame = pd.DataFrame(_responses('m', 4, 'validation', 0))
        pairs = _pairs(4, 'validation', 'm')
        paired, exclusions = outcomes_from_pairs(frame, pairs, 'refusal_rules')
        self.assertEqual(len(paired), 4)
        self.assertEqual(exclusions['pair_endpoints_missing'], 0)
        self.assertIn('refusal_flip', paired.columns)

    def test_two_models_pool_and_validation_only_protocol(self):
        rows = _responses('m-a', 24, 'validation', 1) + _responses('m-b', 24, 'validation', 2)
        frame = pd.DataFrame(rows)
        pairs = pd.concat([_pairs(24, 'validation', 'm-a'), _pairs(24, 'validation', 'm-b')], ignore_index=True)
        results = []
        for model_id, part in frame.groupby('model_id'):
            model_pairs = pairs[pairs['original_prompt_id'].str.startswith(model_id)]
            paired, _ = outcomes_from_pairs(part, model_pairs, 'refusal_rules')
            paired['original_prompt_chars'] = 10
            paired['variant_prompt_chars'] = 12
            paired['variant_prompt_text'] = ['alpha beta gamma delta epsilon zeta eta theta extra words %d' % i for i in range(len(paired))]
            distance = np.linspace(0.1, 0.4, len(paired))
            results.append(score_model(paired, distance, [3], n_boot=29, seed=3))
        self.assertEqual({row['protocol'] for row in results}, {'group_kfold_validation'})
        self.assertTrue(all(row['exploratory_selection'] for row in results))
        fixed = score_model(
            results[0]['scored'].assign(original_prompt_chars=10, variant_prompt_chars=12),
            np.full(len(results[0]['scored']), 0.2),
            [3, 5], n_boot=19, seed=3, fixed=(5, 'spectral_margin'),
        )
        self.assertEqual(fixed['chosen_k'], 5)
        self.assertEqual(fixed['chosen_boundary'], 'spectral_margin')
        self.assertFalse(fixed['exploratory_selection'])
        pooled = pooled_metrics(results, n_boot=29, seed=3)
        self.assertEqual(pooled['n_models'], 2)
        self.assertTrue(np.isfinite(pooled['delta_pr_auc']))

    def test_development_split_is_not_scored_as_validation(self):
        rows = _responses('m', 45, 'development', 4) + _responses('m', 20, 'validation', 5)
        frame = pd.DataFrame(rows)
        pairs = pd.concat([_pairs(45, 'development', 'm'), _pairs(20, 'validation', 'm')], ignore_index=True)
        paired, _ = outcomes_from_pairs(frame, pairs, 'refusal_rules')
        paired['original_prompt_chars'] = 10
        paired['variant_prompt_chars'] = 12
        paired['variant_prompt_text'] = ['word ' * 12] * len(paired)
        distance = np.full(len(paired), 0.2)
        result = score_model(paired, distance, [3], n_boot=19, seed=6)
        self.assertEqual(result['protocol'], 'development_then_validation')
        self.assertEqual(len(result['scored']), 20)
        fixed = score_model(paired, distance, [3, 5], n_boot=19, seed=6, fixed=(5, 'fiedler_magnitude'))
        self.assertEqual(fixed['chosen_k'], 5)
        self.assertEqual(fixed['chosen_boundary'], 'fiedler_magnitude')
        self.assertFalse(fixed['exploratory_selection'])
        training = paired.loc[paired['split'].eq('development'), ['group_id', 'variant_prompt_text']]
        mask = template_keep_mask(result['scored'], training)
        self.assertEqual(len(mask), 20)
        self.assertFalse(bool(mask.any()))

    def test_shared_template_is_visible_when_the_query_group_also_has_it(self):
        text = 'alpha beta gamma delta epsilon zeta eta theta iota kappa'
        scored = pd.DataFrame({
            'group_id': ['a', 'b'],
            'variant_prompt_text': [text + ' one', text + ' two'],
        })
        mask = template_keep_mask(scored, scored)
        self.assertFalse(bool(mask.any()))

    def test_short_validation_templates_are_not_applicable(self):
        from multimodel import template_fold_mask, template_summary
        scored = pd.DataFrame({
            'group_id': ['a', 'b'],
            'variant_prompt_text': ['bake a cake', 'boil an egg'],
            'refusal_flip': [0, 1],
        })
        self.assertIsNone(template_fold_mask(scored))
        paired = scored.assign(split='validation')
        summary = template_summary(scored, paired, np.array([0.2, 0.4]), np.array([0.3, 0.5]), 0, 1)
        self.assertEqual(summary['status'], 'n/a')
        self.assertNotIn('delta_pr_auc', summary)

    def test_fold_reports_dropped_pairs_without_copying_delta(self):
        from multimodel import template_summary
        stem = 'alpha beta gamma delta epsilon zeta eta theta iota kappa'
        scored = pd.DataFrame({
            'group_id': ['a', 'b'],
            'variant_prompt_text': [stem + ' one', stem + ' two'],
            'refusal_flip': [0, 1],
        })
        paired = scored.assign(split='validation')
        summary = template_summary(scored, paired, np.array([0.2, 0.8]), np.array([0.3, 0.9]), 0, 1)
        self.assertEqual(summary['status'], 'compared')
        self.assertEqual(summary['n_dropped'], 2)
        self.assertEqual(summary['n_kept'], 0)
        self.assertIsNone(summary.get('delta_pr_auc'))

    def test_tables_split_into_three_files(self):
        import tempfile
        from multimodel import write_tables
        row = {
            'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'family': 'Qwen', 'label_source': 'refusal_rules',
            'protocol': 'group_kfold_validation', 'n_pairs': 4, 'flip_rate': 0.5,
            'baseline_pr_auc': 0.4, 'full_pr_auc': 0.5, 'delta': 0.1, 'lower': -0.1, 'upper': 0.2,
            'template_held_out': {'status': 'n/a'},
            'without_overlap': {'n_pairs': 3, 'delta_pr_auc': 0.05, 'delta_lower': -0.1, 'delta_upper': 0.1},
            'k_sensitivity': [],
        }
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'layer2_tables.tex').write_text('old', encoding='utf-8')
            write_tables(folder, [row], None, {'n_pairs': 4, 'baseline_pr_auc': 0.4, 'full_pr_auc': 0.5,
                                                'delta_pr_auc': 0.1, 'delta_lower': -0.1, 'delta_upper': 0.2}, 8)
            numbers = (folder / 'layer2_numbers.tex').read_text(encoding='utf-8')
            main = (folder / 'layer2_main_table.tex').read_text(encoding='utf-8')
            sens = (folder / 'layer2_sensitivity_table.tex').read_text(encoding='utf-8')
            self.assertIn('\\LtwoFlipBy', numbers)
            self.assertIn('\\LtwoPrimaryModels', numbers)
            self.assertIn('Mistral-7B-Instruct-v0.1, Zephyr-7b-beta', numbers)
            header = [line for line in main.splitlines() if line.startswith('Model')][0]
            self.assertEqual(header.count('&'), 6)
            self.assertIn('rules/cv', main)
            self.assertNotIn('Family', main)
            self.assertIn('template held out & Qwen2.5-0.5B-Instruct & n/a & n/a', sens)
            self.assertFalse((folder / 'layer2_tables.tex').exists())


if __name__ == '__main__':
    unittest.main()
