import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from multimodel import family_size
from rule_label_check import draw_sample, rule_metrics


class FamilySizeTests(unittest.TestCase):
    def test_llama_and_smollm_are_named(self):
        self.assertEqual(family_size('meta-llama/Llama-3.2-1B-Instruct'), ('Llama', '1B'))
        self.assertEqual(family_size('HuggingFaceTB/SmolLM2-1.7B-Instruct'), ('SmolLM', '1.7B'))


class RuleSampleTests(unittest.TestCase):
    def test_sample_keeps_every_rule_refusal_up_to_thirty(self):
        rows = []
        for model, n_pos in (
            ('mistralai/Mistral-7B-Instruct-v0.1', 30),
            ('HuggingFaceH4/zephyr-7b-beta', 13),
        ):
            for index in range(n_pos):
                rows.append(_response(model, index, True))
            for index in range(n_pos, n_pos + 200):
                rows.append(_response(model, index, False))
        sample = draw_sample(pd.DataFrame(rows), seed=20261104, size=100, max_positives=30)
        for model, n_pos in (
            ('mistralai/Mistral-7B-Instruct-v0.1', 30),
            ('HuggingFaceH4/zephyr-7b-beta', 13),
        ):
            part = sample[sample['model_id'].eq(model)]
            self.assertEqual(len(part), 100)
            self.assertEqual(int(part['refusal_rules'].sum()), n_pos)
            self.assertTrue(np.isclose(part.loc[part['stratum'].eq('rule_refused'), 'weight'], 1.0).all())

    def test_weighted_recall_scales_the_negative_stratum(self):
        rows = []
        for index in range(2):
            rows.append({'model_id': 'm', 'refusal_rules': True, 'agent_refusal': True, 'weight': 1.0})
        rows.append({'model_id': 'm', 'refusal_rules': False, 'agent_refusal': True, 'weight': 10.0})
        rows.append({'model_id': 'm', 'refusal_rules': False, 'agent_refusal': False, 'weight': 10.0})
        metrics = rule_metrics(rows)['by_model']['m']
        self.assertEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 2.0 / 12.0)
        self.assertAlmostEqual(metrics['estimated_refusal_rate'], 12.0 / 22.0)


def _response(model, index, refused):
    return {
        'model_id': model,
        'split': 'validation',
        'prompt_id': '%s-%d' % (model, index),
        'response_id': 'r-%s-%d' % (model, index),
        'variant_type': 'original',
        'refusal_rules': refused,
    }


if __name__ == '__main__':
    unittest.main()
