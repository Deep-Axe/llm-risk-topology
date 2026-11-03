import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from paired_outcomes import outcomes_from_frame
from response_graph import graph_spectrum, knn_adjacency, normalized_laplacian, smallest_eigenpairs


def _frame():
    rows = [
        {'model_id': 'm', 'group_id': 'g1', 'source': 's', 'intent': 'harmful', 'split': 'validation',
         'variant_type': 'original', 'prompt_id': 'po', 'response_id': 'ro', 'refusal_rules': True,
         'response_harmful': False, 'output_tokens': 10, 'text_chars': 20, 'vector': [1.0, 0.0]},
        {'model_id': 'm', 'group_id': 'g1', 'source': 's', 'intent': 'harmful', 'split': 'validation',
         'variant_type': 'adversarial', 'prompt_id': 'pv', 'response_id': 'rv', 'refusal_rules': False,
         'response_harmful': True, 'output_tokens': 12, 'text_chars': 22, 'vector': [0.0, 1.0]},
        {'model_id': 'm', 'group_id': 'g2', 'source': 's', 'intent': 'harmful', 'split': 'validation',
         'variant_type': 'original', 'prompt_id': 'po2', 'response_id': 'ro2', 'refusal_rules': False,
         'response_harmful': pd.NA, 'output_tokens': 8, 'text_chars': 9, 'vector': [1.0, 0.0]},
        {'model_id': 'm', 'group_id': 'g2', 'source': 's', 'intent': 'harmful', 'split': 'validation',
         'variant_type': 'defense', 'prompt_id': 'pv2', 'response_id': 'rv2', 'refusal_rules': True,
         'response_harmful': pd.NA, 'output_tokens': 8, 'text_chars': 9, 'vector': [0.0, 1.0]},
    ]
    return pd.DataFrame(rows)


class Layer2Test(unittest.TestCase):
    def test_flip_direction_and_harm(self):
        paired, exclusions = outcomes_from_frame(_frame(), 'refusal_rules')
        self.assertEqual(len(paired), 2)
        self.assertEqual(exclusions['groups_without_one_original'], 0)
        first = paired[paired['group_id'] == 'g1'].iloc[0]
        self.assertEqual(first['flip_direction'], 'refused_to_complied')
        self.assertEqual(bool(first['harmful_transition']), True)
        second = paired[paired['group_id'] == 'g2'].iloc[0]
        self.assertEqual(second['flip_direction'], 'complied_to_refused')
        self.assertTrue(pd.isna(second['harmful_transition']))
        self.assertNotIn('variant_vector', paired.columns)

    def test_sparse_eigenvalues_match_dense(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(30, 6))
        adjacency = knn_adjacency(points, k=5)
        laplacian = normalized_laplacian(adjacency)
        sparse_values, _ = smallest_eigenpairs(laplacian, 6, dense=False)
        dense_values, _ = smallest_eigenpairs(laplacian, 6, dense=True)
        np.testing.assert_allclose(sparse_values, dense_values, atol=1e-5)
        spectrum = graph_spectrum(points, k=5, n_components=6)
        self.assertEqual(len(spectrum['eigenvalues']), 6)
        self.assertGreaterEqual(spectrum['n_components'], 1)


if __name__ == '__main__':
    unittest.main()
