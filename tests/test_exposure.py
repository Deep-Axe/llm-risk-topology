import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from exposure import Q_GRID, brute_force, build, descendant_reach, edge_uniform, exposure, tree_exact  # noqa: E402

CHAIN = [('a/root', 'a/one'), ('a/one', 'a/two'), ('a/two', 'a/three')]
TREE = [('t/root', 't/x'), ('t/root', 't/y'), ('t/x', 't/x1'), ('t/x', 't/x2'), ('t/y', 't/y1')]
# Two routes from the root to the merged child; it must be counted once.
DIAMOND = [('d/root', 'd/left'), ('d/root', 'd/right'), ('d/left', 'd/merge'), ('d/right', 'd/merge'),
           ('d/merge', 'd/leaf')]


def by_node(rows, q):
    return {row['model_id']: row for row in rows if row['q'] == q}


class ExposureTest(unittest.TestCase):
    def test_uniform_is_deterministic_and_in_range(self):
        value = edge_uniform(42, 3, 'a/b', 'c/d')
        self.assertEqual(value, edge_uniform(42, 3, 'a/b', 'c/d'))
        self.assertNotEqual(value, edge_uniform(42, 4, 'a/b', 'c/d'))
        self.assertTrue(0 <= value < 1)

    def test_reach_counts_merged_descendant_once(self):
        children, order = build(DIAMOND)
        self.assertEqual(descendant_reach(children, order)['d/root'], 4)

    def test_q_zero_and_one(self):
        rows = exposure(DIAMOND + TREE, replicates=5)
        for row in by_node(rows, 0.0).values():
            self.assertEqual(row['e_q_mean'], 0)
        for row in by_node(rows, 1.0).values():
            self.assertEqual(row['e_q_mean'], row['reach'])
            self.assertEqual(row['e_q_se'], 0)

    def test_tree_exact_matches_brute_force(self):
        children, order = build(TREE + CHAIN)
        exact = tree_exact(children, order)
        for i, q in enumerate(Q_GRID):
            truth = brute_force(TREE + CHAIN, q)
            for node, values in exact.items():
                self.assertAlmostEqual(values[i], truth[node], places=12)

    def test_tree_exact_is_none_with_two_paths(self):
        children, order = build(DIAMOND)
        exact = tree_exact(children, order)
        self.assertIsNone(exact['d/root'])
        self.assertIsNotNone(exact['d/left'])

    def test_diamond_closed_form(self):
        q = 0.5
        reach_merge = 1 - (1 - q * q) ** 2
        expected = 2 * q + reach_merge + reach_merge * q
        self.assertAlmostEqual(brute_force(DIAMOND, q)['d/root'], expected, places=12)

    def test_monte_carlo_agrees_with_brute_force(self):
        rows = exposure(DIAMOND + CHAIN, replicates=4000, seed=7)
        for q in (0.3, 0.7):
            truth = brute_force(DIAMOND + CHAIN, q)
            for node, row in by_node(rows, q).items():
                self.assertLess(abs(row['e_q_mean'] - truth[node]), 4 * row['e_q_se'] + 1e-9, (node, q))

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            build([('c/a', 'c/b'), ('c/b', 'c/a')])

    def test_same_seed_same_result(self):
        self.assertEqual(exposure(DIAMOND, replicates=20, seed=3), exposure(DIAMOND, replicates=20, seed=3))


if __name__ == '__main__':
    unittest.main()
