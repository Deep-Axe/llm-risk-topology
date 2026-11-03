import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from power_analysis import (
    average_precision,
    correlation_for_auroc,
    delta_pr_auc_interval,
    draw_split,
    load_pair_structure,
    one_replicate,
    resample_group_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'data' / 'processed' / 'prompts' / 'pilot_v4'


class PowerAnalysisTest(unittest.TestCase):
    def test_average_precision_matches_sklearn(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, size=40)
        scores = rng.normal(size=40)
        self.assertAlmostEqual(average_precision(y, scores), average_precision_score(y, scores), places=10)

    def test_bootstrap_resamples_whole_groups(self):
        group_ids = np.array([0, 0, 0, 1, 1, 2])
        rows, chosen = resample_group_rows(group_ids, np.random.default_rng(1))
        self.assertEqual(chosen.size, 3)
        widths = {0: 3, 1: 2, 2: 1}
        self.assertEqual(rows.size, sum(widths[int(group)] for group in chosen))
        members = {0: {0, 1, 2}, 1: {3, 4}, 2: {5}}
        offset = 0
        for group in chosen:
            width = len(members[int(group)])
            self.assertEqual(set(rows[offset:offset + width].tolist()), members[int(group)])
            offset += width

    def test_prevalence_and_baseline_auroc(self):
        prevalence = 0.2
        r0 = correlation_for_auroc(prevalence, 0.75, n=120000, seed=11)
        drawn = draw_split(
            np.ones(8000, dtype=int), 1, prevalence, 0.0, 0.0, r0, 0.0, np.random.default_rng(12),
        )
        self.assertAlmostEqual(drawn['y'].mean(), prevalence, delta=0.015)
        self.assertAlmostEqual(roc_auc_score(drawn['y'], drawn['s0']), 0.75, delta=0.02)

    def test_null_lower_bound_rate_is_near_one_sided_2_5_percent(self):
        structure = {
            'all_sizes': np.array([1, 1, 1, 2], dtype=int),
            'test_sizes': np.ones(40, dtype=int),
            'test_groups': 40,
            'dev_per_test': 2.0,
        }
        r0 = correlation_for_auroc(0.3, 0.75, n=80000, seed=3)
        rng = np.random.default_rng(21)
        hits = 0
        reps = 200
        for _ in range(reps):
            _, lower, _, _ = one_replicate(
                structure, 40, 2, 0.3, 0.15, 0.35, r0, 0.0, 99, rng,
            )
            if np.isfinite(lower) and lower > 0.0:
                hits += 1
        rate = hits / reps
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 0.08)

    def test_fixed_seed_repeats(self):
        structure = load_pair_structure(PILOT)
        r0 = 0.4
        kwargs = dict(
            structure=structure,
            n_test_groups=structure['test_groups'],
            n_checkpoints=2,
            prevalence=0.2,
            rho_within=0.15,
            rho_cross=0.35,
            r0=r0,
            ra=0.2,
            n_boot=19,
        )
        first = one_replicate(**kwargs, rng=np.random.default_rng(8))
        second = one_replicate(**kwargs, rng=np.random.default_rng(8))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        scores = np.array([0.1, 0.4, 0.2, 0.9])
        groups = np.array([0, 0, 1, 1])
        interval = delta_pr_auc_interval(y, scores, scores + 0.01, groups, 15, np.random.default_rng(1))
        again = delta_pr_auc_interval(y, scores, scores + 0.01, groups, 15, np.random.default_rng(1))
        self.assertEqual(interval, again)


if __name__ == '__main__':
    unittest.main()
