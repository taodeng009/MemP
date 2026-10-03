import unittest

import numpy as np

from ProcedureMem.analyze_task_semantics_linear_probe import (
    canonical_query, grouped_folds, ridge_predictions,
)


class LinearProbeTests(unittest.TestCase):
    def test_query_groups_never_split_and_cover_every_task(self):
        groups = ['a'] * 4 + ['b'] * 3 + ['c', 'd', 'e', 'f', 'g', 'h']
        folds = grouped_folds(groups, 5, 42)
        self.assertEqual(sorted(np.concatenate(folds)), list(range(len(groups))))
        group_fold = {}
        for fold_id, indices in enumerate(folds):
            for i in indices:
                if groups[i] in group_fold:
                    self.assertEqual(group_fold[groups[i]], fold_id)
                group_fold[groups[i]] = fold_id
        self.assertEqual(canonical_query('  Put A Mug   On Desk.! '), 'put a mug on desk')

    def test_dual_ridge_matches_primal_with_train_only_scaling(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(12, 7))
        train[:, -1] = 3
        test = rng.normal(size=(4, 7)) + 10
        y = rng.normal(size=12)
        center, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale < 1e-12] = 1
        z = (train - center) / scale
        for alpha, predictions in ridge_predictions(train, y, test, [0.1, 100]).items():
            beta = np.linalg.solve(z.T @ z + alpha * np.eye(7), z.T @ (y - y.mean()))
            expected = y.mean() + ((test - center) / scale) @ beta
            np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
