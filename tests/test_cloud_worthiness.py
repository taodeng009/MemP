import unittest
import numpy as np
from ProcedureMem.analyze_cloud_worthiness import classification_metrics, logistic_predictions


class CloudWorthinessTests(unittest.TestCase):
    def test_metrics_perfect_reversed_and_ties(self):
        y = np.array([0., 0., 1., 1.])
        perfect = classification_metrics(y, np.array([.1, .2, .8, .9]))
        self.assertEqual(perfect['roc_auc'], 1)
        self.assertEqual(perfect['pr_auc_trapezoidal'], 1)
        self.assertEqual(perfect['average_precision'], 1)
        self.assertEqual(classification_metrics(y, np.array([.9, .8, .2, .1]))['roc_auc'], 0)
        tied = classification_metrics(y, np.ones(4))
        self.assertEqual(tied['roc_auc'], .5)
        self.assertEqual(tied['average_precision'], .5)

    def test_logistic_constant_features_and_regularized_linear_signal(self):
        y = np.array([0., 0., 0., 1.])
        result = logistic_predictions(np.zeros((4, 3)), y, np.zeros((2, 3)), [1.])
        np.testing.assert_allclose(result[1.], [.25, .25], atol=1e-8)
        x = np.array([[-2.], [-1.], [1.], [2.]])
        y = np.array([0., 0., 1., 1.])
        p = logistic_predictions(x, y, x, [1.])[1.]
        self.assertTrue(np.all(np.diff(p) > 0))
        np.testing.assert_allclose(p, 1 - p[::-1], atol=1e-8)
        # Primal gradient check of the reduced Newton solution.
        standardized = x[:, 0] / x[:, 0].std()
        w = np.log(p / (1 - p))[-1] / standardized[-1]
        self.assertAlmostEqual(float(standardized @ (p - y) + w), 0, places=6)


if __name__ == '__main__':
    unittest.main()
