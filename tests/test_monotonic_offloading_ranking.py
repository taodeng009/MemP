import unittest
import numpy as np
from ProcedureMem.analyze_monotonic_offloading_ranking import spearman


class MonotonicRankingTests(unittest.TestCase):
    def test_ties_direction_and_constant(self):
        x = np.array([1., 2., 2., 4.])
        self.assertAlmostEqual(spearman(x, x), 1)
        self.assertAlmostEqual(spearman(-x, x), -1)
        self.assertIsNone(spearman(np.ones(4), x))
        np.testing.assert_array_equal(np.argsort(-x, kind='stable'), [3, 1, 2, 0])


if __name__ == '__main__':
    unittest.main()
