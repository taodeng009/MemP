import unittest
import numpy as np
from ProcedureMem.analyze_cumulative_offloading_gain import analyze, intervals


class CumulativeGainTests(unittest.TestCase):
    def test_endpoints_random_expectation_and_oracle(self):
        gain = np.array([1., 1/3, 0., -1.])
        rows, summary, _ = analyze(gain, {'minus_ru': np.arange(4.), 'bd': -np.arange(4.)}, 200, 42)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]['random_expectation'], gain.mean())
        for key in ['minus_ru_gain', 'bd_gain', 'oracle_gain', 'random_pointwise_95_low', 'random_pointwise_95_high']:
            self.assertAlmostEqual(rows[-1][key], gain.sum())
        self.assertEqual(rows[-1]['bd_familywise_p'], 1)
        for row in rows:
            self.assertGreaterEqual(row['oracle_gain'] + 1e-12, row['bd_gain'])
            self.assertGreaterEqual(row['oracle_gain'] + 1e-12, row['minus_ru_gain'])
        self.assertEqual(intervals(np.array([True, True, False, True])), [[1, 2], [4, 4]])


if __name__ == '__main__':
    unittest.main()
