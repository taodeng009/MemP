import unittest

from ProcedureMem.analyze_edge_cloud_propensities import values


class PropensityTests(unittest.TestCase):
    def test_all_empirical_count_combinations(self):
        for edge in range(4):
            for cloud in range(4):
                rescue, negative, gain = values(edge, cloud)
                self.assertAlmostEqual(rescue, (1 - edge / 3) * cloud / 3)
                self.assertAlmostEqual(negative, edge / 3 * (1 - cloud / 3))
                self.assertAlmostEqual(gain, cloud / 3 - edge / 3)
                self.assertAlmostEqual(rescue - negative, gain)
                self.assertEqual(negative > 0, edge > 0 and cloud < 3)


if __name__ == '__main__':
    unittest.main()
