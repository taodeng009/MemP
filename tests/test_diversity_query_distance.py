import unittest

from ProcedureMem.analyze_diversity_query_distance import compute_pool_query_metrics


class PoolQueryDistanceTests(unittest.TestCase):
    def test_computes_coverage_and_mean_nearest_distance(self):
        pools = [
            {
                "pool_id": "pool-a",
                "memory_ids": ["memory-a", "memory-b"],
            }
        ]
        distances = {
            "memory-a": (0.2, 0.8, 0.6),
            "memory-b": (0.4, 0.3, 0.7),
        }

        result = compute_pool_query_metrics(
            pools,
            distances,
            query_count=3,
            threshold=0.5,
        )

        self.assertAlmostEqual(result["pool-a"]["coverage"], 2 / 3)
        self.assertAlmostEqual(
            result["pool-a"]["mean_nearest_query_memory_distance"],
            (0.2 + 0.3 + 0.6) / 3,
        )


if __name__ == "__main__":
    unittest.main()
