import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.build_diversity_pools import (
    generate_unique_subsets,
    mean_nearest_neighbor_squared_l2,
    select_across_quantile_bins,
)
from ProcedureMem.summarize_diversity_experiment import build_diversity_results


class DiversityPoolConstructionTests(unittest.TestCase):
    def test_mean_nearest_neighbor_squared_l2(self):
        distances = {
            "a": {"a": 0.0, "b": 1.0, "c": 9.0},
            "b": {"a": 1.0, "b": 0.0, "c": 4.0},
            "c": {"a": 9.0, "b": 4.0, "c": 0.0},
        }
        self.assertAlmostEqual(
            mean_nearest_neighbor_squared_l2(("a", "b", "c"), distances),
            2.0,
        )

    def test_random_subsets_are_reproducible_unique_and_equal_size(self):
        ids = [f"mem_{index:04d}" for index in range(12)]
        first = generate_unique_subsets(ids, pool_size=4, count=30, seed=17)
        second = generate_unique_subsets(ids, pool_size=4, count=30, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(len(pool) == 4 for pool in first))

    def test_formal_pools_cover_all_quantile_bins_without_duplicate_subsets(self):
        candidates = [
            {"memory_ids": [f"a{index}", f"b{index}"], "diversity": float(index)}
            for index in range(40)
        ]
        pools = select_across_quantile_bins(
            candidates,
            bin_count=4,
            pools_per_bin=3,
            seed=11,
        )
        self.assertEqual(len(pools), 12)
        self.assertEqual(
            [sum(pool["quantile_bin"] == index for pool in pools) for index in range(4)],
            [3, 3, 3, 3],
        )
        subsets = [tuple(pool["memory_ids"]) for pool in pools]
        self.assertEqual(len(subsets), len(set(subsets)))


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is not installed")
class DiversitySummaryTests(unittest.TestCase):
    def test_summary_uses_all_pools_and_records_spearman(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = []
            for index in range(4):
                pool_id = f"q{index:02d}_p00"
                pools.append(
                    {
                        "pool_id": pool_id,
                        "quantile_bin": index,
                        "quantile_range": [index / 4, (index + 1) / 4],
                        "diversity": float(index + 1),
                        "memory_ids": [f"m{index}a", f"m{index}b"],
                    }
                )
                condition = root / f"diversity_{pool_id}"
                condition.mkdir()
                parameters = {
                    key: "same"
                    for key in (
                        "model",
                        "agent_api_base_url",
                        "embedding_model",
                        "split",
                        "seed",
                        "batch_size",
                        "max_steps",
                        "temperature",
                        "top_p",
                        "few_shot",
                        "top_k",
                        "score_threshold",
                        "manifest_sha256",
                    )
                }
                parameters.update(
                    {
                        "condition_mode": "diversity_pool",
                        "pool_id": pool_id,
                        "manifest": "tasks.json",
                    }
                )
                (condition / "summary.json").write_text(
                    json.dumps(
                        {
                            "parameters": parameters,
                            "task_ids": ["task-a", "task-b"],
                            "task_count": 2,
                            "success_rate": index / 3,
                        }
                    ),
                    encoding="utf-8",
                )
            pool_path = root / "pools.json"
            pool_path.write_text(
                json.dumps(
                    {
                        "diversity_metric": "mean_nearest_neighbor_squared_l2",
                        "generation_parameters": {"pool_size": 2},
                        "pools": pools,
                    }
                ),
                encoding="utf-8",
            )

            result = build_diversity_results(root, pool_path)

            self.assertEqual(len(result["pool_results"]), 4)
            self.assertAlmostEqual(result["spearman"]["rho"], 1.0)
            self.assertEqual(len(result["quantile_trend"]), 4)


if __name__ == "__main__":
    unittest.main()
