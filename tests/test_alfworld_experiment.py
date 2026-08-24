import json
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.alfworld_experiment import (
    build_condition_comparison,
    build_paired_comparison,
    build_task_manifest,
    inject_memory,
    inject_trajectories,
    load_json,
    manifest_sha256,
    maybe_write_scheduling_comparison,
    maybe_write_online_construction_comparison,
    retrieval_records,
    reranked_retrieval_records,
    summarize_results,
    task_query,
    validate_task_manifest,
    write_results,
)


class FakeDocument:
    def __init__(self, **metadata):
        self.metadata = metadata


class SchedulingComparisonTests(unittest.TestCase):
    def test_online_comparison_uses_fifo_and_greedy_with_matching_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "condition_mode": "online_construction",
                "model": "model",
                "agent_api_base_url": "api",
                "embedding_model": "embedding",
                "split": "valid_unseen",
                "seed": 42,
                "batch_size": 1,
                "max_steps": 30,
                "temperature": 0,
                "top_p": 1.0,
                "few_shot": True,
                "top_k": 3,
                "score_threshold": 0.5,
                "manifest_sha256": "manifest",
                "interval_size": 10,
                "construction_capacity": 2,
                "construction_method": "direct",
                "arrival_policy": "success_only",
                "memory_build_model": "builder",
                "warm_start_count": 0,
                "warm_start_seed": 42,
                "warm_start_memory_file": None,
                "initial_available_memory_ids": [],
            }
            for policy, success_rate, steps in (
                ("fifo", 0.4, 22.0),
                ("greedy_novelty", 0.5, 20.0),
            ):
                directory = root / policy
                directory.mkdir()
                summary = {
                    "condition": f"online_construction_{policy}",
                    "task_ids": ["a", "b"],
                    "success_rate": success_rate,
                    "average_steps": steps,
                    "parameters": dict(common, schedule_policy=policy),
                    "online_construction_summary": {
                        "arrival_count": 3,
                        "construction_success_count": 2,
                        "construction_failure_count": 0,
                        "final_queue_length": 1,
                        "waiting_intervals_mean": 0.5,
                        "online_retrieval_count": 4,
                        "retrieved_constructed_memory_count": 2,
                    },
                }
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )

            comparison = maybe_write_online_construction_comparison(root)

            self.assertIsNotNone(comparison)
            self.assertAlmostEqual(
                comparison["greedy_minus_fifo_success_rate_percentage_points"],
                10.0,
            )
            self.assertEqual(comparison["greedy_minus_fifo_average_steps"], -2.0)

    def test_online_comparison_rejects_different_initial_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for policy, initial_ids in (
                ("fifo", ["mem_0000"]),
                ("greedy_novelty", ["mem_0001"]),
            ):
                directory = root / policy
                directory.mkdir()
                summary = {
                    "condition": f"online_construction_{policy}",
                    "task_ids": ["a"],
                    "success_rate": 0.5,
                    "average_steps": 20.0,
                    "parameters": {
                        "condition_mode": "online_construction",
                        "schedule_policy": policy,
                        "warm_start_count": 1,
                        "warm_start_seed": 7,
                        "warm_start_memory_file": "documents.json",
                        "initial_available_memory_ids": initial_ids,
                    },
                }
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "initial_available_memory_ids"):
                maybe_write_online_construction_comparison(root)

    def test_comparison_supports_novelty_sum_and_coverage_oracles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_parameters = {
                "condition_mode": "cloud_scheduled",
                "model": "model",
                "split": "valid_unseen",
                "seed": 42,
                "batch_size": 1,
                "max_steps": 30,
                "temperature": 0,
                "top_p": 1.0,
                "few_shot": True,
                "top_k": 3,
                "manifest_sha256": "manifest",
                "candidate_pool_sha256": "candidates",
                "interval_size": 10,
                "construction_capacity": 5,
                "scheduled_score_threshold": 0.5,
            }
            conditions = (
                ("random_1", "random", 0.25, 25.0, 1),
                ("random_2", "random", 0.50, 20.0, 2),
                ("greedy_novelty", "greedy_novelty", 0.60, 20.0, None),
                ("oracle_sum", "oracle_sum", 0.50, 19.0, None),
                ("oracle_coverage", "oracle_coverage", 0.75, 18.0, None),
            )
            for condition, policy, success_rate, average_steps, seed in conditions:
                condition_dir = root / condition
                condition_dir.mkdir()
                summary = {
                    "condition": condition,
                    "task_ids": ["task-a", "task-b"],
                    "success_rate": success_rate,
                    "average_steps": average_steps,
                    "parameters": dict(
                        common_parameters,
                        schedule_policy=policy,
                        scheduler_seed=seed,
                    ),
                }
                (condition_dir / "summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )

            comparison = maybe_write_scheduling_comparison(root)

            self.assertIsNotNone(comparison)
            self.assertEqual(len(comparison["oracle_runs"]), 2)
            self.assertEqual(len(comparison["novelty_runs"]), 1)
            self.assertAlmostEqual(
                comparison["novelty_runs"][0][
                    "minus_random_success_rate_percentage_points"
                ],
                22.5,
            )
            self.assertEqual(
                comparison[
                    "oracle_coverage_minus_oracle_sum_success_rate_percentage_points"
                ],
                25.0,
            )
            self.assertEqual(
                comparison["oracle_coverage_minus_oracle_sum_average_steps"],
                -1.0,
            )
            self.assertAlmostEqual(
                comparison[
                    "oracle_coverage_minus_greedy_novelty_success_rate_percentage_points"
                ],
                15.0,
            )

    def test_comparison_rejects_different_warm_start_pools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_parameters = {
                "condition_mode": "cloud_scheduled",
                "model": "model",
                "split": "valid_unseen",
                "seed": 42,
                "batch_size": 1,
                "max_steps": 30,
                "temperature": 0,
                "top_p": 1.0,
                "few_shot": True,
                "top_k": 3,
                "manifest_sha256": "manifest",
                "candidate_pool_sha256": "candidates",
                "interval_size": 10,
                "construction_capacity": 5,
                "scheduled_score_threshold": 0.5,
                "warm_start_count": 1,
                "warm_start_seed": 7,
            }
            conditions = (
                ("random", "random", ["mem_0000"], "pool-a"),
                ("coverage", "oracle_coverage", ["mem_0001"], "pool-b"),
            )
            for condition, policy, memory_ids, pool_hash in conditions:
                condition_dir = root / condition
                condition_dir.mkdir()
                parameters = dict(
                    common_parameters,
                    schedule_policy=policy,
                    scheduler_seed=1 if policy == "random" else None,
                    initial_available_memory_ids=memory_ids,
                    initial_available_pool_sha256=pool_hash,
                )
                summary = {
                    "condition": condition,
                    "task_ids": ["task-a"],
                    "success_rate": 0.5,
                    "average_steps": 20.0,
                    "parameters": parameters,
                }
                (condition_dir / "summary.json").write_text(
                    json.dumps(summary),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "warm-start memory pools"):
                maybe_write_scheduling_comparison(root)


class TaskManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary.name)
        self.gamefiles = []
        for index in range(6):
            path = (
                self.data_root
                / "json_2.1.1"
                / "valid_unseen"
                / "pick_and_place_simple"
                / f"trial_{index}"
                / "game.tw-pddl"
            )
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            self.gamefiles.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_seeded_manifest_is_deterministic_and_validates(self):
        first = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=17,
            limit_tasks=4,
        )
        second = build_task_manifest(
            list(reversed(self.gamefiles)),
            data_root=self.data_root,
            split="valid_unseen",
            seed=17,
            limit_tasks=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 4)
        selected = validate_task_manifest(
            first,
            split="valid_unseen",
            available_gamefiles=self.gamefiles,
            data_root=self.data_root,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(manifest_sha256(first)), 64)

    def test_explicit_task_ids_preserve_order(self):
        task_ids = [
            "json_2.1.1/valid_unseen/pick_and_place_simple/trial_4/game.tw-pddl",
            "json_2.1.1/valid_unseen/pick_and_place_simple/trial_1/game.tw-pddl",
        ]
        manifest = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=42,
            task_ids=task_ids,
        )
        self.assertEqual([task["task_id"] for task in manifest["tasks"]], task_ids)

    def test_manifest_rejects_wrong_split_and_missing_task(self):
        manifest = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=42,
            limit_tasks=2,
        )
        with self.assertRaisesRegex(ValueError, "Manifest split"):
            validate_task_manifest(
                manifest,
                split="valid_seen",
                available_gamefiles=self.gamefiles,
                data_root=self.data_root,
            )
        with self.assertRaisesRegex(ValueError, "missing from ALFWorld"):
            validate_task_manifest(
                manifest,
                split="valid_unseen",
                available_gamefiles=self.gamefiles[2:],
                data_root=self.data_root,
            )


class ResultSummaryTests(unittest.TestCase):
    @staticmethod
    def _results(condition="no_memory"):
        return [
            {
                "task_id": "task-1",
                "condition": condition,
                "split": "valid_unseen",
                "reward": True,
                "steps": 4,
                "termination_reason": "success",
                "error": None,
            },
            {
                "task_id": "task-2",
                "condition": condition,
                "split": "valid_unseen",
                "reward": False,
                "steps": 10,
                "termination_reason": "max_steps",
                "error": None,
            },
        ]

    def test_summary_metrics_and_files(self):
        summary = summarize_results(self._results())
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_steps"], 7)
        self.assertEqual(summary["average_success_steps"], 4)

        with tempfile.TemporaryDirectory() as directory:
            written = write_results(
                directory,
                self._results(),
                summary_metadata={"model": "test-model"},
            )
            self.assertEqual(written["model"], "test-model")
            self.assertTrue((Path(directory) / "results.jsonl").is_file())
            self.assertTrue((Path(directory) / "summary.csv").is_file())
            self.assertEqual(load_json(Path(directory) / "summary.json")["task_count"], 2)
            self.assertEqual(len(list((Path(directory) / "tasks").glob("*.json"))), 2)

    def test_paired_comparison_requires_identical_tasks_and_parameters(self):
        parameters = {
            "model": "model",
            "split": "valid_unseen",
            "seed": 42,
            "batch_size": 1,
            "max_steps": 30,
            "temperature": 1.0,
            "few_shot": True,
            "top_k": 3,
            "manifest_sha256": "hash",
        }
        no_memory = summarize_results(self._results("no_memory"))
        memory_results = self._results("memory")
        memory_results[1]["reward"] = True
        memory_results[1]["termination_reason"] = "success"
        memory = summarize_results(memory_results)
        no_memory["parameters"] = dict(parameters)
        memory["parameters"] = dict(parameters)

        comparison = build_paired_comparison(no_memory, memory)
        self.assertEqual(comparison["absolute_improvement"], 0.5)
        self.assertEqual(comparison["absolute_improvement_percentage_points"], 50)

        memory["parameters"]["temperature"] = 0.5
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_paired_comparison(no_memory, memory)

    def test_memory_retrieval_is_recorded_and_injected(self):
        records = retrieval_records(
            [
                (
                    FakeDocument(query="cool an apple", workflow="find a fridge", source="traj-1"),
                    0.25,
                )
            ]
        )
        self.assertEqual(records[0]["rank"], 1)
        self.assertEqual(records[0]["score"], 0.25)
        prompt = inject_memory("Your task is to: cool bread", records)
        self.assertIn("find a fridge", prompt)
        self.assertEqual(task_query("room\nYour task is to: cool bread"), "cool bread")

    def test_reranked_retrieval_keeps_vector_and_rerank_scores_distinct(self):
        records = reranked_retrieval_records(
            [
                {
                    "document": FakeDocument(
                        query="cool an apple",
                        workflow="find a fridge",
                        source="traj-1",
                    ),
                    "vector_rank": 7,
                    "vector_score": 0.42,
                    "rerank_rank": 1,
                    "rerank_score": 0.91,
                }
            ]
        )
        record = records[0]
        self.assertEqual(record["vector_rank"], 7)
        self.assertEqual(record["vector_score_type"], "faiss_l2_distance")
        self.assertFalse(record["vector_higher_is_better"])
        self.assertEqual(record["rerank_rank"], 1)
        self.assertEqual(record["rerank_score_type"], "openmem_relevance_score")
        self.assertTrue(record["rerank_higher_is_better"])
        self.assertIn("find a fridge", inject_memory("observation", records))

    def test_memory_rerank_comparison_counts_paired_flips(self):
        parameters = {
            "model": "model",
            "agent_api_base_url": "url",
            "split": "valid_unseen",
            "seed": 42,
            "batch_size": 2,
            "max_steps": 30,
            "temperature": 0,
            "top_p": 1.0,
            "few_shot": True,
            "manifest_sha256": "hash",
        }
        baseline_summary = {
            "condition": "memory",
            "split": "valid_unseen",
            "task_ids": ["a", "b", "c", "d"],
            "success_rate": 0.5,
            "parameters": dict(parameters, top_k=10),
            "retrieval_summary": {"similarity_search_latency_ms_mean": 20.0},
        }
        rerank_summary = {
            "condition": "memory_rerank",
            "split": "valid_unseen",
            "task_ids": ["a", "b", "c", "d"],
            "success_rate": 0.75,
            "parameters": dict(parameters, rerank_candidate_k=20, rerank_top_n=10),
            "rerank_summary": {"rerank_pipeline_latency_ms_mean": 400.0},
        }
        baseline_results = [
            {"task_id": "a", "reward": True},
            {"task_id": "b", "reward": True},
            {"task_id": "c", "reward": False},
            {"task_id": "d", "reward": False},
        ]
        rerank_results = [
            {"task_id": "a", "reward": True},
            {"task_id": "b", "reward": False},
            {"task_id": "c", "reward": True},
            {"task_id": "d", "reward": True},
        ]
        comparison = build_condition_comparison(
            baseline_summary,
            rerank_summary,
            baseline_results,
            rerank_results,
        )
        self.assertEqual(comparison["failure_to_success"], 2)
        self.assertEqual(comparison["success_to_failure"], 1)
        self.assertEqual(comparison["both_success"], 1)
        self.assertEqual(comparison["both_failure"], 0)
        self.assertEqual(comparison["absolute_improvement_percentage_points"], 25)
        self.assertEqual(comparison["rerank_added_latency_ms_mean"], 380.0)

    def test_raw_trajectory_retrieval_is_recorded_and_injected(self):
        records = retrieval_records(
            [
                (
                    FakeDocument(
                        memory_type="raw_trajectory",
                        query="put a clean potato in microwave",
                        trajectory=(
                            "Human:\nroom\n\nYour task is to: clean potato\n\n"
                            "Assistant:\nThought: find it\nAction: go to fridge 1"
                        ),
                        trajectory_index=7,
                        task_type="clean_then_place",
                        source="alfworld",
                    ),
                    0.125,
                )
            ]
        )
        self.assertEqual(records[0]["trajectory_index"], 7)
        self.assertEqual(records[0]["raw_score"], 0.125)
        self.assertFalse(records[0]["higher_is_better"])
        prompt = inject_trajectories("current observation", records)
        self.assertIn("Here are some trajectories for solving similar tasks:", prompt)
        self.assertIn('"task_name": "put a clean potato in microwave"', prompt)
        self.assertIn("Human:\nroom", prompt)
        self.assertNotIn("guidelines", prompt)


if __name__ == "__main__":
    unittest.main()
