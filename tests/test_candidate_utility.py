import json
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.candidate_utility import (
    condition_memory_ids,
    coverage_proxy_scores,
    load_snapshot,
    stratified_proxy_selection,
    summarize_baseline_stability,
    summarize_candidate_utility,
    validate_workflow_cache,
)


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

    def embed_documents(self, texts):
        return [[self.values[text]] for text in texts]


class CandidateUtilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)
        summary = {
            "parameters": {
                "condition_mode": "online_construction",
                "warm_start_count": 0,
                "split": "train",
                "seed": 42,
            }
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        results = [
            {
                "task_id": "json_2.1.1/train/type/trial_a/game.tw-pddl",
                "task_index": 40,
                "interval_id": 2,
                "query": "future",
            },
            {
                "task_id": "json_2.1.1/train/type/trial_b/game.tw-pddl",
                "task_index": 41,
                "interval_id": 2,
                "query": "future",
            },
        ]
        trajectories = [
            {
                "queue_id": "base",
                "task_id": "base-task",
                "task_index": 0,
                "task_type": "type",
                "query": "base-query",
                "trajectory": [{"from": "human", "value": "base"}],
                "steps": 4,
                "arrival_interval": 0,
            }
        ]
        for index in range(6):
            trajectories.append(
                {
                    "queue_id": f"p{index}",
                    "task_id": f"pending-{index}",
                    "task_index": index + 1,
                    "task_type": "type",
                    "query": f"candidate-{index}",
                    "trajectory": [
                        {"from": "human", "value": f"trajectory-{index}"}
                    ],
                    "steps": index + 5,
                    "arrival_interval": 1,
                }
            )
        queue_events = [
            {
                "interval_id": 1,
                "pending_queue_ids_before_selection": [f"p{i}" for i in range(6)],
            }
        ]
        construction_events = [
            {
                "interval_id": 0,
                "queue_id": "base",
                "construction_result": "success",
                "workflow": "base workflow",
                "constructed_memory_id": "online_base",
                "available_from_interval": 1,
            },
            {
                "interval_id": 1,
                "queue_id": "p0",
                "construction_result": "success",
                "workflow": "not yet available",
                "constructed_memory_id": "online_p0",
                "available_from_interval": 2,
            },
        ]
        self._write_jsonl("results.jsonl", results)
        self._write_jsonl("online_trajectories.jsonl", trajectories)
        self._write_jsonl("queue_events.jsonl", queue_events)
        self._write_jsonl("construction_events.jsonl", construction_events)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_jsonl(self, name, rows):
        (self.run_dir / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_snapshot_uses_available_boundary_pending_queue_and_next_tasks(self):
        snapshot = load_snapshot(self.run_dir, 1)

        self.assertEqual(
            [row["memory_id"] for row in snapshot["baseline_memories"]],
            ["online_base"],
        )
        self.assertEqual(snapshot["pending_queue_ids"], [f"p{i}" for i in range(6)])
        self.assertEqual(
            snapshot["downstream_task_ids"],
            [
                "json_2.1.1/train/type/trial_a/game.tw-pddl",
                "json_2.1.1/train/type/trial_b/game.tw-pddl",
            ],
        )
        self.assertEqual(
            [task["source_task_index"] for task in snapshot["task_manifest"]["tasks"]],
            [40, 41],
        )

    def test_proxy_stratification_selects_high_medium_and_low(self):
        snapshot = load_snapshot(self.run_dir, 1)
        embedding = FakeEmbedding(
            {
                "base-query": 0,
                "candidate-0": 10,
                "candidate-1": 9,
                "candidate-2": 7,
                "candidate-3": 5,
                "candidate-4": 1,
                "candidate-5": 0,
                "future": 10,
            }
        )
        scores = coverage_proxy_scores(snapshot, embedding)
        selected, annotated = stratified_proxy_selection(scores)

        by_stratum = {
            name: [row["queue_id"] for row in selected if row["proxy_stratum"] == name]
            for name in ("high", "medium", "low")
        }
        self.assertEqual(by_stratum["high"], ["p0", "p1"])
        self.assertEqual(by_stratum["medium"], ["p2", "p3"])
        self.assertEqual(set(by_stratum["low"]), {"p4", "p5"})
        self.assertEqual(sum(bool(row["selected"]) for row in annotated), 6)
        self.assertEqual(scores[-1]["coverage_proxy_gain"], 0.0)

    def test_workflow_cache_must_cover_selected_ids(self):
        rows = [
            {
                "queue_id": "p0",
                "construction_result": "success",
                "workflow": "workflow",
            }
        ]
        self.assertEqual(
            list(validate_workflow_cache(rows, ["p0"])),
            ["p0"],
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_workflow_cache(rows, ["p0", "p1"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_workflow_cache(rows + rows, ["p0"])

    def test_proxy_strata_do_not_overlap_when_all_gains_are_zero(self):
        rows = [
            {
                "queue_id": f"p{index}",
                "pending_order": index,
                "coverage_proxy_gain": 0.0,
            }
            for index in range(6)
        ]

        selected, _ = stratified_proxy_selection(rows)

        self.assertEqual(len({row["queue_id"] for row in selected}), 6)
        self.assertEqual(
            [row["proxy_stratum"] for row in selected],
            ["high", "high", "medium", "medium", "low", "low"],
        )

    def test_condition_memory_differs_by_exactly_one_candidate(self):
        self.assertEqual(
            condition_memory_ids(["m0", "m1"], "probe_p0"),
            ["m0", "m1", "probe_p0"],
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            condition_memory_ids(["m0"], "m0")

    def test_utility_and_retrieval_exposure_are_paired_by_task(self):
        baseline = [
            {"task_id": "a", "reward": False},
            {"task_id": "b", "reward": True},
            {"task_id": "c", "reward": False},
            {"task_id": "d", "reward": True},
        ]
        candidate = [
            {"task_id": "a", "reward": True, "retrieved_memory_ids": ["probe"]},
            {"task_id": "b", "reward": False, "retrieved_memory_ids": ["probe"]},
            {"task_id": "c", "reward": True, "retrieved_memory_ids": []},
            {"task_id": "d", "reward": True, "retrieved_memory_ids": []},
        ]

        summary = summarize_candidate_utility(
            baseline,
            candidate,
            candidate_memory_id="probe",
            candidate_metadata={"queue_id": "p0"},
        )

        self.assertEqual(summary["utility"], 1)
        self.assertEqual(summary["gained_task_ids"], ["a", "c"])
        self.assertEqual(summary["lost_task_ids"], ["b"])
        self.assertEqual(summary["gained_retrieved_task_ids"], ["a"])
        self.assertEqual(summary["lost_retrieved_task_ids"], ["b"])
        self.assertEqual(summary["gained_unretrieved_task_ids"], ["c"])

    def test_baseline_stability_separates_same_and_changed_retrieval_flips(self):
        repeats = [
            [
                {
                    "task_id": "a",
                    "reward": True,
                    "steps": 5,
                    "query": "qa",
                    "retrieved_memory_ids": ["m1"],
                },
                {
                    "task_id": "b",
                    "reward": False,
                    "steps": 30,
                    "query": "qb",
                    "retrieved_memory_ids": ["m2"],
                },
            ],
            [
                {
                    "task_id": "a",
                    "reward": False,
                    "steps": 30,
                    "query": "qa",
                    "retrieved_memory_ids": ["m1"],
                },
                {
                    "task_id": "b",
                    "reward": False,
                    "steps": 30,
                    "query": "qb",
                    "retrieved_memory_ids": ["other"],
                },
            ],
            [
                {
                    "task_id": "a",
                    "reward": True,
                    "steps": 6,
                    "query": "qa",
                    "retrieved_memory_ids": ["m1"],
                },
                {
                    "task_id": "b",
                    "reward": True,
                    "steps": 8,
                    "query": "qb",
                    "retrieved_memory_ids": ["m2"],
                },
            ],
        ]

        summary = summarize_baseline_stability(repeats)

        self.assertEqual(summary["repeat_count"], 3)
        self.assertEqual(
            [row["success_count"] for row in summary["repeat_summaries"]],
            [1, 0, 2],
        )
        self.assertEqual(summary["unstable_task_count"], 2)
        self.assertEqual(summary["total_pairwise_comparisons"], 6)
        self.assertEqual(summary["total_pairwise_flips"], 4)
        self.assertEqual(
            summary["pairwise"]["same_retrieval"]["comparisons"], 4
        )
        self.assertEqual(summary["pairwise"]["same_retrieval"]["flips"], 3)
        self.assertEqual(
            summary["pairwise"]["changed_retrieval"]["comparisons"], 2
        )
        self.assertEqual(summary["pairwise"]["changed_retrieval"]["flips"], 1)
        self.assertEqual(summary["retrieval_unstable_task_ids"], ["b"])


if __name__ == "__main__":
    unittest.main()
