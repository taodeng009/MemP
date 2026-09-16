import json
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.historical_same_state_diagnostic import (
    aggregate_retrieval_metrics,
    analyze_same_state,
    load_same_state_snapshot,
    snapshot_intervals,
    source_construction_capacity,
)


class FakeEmbedding:
    def __init__(self, values):
        self.values = values

    def embed_documents(self, texts):
        return [[self.values[text]] for text in texts]


class HistoricalSameStateDiagnosticTests(unittest.TestCase):
    def test_loads_fixed_history_memory_queue_and_future_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "parameters": {
                            "condition_mode": "online_construction",
                            "warm_start_count": 0,
                            "construction_capacity": 3,
                        }
                    }
                ),
                encoding="utf-8",
            )

            def write_jsonl(name, rows):
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            write_jsonl(
                "results.jsonl",
                [
                    {
                        "task_id": "h0",
                        "task_index": 0,
                        "query": "history-0",
                        "interval_id": 0,
                    },
                    {
                        "task_id": "h1",
                        "task_index": 1,
                        "query": "history-1",
                        "interval_id": 1,
                    },
                    {
                        "task_id": "f2",
                        "task_index": 2,
                        "query": "future-2",
                        "interval_id": 2,
                    },
                ],
            )
            write_jsonl(
                "online_trajectories.jsonl",
                [
                    {
                        "queue_id": "base",
                        "task_index": 0,
                        "query": "history-0",
                    },
                    {
                        "queue_id": "pending",
                        "task_index": 1,
                        "query": "history-1",
                    },
                ],
            )
            write_jsonl(
                "queue_events.jsonl",
                [
                    {
                        "interval_id": 1,
                        "pending_queue_ids_before_selection": ["pending"],
                    }
                ],
            )
            write_jsonl(
                "construction_events.jsonl",
                [
                    {
                        "queue_id": "base",
                        "construction_result": "success",
                        "constructed_memory_id": "online_base",
                        "available_from_interval": 1,
                    }
                ],
            )

            snapshot = load_same_state_snapshot(root, 1)

            self.assertEqual(len(snapshot["history"]), 2)
            self.assertEqual(
                snapshot["available_memories"][0]["source_task_index"], 0
            )
            self.assertEqual(
                snapshot["pending_candidates"][0]["source_task_index"], 1
            )
            self.assertEqual(snapshot["future_tasks"][0]["task_index"], 2)
            self.assertEqual(snapshot_intervals(root), [1])
            self.assertEqual(source_construction_capacity(root), 3)

    def test_empty_available_pool_uses_bootstrap_and_distance_recovery(self):
        snapshot = {
            "snapshot_interval": 0,
            "history": [
                {"task_index": 0, "query": "h0"},
                {"task_index": 1, "query": "h1"},
            ],
            "available_memories": [],
            "pending_candidates": [
                {
                    "memory_id": "a",
                    "source_task_index": 0,
                    "query": "h1",
                },
                {
                    "memory_id": "b",
                    "source_task_index": 1,
                    "query": "h0",
                },
            ],
            "future_tasks": [{"task_index": 2, "query": "h1"}],
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding({"h0": 0.0, "h1": 2.0}),
            capacity=1,
        )

        self.assertIsNone(
            report["realized_future_coverage_gain"]["future_oracle"]
        )
        self.assertEqual(
            report["realized_future_nearest_distance_sum"]["future_oracle"],
            0.0,
        )
        self.assertIn(
            "historical_bootstrap_distance_sum",
            report["first_step_scores"]["a"],
        )
        self.assertEqual(
            report["realized_future_retrieval_metrics"]["future_oracle"],
            {
                "task_count": 1,
                "hit_count": 1,
                "hr": 1.0,
                "bd": 0.0,
                "ru": 0.5,
                "best_distance_sum_hit": 0.0,
                "retrieval_utility_sum": 0.5,
            },
        )

    def test_reports_overlap_ranking_and_oracle_gain_recovery(self):
        snapshot = {
            "snapshot_interval": 1,
            "history": [
                {"task_index": 0, "query": "h0"},
                {"task_index": 1, "query": "h1"},
                {"task_index": 2, "query": "h2"},
            ],
            "available_memories": [
                {"memory_id": "m", "source_task_index": 0, "query": "h0"}
            ],
            "pending_candidates": [
                {
                    "memory_id": "a",
                    "source_task_index": 0,
                    "query": "h1",
                },
                {
                    "memory_id": "b",
                    "source_task_index": 1,
                    "query": "h0",
                },
                {
                    "memory_id": "c",
                    "source_task_index": 2,
                    "query": "cquery",
                },
            ],
            "future_tasks": [{"task_index": 3, "query": "future"}],
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding(
                {
                    "h0": 0.0,
                    "h1": 10.0,
                    "h2": 5.0,
                    "cquery": 20.0,
                    "future": 10.0,
                }
            ),
            capacity=1,
        )

        self.assertEqual(report["selections"]["fifo"], ["a"])
        self.assertEqual(
            report["selections"]["historical_cross_task"], ["b"]
        )
        self.assertEqual(report["selections"]["future_oracle"], ["a"])
        self.assertEqual(
            report["top_cc_overlap"]["historical_vs_oracle_fraction"], 0.0
        )
        self.assertEqual(
            report["realized_future_coverage_gain"]["future_oracle"], 100.0
        )
        self.assertEqual(report["oracle_gain_recovery"]["historical"], 0.0)
        self.assertIsNotNone(
            report["historical_vs_oracle_first_step_spearman"]
        )
        self.assertEqual(
            report["realized_future_retrieval_metrics"]["historical_cross_task"][
                "hr"
            ],
            0.0,
        )
        self.assertIsNone(
            report["realized_future_retrieval_metrics"]["historical_cross_task"][
                "bd"
            ]
        )
        self.assertEqual(
            report["realized_future_retrieval_metrics"]["future_oracle"]["ru"],
            0.5,
        )

    def test_exact_retrieval_policy_is_selected_by_option(self):
        snapshot = {
            "snapshot_interval": 1,
            "history": [
                {"task_index": 0, "query": "h0"},
                {"task_index": 1, "query": "h1"},
                {"task_index": 2, "query": "h2"},
            ],
            "available_memories": [
                {"memory_id": "m", "source_task_index": 0, "query": "h0"}
            ],
            "pending_candidates": [
                {"memory_id": "a", "source_task_index": 0, "query": "h1"},
                {"memory_id": "b", "source_task_index": 1, "query": "h0"},
                {"memory_id": "c", "source_task_index": 2, "query": "far"},
            ],
            "future_tasks": [{"task_index": 3, "query": "h1"}],
            "source_parameters": {"top_k": 3, "score_threshold": 0.5},
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding(
                {"h0": 0.0, "h1": 10.0, "h2": 5.0, "far": 20.0}
            ),
            capacity=1,
            historical_policy="historical_cross_task_exact_retrieval",
        )

        self.assertEqual(
            report["historical_policy"],
            "historical_cross_task_exact_retrieval",
        )
        self.assertEqual(report["future_oracle_policy"], "oracle_exact_retrieval")
        self.assertEqual(report["selections"]["historical_cross_task"], ["a"])
        self.assertEqual(report["selections"]["future_oracle"], ["a"])
        self.assertEqual(
            report["first_step_scores"]["a"]["historical"][
                "retrieval_utility_gain"
            ],
            0.5,
        )

    def test_hit_quality_policy_uses_alpha_and_metric_options(self):
        snapshot = {
            "snapshot_interval": 0,
            "history": [
                {"task_index": 0, "query": "zero"},
                {"task_index": 1, "query": "one"},
            ],
            "available_memories": [],
            "pending_candidates": [
                {
                    "memory_id": "a",
                    "source_task_index": 0,
                    "query": "one",
                },
                {
                    "memory_id": "b",
                    "source_task_index": 1,
                    "query": "zero",
                },
            ],
            "future_tasks": [{"task_index": 2, "query": "zero"}],
            "source_parameters": {"top_k": 1, "score_threshold": 0.5},
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding({"zero": 0.0, "one": 1.0}),
            capacity=1,
            historical_policy="historical_cross_task_hit_quality",
            hit_quality_alpha=0.25,
            hit_quality_metric="ru",
        )

        self.assertEqual(report["future_oracle_policy"], "oracle_hit_quality")
        self.assertEqual(report["hit_quality_alpha"], 0.25)
        self.assertEqual(report["hit_quality_metric"], "ru")
        detail = report["first_step_scores"]["a"]["historical"]
        self.assertFalse(detail["coverage_bootstrap"])
        self.assertIn("hit_gain", detail)
        self.assertIn("quality_gain", detail)
        self.assertIn("priority", detail)

    def test_available_memory_is_excluded_from_its_own_source_task(self):
        snapshot = {
            "snapshot_interval": 1,
            "history": [
                {"task_index": 0, "query": "same"},
                {"task_index": 1, "query": "other"},
            ],
            "available_memories": [
                {
                    "memory_id": "self-memory",
                    "source_task_index": 0,
                    "query": "same",
                }
            ],
            "pending_candidates": [
                {
                    "memory_id": "cross",
                    "source_task_index": 1,
                    "query": "same",
                }
            ],
            "future_tasks": [{"task_index": 2, "query": "other"}],
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding({"same": 0.0, "other": 2.0}),
            capacity=1,
        )

        score = report["first_step_scores"]["cross"]
        self.assertEqual(score["historical_newly_covered"], 1.0)

    def test_retrieval_metrics_use_source_top_k_threshold_and_pool_weighting(self):
        snapshot = {
            "snapshot_interval": 0,
            "history": [{"task_index": 0, "query": "zero"}],
            "available_memories": [],
            "pending_candidates": [
                {
                    "memory_id": "a",
                    "source_task_index": 0,
                    "query": "zero",
                },
                {
                    "memory_id": "b",
                    "source_task_index": 0,
                    "query": "half",
                },
            ],
            "future_tasks": [
                {"task_index": 1, "query": "zero"},
                {"task_index": 2, "query": "one"},
            ],
            "source_parameters": {"top_k": 2, "score_threshold": 1.0},
        }
        report = analyze_same_state(
            snapshot,
            FakeEmbedding({"zero": 0.0, "half": 0.5, "one": 1.0}),
            capacity=2,
        )

        metrics = report["realized_future_retrieval_metrics"]["fifo"]
        self.assertEqual(report["future_retrieval_config"]["top_k"], 2)
        self.assertEqual(metrics["hit_count"], 2)
        self.assertEqual(metrics["hr"], 1.0)
        self.assertEqual(metrics["bd"], 0.125)
        self.assertEqual(metrics["ru"], 1.25)

        overall = aggregate_retrieval_metrics([report, report])["fifo"]
        self.assertEqual(overall["task_count"], 4)
        self.assertEqual(overall["hit_count"], 4)
        self.assertEqual(overall["hr"], 1.0)
        self.assertEqual(overall["bd"], 0.125)
        self.assertEqual(overall["ru"], 1.25)


if __name__ == "__main__":
    unittest.main()
