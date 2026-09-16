import json
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.historical_same_state_diagnostic import (
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


if __name__ == "__main__":
    unittest.main()
