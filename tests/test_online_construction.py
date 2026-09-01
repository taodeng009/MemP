import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ProcedureMem.online_construction import (
    FIFOScheduler,
    FIFOShortestFirstScheduler,
    OnlineConstructionController,
    OnlineConstructionQueue,
    OnlineTrajectoryCandidate,
    estimate_historical_utilities,
    load_warm_start_documents,
    oracle_future_query_window,
    parse_oracle_lookahead_horizon,
    trajectory_queue_id,
)
from ProcedureMem.cloud_scheduling import OracleExactRetrievalScheduler


class FakeEmbedding:
    vectors = {
        "available": [0.0],
        "near": [1.0],
        "far": [10.0],
        "next-far": [9.0],
        "task-a": [2.0],
        "task-b": [3.0],
    }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


class FakeMemory:
    retrieve_num = 3
    score_threshold = 0.5

    def __init__(self, *, fail_queries=(), documents=()):
        self.documents = list(documents)
        self.embedding = FakeEmbedding()
        self.cached_embedder = self.embedding
        self.fail_queries = set(fail_queries)
        self.save_count = 0
        self.rebuild_count = 0

    def build_document(self, item):
        if item["query"] in self.fail_queries:
            raise RuntimeError("builder unavailable")
        metadata = {
            "query": item["query"],
            "workflow": f"workflow for {item['query']}",
            "memory_id": item["memory_id"],
            **item["metadata"],
        }
        return SimpleNamespace(page_content=item["query"], metadata=metadata)

    def append_documents(self, documents):
        self.documents.extend(documents)

    def save_documents(self):
        self.save_count += 1

    def rebuild_index(self):
        self.rebuild_count += 1


def result(
    index, *, reward=True, query=None, action="go to table 1", steps=5
):
    query = query or f"task-{chr(ord('a') + index)}"
    return {
        "task_id": f"task/{index}",
        "task_index": index,
        "task_type": "pick_and_place_simple",
        "query": query,
        "reward": reward,
        "steps": steps,
        "trajectory": [
            {"from": "human", "value": f"Your task is to: {query}"},
            {"from": "gpt", "value": f"Action: {action}"},
            {"from": "human", "value": "Observation: done"},
        ],
    }


class QueueTests(unittest.TestCase):
    def test_oracle_horizon_accepts_any_positive_integer_and_all_remaining(self):
        self.assertEqual(parse_oracle_lookahead_horizon("1"), 1)
        self.assertEqual(parse_oracle_lookahead_horizon("5"), 5)
        self.assertEqual(
            parse_oracle_lookahead_horizon("all_remaining"), "all_remaining"
        )
        for invalid in ("0", "-1", "1.5", "everything"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_oracle_lookahead_horizon(invalid)

    def test_oracle_future_window_clamps_to_all_remaining(self):
        queries = tuple(f"q{index}" for index in range(24))

        fixed, fixed_horizon = oracle_future_query_window(
            queries,
            current_end=10,
            interval_size=5,
            requested_horizon=2,
        )
        clamped, clamped_horizon = oracle_future_query_window(
            queries,
            current_end=10,
            interval_size=5,
            requested_horizon=10,
        )
        all_remaining, all_horizon = oracle_future_query_window(
            queries,
            current_end=10,
            interval_size=5,
            requested_horizon="all_remaining",
        )

        self.assertEqual(fixed, tuple(f"q{index}" for index in range(10, 20)))
        self.assertEqual(fixed_horizon, 2)
        self.assertEqual(clamped, tuple(f"q{index}" for index in range(10, 24)))
        self.assertEqual(clamped_horizon, 3)
        self.assertEqual(all_remaining, clamped)
        self.assertEqual(all_horizon, 3)

    def test_fifo_preserves_arrival_order(self):
        selection = FIFOScheduler().select(["q2", "q1", "q3"], 2)
        self.assertEqual(selection.memory_ids, ("q2", "q1"))

    def test_same_query_different_trajectories_have_different_ids(self):
        first = result(0, query="same")
        second = result(0, query="same", action="go to counter 1")
        first_id = trajectory_queue_id(
            task_id=first["task_id"],
            task_index=first["task_index"],
            trajectory=first["trajectory"],
        )
        second_id = trajectory_queue_id(
            task_id=second["task_id"],
            task_index=second["task_index"],
            trajectory=second["trajectory"],
        )
        self.assertNotEqual(first_id, second_id)

    def test_shortest_first_preserves_interval_fifo(self):
        candidates = {
            "new_short": SimpleNamespace(
                arrival_interval=1, steps=1, task_index=3
            ),
            "old_long": SimpleNamespace(
                arrival_interval=0, steps=10, task_index=0
            ),
            "old_short_later": SimpleNamespace(
                arrival_interval=0, steps=4, task_index=2
            ),
            "old_short_earlier": SimpleNamespace(
                arrival_interval=0, steps=4, task_index=1
            ),
        }

        selection = FIFOShortestFirstScheduler().select(
            candidates,
            3,
            candidates=candidates,
        )

        self.assertEqual(
            selection.memory_ids,
            ("old_short_earlier", "old_short_later", "old_long"),
        )

    def test_unknown_queue_item_is_rejected(self):
        queue = OnlineConstructionQueue()
        with self.assertRaises(KeyError):
            queue.get("missing")


class ControllerTests(unittest.TestCase):
    def test_historical_counters_deduplicate_each_task(self):
        warm = SimpleNamespace(
            page_content="available",
            metadata={"memory_id": "warm_0", "query": "available"},
        )
        controller = OnlineConstructionController(
            memory=FakeMemory(documents=[warm]), policy="fifo", capacity=1
        )

        controller.record_retrieval_outcomes(
            [
                {
                    "retrieved_memory_ids": ["warm_0", "warm_0"],
                    "reward": True,
                },
                {"retrieved_memory_ids": ["warm_0"], "reward": False},
                {"retrieved_memory_ids": [], "reward": True},
            ]
        )

        self.assertEqual(
            controller.historical_memory_stats["warm_0"],
            {"retrieval_count": 2, "success_count": 1},
        )

    def test_warm_start_enters_historical_pool_at_five_retrievals(self):
        warm = SimpleNamespace(
            page_content="available",
            metadata={"memory_id": "warm_0", "query": "available"},
        )
        controller = OnlineConstructionController(
            memory=FakeMemory(documents=[warm]),
            policy="oracle_exact_retrieval_historical_utility",
            capacity=1,
        )
        controller.record_retrieval_outcomes(
            [
                {"retrieved_memory_ids": ["warm_0"], "reward": index != 0}
                for index in range(4)
            ]
        )
        self.assertEqual(controller._historical_references(), ({}, {}))

        controller.record_retrieval_outcomes(
            [{"retrieved_memory_ids": ["warm_0"], "reward": True}]
        )
        queries, utilities = controller._historical_references()

        self.assertEqual(queries, {"warm_0": "available"})
        self.assertEqual(utilities, {"warm_0": 0.8})

        controller.admit_results(
            [result(0, query="near"), result(1, query="next-far")],
            interval_id=0,
        )
        controller.construct(interval_id=0, future_queries=["next-far"])
        construction = controller.construction_events[0]

        self.assertEqual(construction["historical_reference_count"], 1)
        self.assertAlmostEqual(construction["historical_utility_estimate"], 0.8)
        self.assertGreater(construction["adjusted_score"], 0.0)

    def test_only_successes_arrive_and_unselected_item_persists(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="fifo", capacity=1
        )
        arrived = controller.admit_results(
            [result(0), result(1, reward=False), result(1)], interval_id=0
        )

        self.assertEqual(len(arrived), 2)
        event = controller.construct(interval_id=0)

        self.assertEqual(len(event["selected_queue_ids"]), 1)
        self.assertEqual(len(controller.queue), 1)
        self.assertEqual(len(controller.staged_documents), 1)
        remaining_id = controller.queue.pending_ids[0]
        self.assertIn(remaining_id, arrived)

    def test_staged_document_activates_only_in_next_interval(self):
        memory = FakeMemory()
        controller = OnlineConstructionController(
            memory=memory, policy="fifo", capacity=1
        )
        controller.admit_results([result(0)], interval_id=0)
        controller.construct(interval_id=0)

        self.assertEqual(memory.documents, [])
        with self.assertRaises(ValueError):
            controller.activate_staged(interval_id=0)
        activated = controller.activate_staged(interval_id=1)

        self.assertEqual(len(activated), 1)
        self.assertEqual(len(memory.documents), 1)
        self.assertEqual(memory.save_count, 1)
        self.assertEqual(memory.rebuild_count, 1)

    def test_builder_failure_is_logged_and_kept_pending(self):
        memory = FakeMemory(fail_queries={"task-a"})
        controller = OnlineConstructionController(
            memory=memory, policy="fifo", capacity=1
        )
        arrived = controller.admit_results([result(0)], interval_id=2)
        event = controller.construct(interval_id=2)

        self.assertEqual(event["selected_queue_ids"], arrived)
        self.assertEqual(len(controller.queue), 1)
        self.assertEqual(
            controller.construction_events[0]["construction_result"], "failure"
        )
        self.assertIn("builder unavailable", controller.construction_events[0]["error"])

    def test_waiting_time_uses_arrival_interval(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="fifo", capacity=1
        )
        controller.admit_results([result(0)], interval_id=1)
        controller.construct(interval_id=3)
        self.assertEqual(controller.construction_events[0]["waiting_intervals"], 2)

    def test_greedy_novelty_selects_farthest_from_available(self):
        available = SimpleNamespace(
            page_content="available",
            metadata={"memory_id": "warm_0", "query": "available"},
        )
        controller = OnlineConstructionController(
            memory=FakeMemory(documents=[available]),
            policy="greedy_novelty",
            capacity=1,
        )
        controller.admit_results(
            [result(0, query="near"), result(1, query="far")], interval_id=0
        )
        event = controller.construct(interval_id=0)
        selected = event["selected_queue_ids"][0]
        self.assertEqual(
            controller.construction_events[0]["source_task_id"], "task/1"
        )
        self.assertNotIn(selected, controller.queue.pending_ids)

    def test_fifo_shortest_first_prefers_shorter_within_same_interval(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="fifo_shortest_first", capacity=2
        )
        arrived = controller.admit_results(
            [
                result(0, steps=9),
                result(1, steps=4),
                result(2, steps=6),
            ],
            interval_id=0,
        )

        event = controller.construct(interval_id=0)

        self.assertEqual(
            event["selected_queue_ids"],
            [arrived[1], arrived[2]],
        )
        self.assertEqual(
            [item["source_steps"] for item in event["construction_results"]],
            [4, 6],
        )
        self.assertEqual(controller.queue.pending_ids, (arrived[0],))

    def test_oracle_coverage_selects_best_next_interval_coverage(self):
        available = SimpleNamespace(
            page_content="available",
            metadata={"memory_id": "warm_0", "query": "available"},
        )
        controller = OnlineConstructionController(
            memory=FakeMemory(documents=[available]),
            policy="oracle_coverage",
            capacity=1,
        )
        controller.admit_results(
            [result(0, query="near"), result(1, query="far")], interval_id=0
        )

        event = controller.construct(
            interval_id=0,
            next_interval_queries=["next-far"],
        )

        self.assertEqual(controller.construction_events[0]["source_task_id"], "task/1")
        self.assertEqual(event["oracle_next_interval_query_count"], 1)
        selected_id = event["selected_queue_ids"][0]
        self.assertEqual(
            event["oracle_scores"][selected_id]["score_type"],
            "faiss_l2_marginal_gain",
        )
        self.assertIsNotNone(
            controller.construction_events[0]["oracle_score"]
        )

    def test_oracle_coverage_requires_next_interval_queries(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="oracle_coverage", capacity=1
        )
        controller.admit_results([result(0)], interval_id=0)

        with self.assertRaisesRegex(ValueError, "next-interval"):
            controller.construct(interval_id=0)

    def test_exact_retrieval_oracle_records_horizon_and_retrieval_config(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="oracle_exact_retrieval", capacity=1
        )
        controller.admit_results(
            [result(0, query="near"), result(1, query="next-far")],
            interval_id=0,
        )

        event = controller.construct(
            interval_id=0,
            future_queries=["next-far"],
            requested_lookahead_horizon=5,
            effective_lookahead_horizon=1,
            future_interval_count=1,
        )

        self.assertEqual(controller.construction_events[0]["source_task_id"], "task/1")
        self.assertEqual(event["oracle_requested_lookahead_horizon"], 5)
        self.assertEqual(event["oracle_effective_lookahead_horizon"], 1)
        self.assertEqual(event["oracle_future_query_count"], 1)
        self.assertEqual(event["oracle_retrieval_top_k"], 3)
        self.assertEqual(event["oracle_retrieval_threshold"], 0.5)
        selected_id = event["selected_queue_ids"][0]
        self.assertEqual(
            event["oracle_scores"][selected_id]["score_type"],
            "faiss_squared_l2_topk_threshold_marginal_gain",
        )

    def test_final_interval_records_backlog_without_building(self):
        controller = OnlineConstructionController(
            memory=FakeMemory(), policy="fifo", capacity=1
        )
        arrived = controller.admit_results([result(0)], interval_id=4)
        event = controller.record_final_queue(interval_id=4)
        event["arrived_queue_ids"] = arrived

        self.assertEqual(event["selected_queue_ids"], [])
        self.assertEqual(len(controller.construction_events), 0)
        self.assertEqual(len(controller.queue), 1)

    def test_zero_capacity_admits_trajectories_without_construction(self):
        memory = FakeMemory()
        controller = OnlineConstructionController(
            memory=memory, policy="greedy_novelty", capacity=0
        )
        arrived = controller.admit_results([result(0), result(1)], interval_id=0)

        event = controller.construct(interval_id=0)

        self.assertEqual(len(arrived), 2)
        self.assertEqual(event["selected_queue_ids"], [])
        self.assertEqual(event["pending_queue_ids_after_construction"], arrived)
        self.assertEqual(len(controller.queue), 2)
        self.assertEqual(controller.construction_events, [])
        self.assertEqual(controller.staged_documents, [])
        self.assertEqual(memory.documents, [])


class WarmStartTests(unittest.TestCase):
    def test_same_file_count_and_seed_load_same_initial_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "documents.json"
            documents = [
                {
                    "page_content": f"query-{index}",
                    "metadata": {
                        "query": f"query-{index}",
                        "workflow": f"workflow-{index}",
                    },
                }
                for index in range(5)
            ]
            path.write_text(json.dumps(documents), encoding="utf-8")

            first_documents, first_ids = load_warm_start_documents(
                path, count=2, seed=7
            )
            second_documents, second_ids = load_warm_start_documents(
                path, count=2, seed=7
            )

            self.assertEqual(first_ids, second_ids)
            self.assertEqual(
                [document.metadata["memory_id"] for document in first_documents],
                [document.metadata["memory_id"] for document in second_documents],
            )
            self.assertTrue(
                all(
                    document.metadata["memory_origin"] == "warm_start"
                    for document in first_documents
                )
            )


class HistoricalUtilityTests(unittest.TestCase):
    def test_distance_weighted_historical_utility(self):
        estimates = estimate_historical_utilities(
            {"pending": "near"},
            {"left": "available", "right": "task-a"},
            {"left": 1.0, "right": 0.0},
            FakeEmbedding(),
        )

        self.assertAlmostEqual(estimates["pending"], 0.5)

    def test_lambda_zero_matches_exact_retrieval(self):
        distances = {
            "available": (0.4,),
            "candidate_a": (0.1,),
            "candidate_b": (0.2,),
        }

        def distance_scorer(_queries, requested_ids):
            return {memory_id: distances[memory_id] for memory_id in requested_ids}

        scheduler = OracleExactRetrievalScheduler()
        baseline = scheduler.select(
            ["candidate_a", "candidate_b"],
            2,
            available_ids=["available"],
            future_queries=["future"],
            distance_scorer=distance_scorer,
            top_k=1,
            score_threshold=0.5,
        )
        corrected = scheduler.select(
            ["candidate_a", "candidate_b"],
            2,
            available_ids=["available"],
            future_queries=["future"],
            distance_scorer=distance_scorer,
            top_k=1,
            score_threshold=0.5,
            historical_utility_estimates={
                "candidate_a": 0.0,
                "candidate_b": 1.0,
            },
            historical_reference_count=1,
            historical_utility_lambda=0.0,
        )

        self.assertEqual(corrected.memory_ids, baseline.memory_ids)


if __name__ == "__main__":
    unittest.main()
