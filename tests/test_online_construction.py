import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ProcedureMem.online_construction import (
    FIFOScheduler,
    OnlineConstructionController,
    OnlineConstructionQueue,
    OnlineTrajectoryCandidate,
    load_warm_start_documents,
    trajectory_queue_id,
)


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


def result(index, *, reward=True, query=None, action="go to table 1"):
    query = query or f"task-{chr(ord('a') + index)}"
    return {
        "task_id": f"task/{index}",
        "task_index": index,
        "task_type": "pick_and_place_simple",
        "query": query,
        "reward": reward,
        "trajectory": [
            {"from": "human", "value": f"Your task is to: {query}"},
            {"from": "gpt", "value": f"Action: {action}"},
            {"from": "human", "value": "Observation: done"},
        ],
    }


class QueueTests(unittest.TestCase):
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

    def test_unknown_queue_item_is_rejected(self):
        queue = OnlineConstructionQueue()
        with self.assertRaises(KeyError):
            queue.get("missing")


class ControllerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
