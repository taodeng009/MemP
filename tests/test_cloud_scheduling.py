import unittest

from ProcedureMem.cloud_scheduling import (
    CandidateMemory,
    GreedyNoveltyScheduler,
    OracleCoverageScheduler,
    OracleHighScheduler,
    RandomScheduler,
    ScheduledWorkflowMemory,
    build_interval_batches,
    memory_id_pool_sha256,
    select_warm_start_ids,
)


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = list(documents)

    def similarity_search_with_score(self, query, **kwargs):
        matches = [
            document
            for document in self.documents
            if document.page_content == query
        ]
        return [(document, 0.0) for document in matches[: kwargs["k"]]]


def fake_store_factory(documents, embedding):
    return FakeVectorStore(documents)


class FakeEmbedding:
    vectors = {
        "query-0": [0.0],
        "query-1": [10.0],
        "query-2": [20.0],
        "future-a": [0.0],
        "future-b": [20.0],
    }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


def candidates(count=4):
    return [
        CandidateMemory(
            memory_id=f"mem_{index:04d}",
            query=f"query-{index}",
            workflow=f"workflow-{index}",
        )
        for index in range(count)
    ]


class WarmStartPoolTests(unittest.TestCase):
    def test_selection_is_deterministic_and_keeps_candidate_order(self):
        ids = [item.memory_id for item in candidates(10)]

        first = select_warm_start_ids(ids, count=4, seed=17)
        second = select_warm_start_ids(ids, count=4, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(
            first,
            tuple(memory_id for memory_id in ids if memory_id in first),
        )
        self.assertEqual(
            memory_id_pool_sha256(first),
            memory_id_pool_sha256(second),
        )

    def test_zero_count_is_cold_start_and_invalid_counts_are_rejected(self):
        ids = [item.memory_id for item in candidates(3)]

        self.assertEqual(select_warm_start_ids(ids, count=0, seed=17), ())
        with self.assertRaises(ValueError):
            select_warm_start_ids(ids, count=-1, seed=17)
        with self.assertRaises(ValueError):
            select_warm_start_ids(ids, count=4, seed=17)
        with self.assertRaises(ValueError):
            select_warm_start_ids(["mem_0000", "mem_0000"], count=1, seed=17)

    def test_initial_memories_are_retrievable_and_removed_from_pending(self):
        memory = ScheduledWorkflowMemory(
            candidates(3),
            embedding=object(),
            retrieve_num=1,
            score_threshold=0.5,
            vector_store_factory=fake_store_factory,
        )
        initial_ids = select_warm_start_ids(
            memory.candidate_order,
            count=2,
            seed=3,
        )

        memory.activate(initial_ids, interval_id=0)
        memory.rebuild_available_index()

        self.assertEqual(memory.available_ids, set(initial_ids))
        self.assertTrue(set(initial_ids).isdisjoint(memory.pending_ids))
        for memory_id in initial_ids:
            query = memory.candidates[memory_id].query
            retrieved = memory.retrieve(query)
            self.assertEqual(retrieved[0][0].metadata["memory_id"], memory_id)
            self.assertEqual(retrieved[0][0].metadata["activated_interval"], 0)

    def test_full_warm_start_exhausts_pending_pool(self):
        ids = [item.memory_id for item in candidates(5)]
        memory = ScheduledWorkflowMemory(
            candidates(5),
            embedding=object(),
            retrieve_num=1,
            vector_store_factory=fake_store_factory,
        )
        initial_ids = select_warm_start_ids(ids, count=5, seed=9)
        memory.activate(initial_ids, interval_id=0)

        self.assertEqual(memory.pending_ids, set())
        self.assertEqual(
            RandomScheduler(ids, seed=1).select(memory.pending_ids, 2).memory_ids,
            (),
        )


class AvailableMemoryTests(unittest.TestCase):
    def make_memory(self):
        return ScheduledWorkflowMemory(
            candidates(3),
            embedding=object(),
            retrieve_num=1,
            score_threshold=0.5,
            vector_store_factory=fake_store_factory,
        )

    def test_unactivated_candidate_cannot_be_retrieved(self):
        memory = self.make_memory()
        memory.activate(["mem_0000"], interval_id=1)
        memory.rebuild_available_index()

        self.assertEqual(memory.retrieve("query-1"), [])
        retrieved = memory.retrieve("query-0")
        self.assertEqual(retrieved[0][0].metadata["memory_id"], "mem_0000")

    def test_new_activation_is_visible_only_after_next_interval_rebuild(self):
        memory = self.make_memory()
        memory.activate(["mem_0000"], interval_id=1)
        memory.rebuild_available_index()

        memory.activate(["mem_0001"], interval_id=2)
        self.assertEqual(memory.retrieve("query-1"), [])

        memory.rebuild_available_index()
        retrieved = memory.retrieve("query-1")
        self.assertEqual(retrieved[0][0].metadata["activated_interval"], 2)


class SchedulerTests(unittest.TestCase):
    def test_random_order_is_deterministic_for_same_seed(self):
        ids = [item.memory_id for item in candidates(10)]
        first = RandomScheduler(ids, seed=17)
        second = RandomScheduler(ids, seed=17)

        self.assertEqual(
            first.select(ids, 10).memory_ids,
            second.select(ids, 10).memory_ids,
        )

    def test_oracle_recomputes_bottom_c_for_each_next_interval(self):
        scheduler = OracleHighScheduler()
        memory = ScheduledWorkflowMemory(
            candidates(3),
            embedding=FakeEmbedding(),
            retrieve_num=1,
        )
        ids = memory.pending_ids

        first = scheduler.select(
            ids,
            1,
            next_interval_queries=["future-a"],
            distance_scorer=memory.oracle_distance_sums,
        )
        second = scheduler.select(
            ids,
            1,
            next_interval_queries=["future-b"],
            distance_scorer=memory.oracle_distance_sums,
        )

        self.assertEqual(first.memory_ids, ("mem_0000",))
        self.assertEqual(second.memory_ids, ("mem_0002",))
        self.assertEqual(first.oracle_distances["mem_0000"], 0.0)

    def test_greedy_novelty_uses_available_pool_and_updates_references(self):
        scheduler = GreedyNoveltyScheduler()
        positions = {
            "mem_0000": 0.0,
            "mem_0001": 10.0,
            "mem_0002": 11.0,
            "mem_0003": 20.0,
        }
        distance_matrix = {
            memory_id: {
                reference_id: (position - positions[reference_id]) ** 2
                for reference_id in positions
            }
            for memory_id, position in positions.items()
        }

        selection = scheduler.select(
            {"mem_0001", "mem_0002", "mem_0003"},
            2,
            available_ids={"mem_0000"},
            distance_matrix=distance_matrix,
        )

        self.assertEqual(selection.memory_ids, ("mem_0003", "mem_0001"))
        self.assertEqual(selection.scheduler_scores["mem_0003"]["value"], 400.0)
        self.assertEqual(
            selection.scheduler_scores["mem_0003"][
                "nearest_reference_memory_id"
            ],
            "mem_0000",
        )
        self.assertEqual(selection.scheduler_scores["mem_0001"]["value"], 100.0)
        self.assertTrue(
            selection.scheduler_scores["mem_0001"]["higher_is_better"]
        )

    def test_greedy_novelty_cold_start_and_ties_use_stable_ids(self):
        scheduler = GreedyNoveltyScheduler()
        ids = {"mem_0002", "mem_0001", "mem_0000"}
        distance_matrix = {
            memory_id: {
                reference_id: float(memory_id != reference_id)
                for reference_id in ids
            }
            for memory_id in ids
        }

        selection = scheduler.select(
            ids,
            2,
            available_ids=set(),
            distance_matrix=distance_matrix,
        )

        self.assertEqual(selection.memory_ids, ("mem_0000", "mem_0001"))
        self.assertIsNone(selection.scheduler_scores["mem_0000"]["value"])
        self.assertEqual(
            selection.scheduler_scores["mem_0000"]["score_type"],
            "empty_reference_tie_break",
        )

    def test_candidate_query_distance_matrix_uses_squared_l2(self):
        memory = ScheduledWorkflowMemory(
            candidates(3),
            embedding=FakeEmbedding(),
            retrieve_num=1,
        )

        first = memory.candidate_query_distance_matrix(memory.candidate_order)
        second = memory.candidate_query_distance_matrix(memory.candidate_order)

        self.assertEqual(first, second)
        self.assertEqual(first["mem_0000"]["mem_0002"], 400.0)
        self.assertEqual(first["mem_0001"]["mem_0002"], 100.0)
        self.assertEqual(first["mem_0001"]["mem_0001"], 0.0)

    def test_oracle_coverage_uses_available_pool_as_marginal_baseline(self):
        scheduler = OracleCoverageScheduler()
        distance_matrix = {
            "mem_0000": (0.0, 400.0),
            "mem_0001": (25.0, 225.0),
            "mem_0002": (400.0, 0.0),
        }

        def score(queries, memory_ids):
            return {
                memory_id: distance_matrix[memory_id]
                for memory_id in memory_ids
            }

        with_available = scheduler.select(
            {"mem_0001", "mem_0002"},
            1,
            available_ids={"mem_0000"},
            next_interval_queries=["future-a", "future-b"],
            distance_scorer=score,
        )
        without_available = scheduler.select(
            {"mem_0001", "mem_0002"},
            1,
            available_ids=set(),
            next_interval_queries=["future-a", "future-b"],
            distance_scorer=score,
        )

        self.assertEqual(with_available.memory_ids, ("mem_0002",))
        self.assertEqual(without_available.memory_ids, ("mem_0001",))
        self.assertEqual(
            with_available.oracle_scores["mem_0002"]["score_type"],
            "faiss_l2_marginal_gain",
        )
        self.assertEqual(with_available.oracle_scores["mem_0002"]["value"], 400.0)
        self.assertEqual(
            without_available.oracle_scores["mem_0001"]["score_type"],
            "faiss_l2_distance_sum",
        )

    def test_oracle_coverage_recomputes_gain_after_each_selection(self):
        scheduler = OracleCoverageScheduler()
        distance_matrix = {
            "mem_0000": (10.0, 10.0),
            "mem_0001": (0.0, 10.0),
            "mem_0002": (1.0, 10.0),
            "mem_0003": (10.0, 0.0),
        }

        def score(queries, memory_ids):
            return {
                memory_id: distance_matrix[memory_id]
                for memory_id in memory_ids
            }

        selection = scheduler.select(
            {"mem_0001", "mem_0002", "mem_0003"},
            2,
            available_ids={"mem_0000"},
            next_interval_queries=["future-a", "future-b"],
            distance_scorer=score,
        )

        self.assertEqual(selection.memory_ids, ("mem_0001", "mem_0003"))
        self.assertTrue(
            all(
                score["score_type"] == "faiss_l2_marginal_gain"
                for score in selection.oracle_scores.values()
            )
        )
        self.assertEqual(
            selection.oracle_scores["mem_0001"]["selection_rank"], 1
        )
        self.assertEqual(
            selection.oracle_scores["mem_0003"]["selection_rank"], 2
        )

    def test_capacity_is_respected_and_memory_is_not_selected_twice(self):
        ids = [item.memory_id for item in candidates(5)]
        scheduler = RandomScheduler(ids, seed=3)
        pending = set(ids)

        first = scheduler.select(pending, 2)
        pending.difference_update(first.memory_ids)
        second = scheduler.select(pending, 2)

        self.assertLessEqual(len(first.memory_ids), 2)
        self.assertLessEqual(len(second.memory_ids), 2)
        self.assertTrue(set(first.memory_ids).isdisjoint(second.memory_ids))


class IntervalBatchTests(unittest.TestCase):
    def test_batches_do_not_cross_logical_interval(self):
        batches = build_interval_batches(8, batch_size=2, interval_size=3)

        self.assertEqual(
            batches,
            [
                (0, 2, 0, True, False),
                (2, 3, 0, False, True),
                (3, 5, 1, True, False),
                (5, 6, 1, False, True),
                (6, 8, 2, True, True),
            ],
        )
        for start, end, interval_id, _, _ in batches:
            self.assertEqual(start // 3, interval_id)
            self.assertEqual((end - 1) // 3, interval_id)


if __name__ == "__main__":
    unittest.main()
