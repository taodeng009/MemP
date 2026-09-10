"""Hit/quality scheduling regression tests; no model or environment calls."""
import unittest
from types import SimpleNamespace

from ProcedureMem.cloud_scheduling import (
    OracleHitQualityScheduler, OracleCoverageScheduler, OracleExactRetrievalScheduler,
)
from ProcedureMem.online_construction import OnlineConstructionController
from test_online_construction import FakeMemory, result


class HitQualityTests(unittest.TestCase):
    def select(self, matrix, available=(), capacity=2, **kwargs):
        return OracleHitQualityScheduler().select(
            set(matrix)-set(available), capacity, available_ids=available,
            future_queries=["q"]*len(next(iter(matrix.values()))),
            distance_scorer=lambda q, ids: {m: matrix[m] for m in ids},
            top_k=kwargs.pop("top_k", 3), score_threshold=0.5, **kwargs)

    def test_hits_boundary_duplicates_and_greedy_update(self):
        matrix = {"a": [0.5, 0.5, 1], "b": [0.5, 0.5, 1], "c": [1, 1, 0.4]}
        s = self.select(matrix, alpha=1)
        self.assertEqual(s.memory_ids, ("a", "c"))
        self.assertEqual(s.oracle_scores["a"]["hit_gain"], 2)
        self.assertEqual(s.oracle_scores["a"]["quality_gain"], 0)
        self.assertEqual(s.oracle_scores["c"]["hit_gain"], 1)
        self.assertEqual(s.oracle_scores["c"]["hit_gain_max"], 1)

    def test_alpha_zero_matches_existing_oracles(self):
        matrix = {"a": [0.2, 0.9], "b": [0.3, 0.4], "c": [0.1, 0.8], "d": [0.7, 0.1]}
        for available in ((), ("a",)):
            for metric in ("bd", "ru"):
                with self.subTest(available=available, metric=metric):
                    args = dict(available_ids=available,
                                distance_scorer=lambda q, ids: {m: matrix[m] for m in ids})
                    pending = set(matrix)-set(available)
                    old = (OracleCoverageScheduler().select(
                        pending, 3, next_interval_queries=["q", "q"], **args)
                        if metric == "bd" else OracleExactRetrievalScheduler().select(
                            pending, 3, future_queries=["q", "q"], top_k=3,
                            score_threshold=0.5, **args))
                    new = self.select(matrix, available, capacity=3, metric=metric, alpha=0)
                    self.assertEqual(new.memory_ids, old.memory_ids)

    def test_alpha_changes_breadth_quality_tradeoff(self):
        matrix = {"a": [0.0, 1.0], "b": [0.49, 0.49]}
        quality = self.select(matrix, capacity=1, alpha=0.25)
        breadth = self.select(matrix, capacity=1, alpha=0.75)
        self.assertEqual(quality.memory_ids, ("a",))
        self.assertEqual(breadth.memory_ids, ("b",))
        score = breadth.oracle_scores["b"]
        self.assertAlmostEqual(score["priority"],
                               0.75*2/(2+1e-8) + 0.25*0.02/(0.5+1e-8))

    def test_bd_bootstrap_and_unclipped_gain(self):
        s = self.select({"a": [0.6], "b": [0.9]}, capacity=1, metric="bd", alpha=1)
        self.assertEqual(s.memory_ids, ("a",))
        self.assertTrue(s.oracle_scores["a"]["coverage_bootstrap"])
        self.assertIsNone(s.oracle_scores["a"]["quality_gain"])
        self.assertIsNone(s.oracle_scores["a"]["priority"])
        s = self.select({"old": [0.9], "a": [0.6]}, ("old",), metric="bd")
        self.assertAlmostEqual(s.oracle_scores["a"]["quality_gain"], 0.3)
        self.assertEqual(s.oracle_scores["a"]["hit_gain"], 0)

    def test_ru_topk_replacement_and_normalization(self):
        s = self.select({"old": [0.3], "a": [0.1], "b": [0.4]}, ("old",), top_k=1)
        self.assertEqual(s.memory_ids, ("a", "b"))
        a, b = s.oracle_scores["a"], s.oracle_scores["b"]
        self.assertAlmostEqual(a["quality_gain"], 0.2)
        self.assertAlmostEqual(a["priority"], 0.5*0.2/(0.2+1e-8))
        self.assertEqual(b["quality_gain"], 0)

    def test_zero_gains_and_validation(self):
        self.assertEqual(self.select({"b": [1], "a": [1]}, alpha=1).memory_ids, ("a", "b"))
        self.assertEqual(self.select({"a": [1]}, capacity=0).memory_ids, ())
        for alpha in (-1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.select({"a": [1]}, alpha=alpha)

    def test_controller_staging_failure_and_metadata(self):
        for metric in ("ru", "bd"):
            memory = FakeMemory(fail_queries=("task-b",))
            controller = OnlineConstructionController(
                memory=memory, policy="oracle_hit_quality", capacity=2,
                hit_quality_metric=metric)
            controller.admit_results([result(0), result(1)], interval_id=0)
            event = controller.construct(interval_id=0, future_queries=["task-a", "task-b"],
                                         requested_lookahead_horizon=1,
                                         effective_lookahead_horizon=1, future_interval_count=1)
            self.assertEqual(event["oracle_future_query_count"], 2)
            self.assertEqual(event["oracle_retrieval_threshold"], 0.5)
            self.assertEqual(len(memory.documents), 0)
            self.assertEqual(len(controller.queue), 1)
            self.assertEqual({e["construction_result"] for e in event["construction_results"]},
                             {"success", "failure"})
            self.assertTrue(all("hit_gain" in e["oracle_score"] for e in event["construction_results"]))
            controller.activate_staged(interval_id=1)
            self.assertEqual(len(memory.documents), 1)

    def test_actual_faiss_threshold_and_distances(self):
        try:
            import faiss
            from langchain_community.vectorstores import FAISS
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_core.documents import Document
        except ImportError:
            self.skipTest("FAISS / LangChain unavailable locally; run this test on server")
        import numpy as np
        from ProcedureMem.memory import Memory
        vectors = np.array([[0.5, 0.5], [1, 0], [0, 0]], dtype=np.float32)
        index = faiss.IndexFlatL2(2)
        index.add(vectors)
        docs = {str(i): Document(page_content=str(i)) for i in range(3)}
        store = FAISS(lambda q: [0.0, 0.0], index, InMemoryDocstore(docs),
                      {i: str(i) for i in range(3)})
        memory = SimpleNamespace(documents=list(docs.values()), vector_store=store,
                                 retrieve_num=3, retrieve_policy="query")
        retrieved = Memory.retrieve(memory, "q")
        self.assertEqual([float(d) for _, d in retrieved], [0.0, 0.5])
        matrix = {str(i): [float(sum(v*v))] for i, v in enumerate(vectors)}
        s = self.select(matrix, alpha=1, capacity=3)
        self.assertEqual(sum(x["hit_gain"] for x in s.oracle_scores.values()), 1)
        self.assertAlmostEqual(sum(x["quality_gain"] for x in s.oracle_scores.values()),
                               sum(max(0.0, 0.5-float(d)) for _, d in retrieved))


if __name__ == "__main__":
    unittest.main()
