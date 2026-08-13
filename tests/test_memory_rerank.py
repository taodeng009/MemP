import unittest
from types import SimpleNamespace

from ProcedureMem.reranker import RerankResponse, RerankResult


class FakeVectorStore:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def similarity_search_with_score(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.items[: kwargs["k"]]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, *, query, documents, top_n):
        self.calls.append((query, documents, top_n))
        return RerankResponse(
            results=(
                RerankResult(index=1, relevance_score=0.9),
                RerankResult(index=0, relevance_score=0.8),
            ),
            latency_ms=12.0,
            request_id="request",
        )


class MemoryRerankTests(unittest.TestCase):
    def test_retrieve_with_rerank_maps_indexes_and_optional_threshold(self):
        try:
            from ProcedureMem.memory import Memory
        except ModuleNotFoundError as exc:
            self.skipTest(f"Memory dependencies unavailable: {exc}")

        documents = [
            SimpleNamespace(
                page_content="first task",
                metadata={"query": "first task", "workflow": "first workflow"},
            ),
            SimpleNamespace(
                page_content="second task",
                metadata={"query": "second task", "workflow": "second workflow"},
            ),
        ]
        memory = Memory.__new__(Memory)
        memory.retrieve_policy = "query"
        memory.documents = documents
        memory.vector_store = FakeVectorStore([(documents[0], 0.1), (documents[1], 0.2)])
        reranker = FakeReranker()

        output = memory.retrieve_with_rerank(
            "current task",
            reranker=reranker,
            candidate_k=20,
            top_n=2,
            score_threshold=0.8,
        )

        self.assertEqual(memory.vector_store.calls[0][1]["score_threshold"], 0.8)
        self.assertEqual(output["items"][0]["document"], documents[1])
        self.assertEqual(output["items"][0]["vector_rank"], 2)
        self.assertEqual(output["items"][0]["rerank_rank"], 1)
        self.assertEqual(output["items"][0]["rerank_score"], 0.9)
        self.assertIn("Reusable workflow: second workflow", reranker.calls[0][1][1])

    def test_retrieve_with_rerank_omits_disabled_threshold(self):
        try:
            from ProcedureMem.memory import Memory
        except ModuleNotFoundError as exc:
            self.skipTest(f"Memory dependencies unavailable: {exc}")

        document = SimpleNamespace(
            page_content="task",
            metadata={"query": "task", "workflow": "workflow"},
        )
        memory = Memory.__new__(Memory)
        memory.retrieve_policy = "query"
        memory.documents = [document]
        memory.vector_store = FakeVectorStore([(document, 0.1)])

        class OneReranker:
            def rerank(self, **kwargs):
                return RerankResponse(
                    results=(RerankResult(index=0, relevance_score=1.0),),
                    latency_ms=1.0,
                )

        memory.retrieve_with_rerank(
            "task", reranker=OneReranker(), candidate_k=1, top_n=1
        )
        self.assertNotIn("score_threshold", memory.vector_store.calls[0][1])


if __name__ == "__main__":
    unittest.main()
