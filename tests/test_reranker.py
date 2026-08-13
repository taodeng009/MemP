import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from ProcedureMem.benchmark_stats import latency_summary
from ProcedureMem.reranker import OpenMemReranker, RerankerError


class FakeResponse:
    def __init__(self, payload=None, *, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class OpenMemRerankerTests(unittest.TestCase):
    def test_normal_response_maps_indexes_and_scores(self):
        session = Mock()
        session.post.return_value = FakeResponse(
            {
                "id": "request-1",
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ],
            }
        )
        reranker = OpenMemReranker(api_key="secret", session=session)
        response = reranker.rerank(
            query="clean an apple",
            documents=["first", "second"],
            top_n=2,
        )

        self.assertEqual([item.index for item in response.results], [1, 0])
        self.assertEqual(response.results[0].relevance_score, 0.9)
        self.assertGreaterEqual(response.latency_ms, 0.0)
        request = session.post.call_args
        self.assertEqual(request.kwargs["json"]["model"], "memos-reranker-4b")
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Token secret")

    def test_empty_documents_do_not_call_api(self):
        session = Mock()
        reranker = OpenMemReranker(api_key="secret", session=session)
        response = reranker.rerank(query="task", documents=[], top_n=1)
        self.assertEqual(response.results, ())
        session.post.assert_not_called()

    def test_timeout_and_non_2xx_are_wrapped(self):
        for error in (TimeoutError("slow"), RuntimeError("bad status")):
            with self.subTest(error=type(error).__name__):
                session = Mock()
                if isinstance(error, TimeoutError):
                    session.post.side_effect = error
                else:
                    session.post.return_value = FakeResponse(error=error)
                reranker = OpenMemReranker(api_key="secret", session=session)
                with self.assertRaises(RerankerError):
                    reranker.rerank(
                        query="task",
                        documents=["candidate"],
                        top_n=1,
                    )

    def test_api_key_is_not_part_of_serializable_response(self):
        session = Mock()
        session.post.return_value = FakeResponse(
            {"results": [{"index": 0, "relevance_score": 1.0}]}
        )
        secret = "do-not-write-this-key"
        response = OpenMemReranker(api_key=secret, session=session).rerank(
            query="task", documents=["candidate"], top_n=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            path.write_text(
                json.dumps(
                    {
                        "latency_ms": response.latency_ms,
                        "results": [item.__dict__ for item in response.results],
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))


class LatencySummaryTests(unittest.TestCase):
    def test_reports_mean_median_and_interpolated_p95(self):
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)


if __name__ == "__main__":
    unittest.main()
