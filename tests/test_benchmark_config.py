import os
import unittest
from unittest.mock import patch

from ProcedureMem.benchmark_config import (
    RERANK_CANDIDATE_THRESHOLD_ENV,
    candidate_score_threshold,
)


class CandidateScoreThresholdTests(unittest.TestCase):
    def test_empty_or_missing_environment_disables_threshold(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(candidate_score_threshold())
        with patch.dict(
            os.environ, {RERANK_CANDIDATE_THRESHOLD_ENV: "  "}, clear=True
        ):
            self.assertIsNone(candidate_score_threshold())

    def test_environment_value_is_parsed(self):
        with patch.dict(
            os.environ, {RERANK_CANDIDATE_THRESHOLD_ENV: "0.5"}, clear=True
        ):
            self.assertEqual(candidate_score_threshold(), 0.5)

    def test_cli_value_overrides_environment(self):
        with patch.dict(
            os.environ, {RERANK_CANDIDATE_THRESHOLD_ENV: "0.5"}, clear=True
        ):
            self.assertEqual(candidate_score_threshold(0.3), 0.3)

    def test_invalid_environment_is_rejected(self):
        for value in ("invalid", "-0.1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {RERANK_CANDIDATE_THRESHOLD_ENV: value}, clear=True
            ):
                with self.assertRaises(ValueError):
                    candidate_score_threshold()


if __name__ == "__main__":
    unittest.main()
