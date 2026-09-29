import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ProcedureMem.llm_usage import (
    LLMCallResult,
    LLMUsage,
    aggregate_usage,
    append_jsonl,
    extract_usage,
    memory_usage_records,
)
from ProcedureMem.analyze_token_workload import workload_summary


class LLMUsageTests(unittest.TestCase):
    def test_extracts_attribute_and_mapping_usage(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            )
        )
        self.assertEqual(extract_usage(response), LLMUsage(11, 3, 14))
        self.assertEqual(
            extract_usage(
                {
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    }
                }
            ),
            LLMUsage(7, 2, 9),
        )

    def test_missing_or_partial_usage_is_not_reported_as_zero(self):
        self.assertEqual(extract_usage({}), LLMUsage(None, None, None))
        aggregate = aggregate_usage(
            [LLMUsage(10, 2, 12), LLMUsage(8, None, None)]
        )
        self.assertFalse(aggregate["usage_complete"])
        self.assertIsNone(aggregate["prompt_tokens"])
        self.assertEqual(aggregate["reported_prompt_tokens"], 10)
        self.assertEqual(aggregate["reported_call_count"], 1)

    def test_complete_usage_aggregates_and_jsonl_round_trips(self):
        aggregate = aggregate_usage([LLMUsage(10, 2, 12), LLMUsage(20, 4, 24)])
        self.assertEqual(aggregate["prompt_tokens"], 30)
        self.assertEqual(aggregate["completion_tokens"], 6)
        self.assertEqual(aggregate["total_tokens"], 36)
        call = LLMCallResult("workflow", LLMUsage(30, 5, 35), "m", "r")
        records = memory_usage_records(
            source_index=4, memory_id="memory_4", call=call
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.jsonl"
            append_jsonl(path, records)
            loaded = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(loaded, records)

    def test_workload_summary_uses_only_complete_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "task.jsonl"
            memory_path = root / "memory.jsonl"
            append_jsonl(
                task_path,
                [
                    {
                        "record_type": "task_aggregate",
                        "usage_complete": True,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                    },
                    {
                        "record_type": "task_aggregate",
                        "usage_complete": False,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                    },
                ],
            )
            append_jsonl(
                memory_path,
                [
                    {
                        "record_type": "memory_aggregate",
                        "usage_complete": True,
                        "prompt_tokens": 40,
                        "completion_tokens": 10,
                    }
                ],
            )
            summary = workload_summary(task_path, memory_path, gamma=1)
        self.assertEqual(summary["task_sample_count"], 1)
        self.assertEqual(summary["memory_sample_count"], 1)
        self.assertEqual(summary["median_ratio"], 2.0)
        self.assertEqual(summary["mean_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
