import json
import tempfile
import unittest
from pathlib import Path

from ProcedureMem.alfworld_experiment import (
    build_paired_comparison,
    build_task_manifest,
    inject_memory,
    load_json,
    manifest_sha256,
    retrieval_records,
    summarize_results,
    task_query,
    validate_task_manifest,
    write_results,
)


class FakeDocument:
    def __init__(self, **metadata):
        self.metadata = metadata


class TaskManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary.name)
        self.gamefiles = []
        for index in range(6):
            path = (
                self.data_root
                / "json_2.1.1"
                / "valid_unseen"
                / "pick_and_place_simple"
                / f"trial_{index}"
                / "game.tw-pddl"
            )
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            self.gamefiles.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_seeded_manifest_is_deterministic_and_validates(self):
        first = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=17,
            limit_tasks=4,
        )
        second = build_task_manifest(
            list(reversed(self.gamefiles)),
            data_root=self.data_root,
            split="valid_unseen",
            seed=17,
            limit_tasks=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 4)
        selected = validate_task_manifest(
            first,
            split="valid_unseen",
            available_gamefiles=self.gamefiles,
            data_root=self.data_root,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(manifest_sha256(first)), 64)

    def test_explicit_task_ids_preserve_order(self):
        task_ids = [
            "json_2.1.1/valid_unseen/pick_and_place_simple/trial_4/game.tw-pddl",
            "json_2.1.1/valid_unseen/pick_and_place_simple/trial_1/game.tw-pddl",
        ]
        manifest = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=42,
            task_ids=task_ids,
        )
        self.assertEqual([task["task_id"] for task in manifest["tasks"]], task_ids)

    def test_manifest_rejects_wrong_split_and_missing_task(self):
        manifest = build_task_manifest(
            self.gamefiles,
            data_root=self.data_root,
            split="valid_unseen",
            seed=42,
            limit_tasks=2,
        )
        with self.assertRaisesRegex(ValueError, "Manifest split"):
            validate_task_manifest(
                manifest,
                split="valid_seen",
                available_gamefiles=self.gamefiles,
                data_root=self.data_root,
            )
        with self.assertRaisesRegex(ValueError, "missing from ALFWorld"):
            validate_task_manifest(
                manifest,
                split="valid_unseen",
                available_gamefiles=self.gamefiles[2:],
                data_root=self.data_root,
            )


class ResultSummaryTests(unittest.TestCase):
    @staticmethod
    def _results(condition="no_memory"):
        return [
            {
                "task_id": "task-1",
                "condition": condition,
                "split": "valid_unseen",
                "reward": True,
                "steps": 4,
                "termination_reason": "success",
                "error": None,
            },
            {
                "task_id": "task-2",
                "condition": condition,
                "split": "valid_unseen",
                "reward": False,
                "steps": 10,
                "termination_reason": "max_steps",
                "error": None,
            },
        ]

    def test_summary_metrics_and_files(self):
        summary = summarize_results(self._results())
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_steps"], 7)
        self.assertEqual(summary["average_success_steps"], 4)

        with tempfile.TemporaryDirectory() as directory:
            written = write_results(
                directory,
                self._results(),
                summary_metadata={"model": "test-model"},
            )
            self.assertEqual(written["model"], "test-model")
            self.assertTrue((Path(directory) / "results.jsonl").is_file())
            self.assertTrue((Path(directory) / "summary.csv").is_file())
            self.assertEqual(load_json(Path(directory) / "summary.json")["task_count"], 2)
            self.assertEqual(len(list((Path(directory) / "tasks").glob("*.json"))), 2)

    def test_paired_comparison_requires_identical_tasks_and_parameters(self):
        parameters = {
            "model": "model",
            "split": "valid_unseen",
            "seed": 42,
            "batch_size": 1,
            "max_steps": 30,
            "temperature": 1.0,
            "few_shot": True,
            "top_k": 3,
            "manifest_sha256": "hash",
        }
        no_memory = summarize_results(self._results("no_memory"))
        memory_results = self._results("memory")
        memory_results[1]["reward"] = True
        memory_results[1]["termination_reason"] = "success"
        memory = summarize_results(memory_results)
        no_memory["parameters"] = dict(parameters)
        memory["parameters"] = dict(parameters)

        comparison = build_paired_comparison(no_memory, memory)
        self.assertEqual(comparison["absolute_improvement"], 0.5)
        self.assertEqual(comparison["absolute_improvement_percentage_points"], 50)

        memory["parameters"]["temperature"] = 0.5
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_paired_comparison(no_memory, memory)

    def test_memory_retrieval_is_recorded_and_injected(self):
        records = retrieval_records(
            [
                (
                    FakeDocument(query="cool an apple", workflow="find a fridge", source="traj-1"),
                    0.25,
                )
            ]
        )
        self.assertEqual(records[0]["rank"], 1)
        self.assertEqual(records[0]["score"], 0.25)
        prompt = inject_memory("Your task is to: cool bread", records)
        self.assertIn("find a fridge", prompt)
        self.assertEqual(task_query("room\nYour task is to: cool bread"), "cool bread")


if __name__ == "__main__":
    unittest.main()
