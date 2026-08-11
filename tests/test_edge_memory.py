import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    from langchain_core.embeddings import Embeddings as EmbeddingsBase
except ModuleNotFoundError:
    class EmbeddingsBase:  # type: ignore[no-redef]
        pass

from ProcedureMem.edge_memory import RawTrajectoryMemory, extract_task_episode
from ProcedureMem.edge_subsets import (
    TASK_FAMILIES,
    build_edge_subset_manifest,
    validate_edge_subset_manifest,
)
from ProcedureMem.summarize_edge_p0 import summarize_edge_p0


def trajectory(query):
    return {
        "query": query,
        "source": "alfworld",
        "facts": None,
        "trajectory": [
            {"from": "human", "value": "generic system instruction"},
            {"from": "gpt", "value": "OK"},
            {
                "from": "human",
                "value": f"You are in a room.\n\nYour task is to: {query}.",
            },
            {"from": "gpt", "value": "Thought: act\nAction: go to table 1"},
            {"from": "human", "value": "Observation: done"},
        ],
    }


class FakeEmbeddings(EmbeddingsBase):
    @staticmethod
    def _vector(text):
        lowered = text.lower()
        return [
            float("clean" in lowered),
            float("heat" in lowered),
            float("cool" in lowered),
            float("examine" in lowered),
            float("two" in lowered),
            float(len(lowered) % 7) / 7.0,
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def __call__(self, text):
        """Support LangChain versions that treat duck-typed embeddings as callables."""
        return self.embed_query(text)


class EdgeSubsetTests(unittest.TestCase):
    def test_stratified_subsets_are_deterministic_and_nested(self):
        queries = (
            [f"put apple {index} in cabinet" for index in range(12)]
            + [f"put a clean apple {index} in cabinet" for index in range(5)]
            + [f"cool apple {index} and put it in cabinet" for index in range(4)]
            + [f"heat apple {index} and put it in cabinet" for index in range(3)]
            + [f"examine apple {index} with the desklamp" for index in range(3)]
            + [f"find two apple {index} and put them in cabinet" for index in range(3)]
        )
        trajectories = [trajectory(query) for query in queries]
        first = build_edge_subset_manifest(
            trajectories,
            source_count=30,
            capacities=(10, 20, 25),
            seed=17,
        )
        second = build_edge_subset_manifest(
            trajectories,
            source_count=30,
            capacities=(10, 20, 25),
            seed=17,
        )
        self.assertEqual(first, second)
        validate_edge_subset_manifest(first, trajectory_count=30)

        subsets = first["subsets"]
        edge10 = set(subsets["10"]["trajectory_indices"])
        edge20 = set(subsets["20"]["trajectory_indices"])
        edge25 = set(subsets["25"]["trajectory_indices"])
        self.assertTrue(edge10 < edge20 < edge25)
        self.assertEqual(sum(subsets["25"]["task_family_counts"].values()), 25)
        self.assertEqual(set(subsets["25"]["task_family_counts"]), set(TASK_FAMILIES))


class RawTrajectoryMemoryTests(unittest.TestCase):
    def test_episode_removes_instruction_and_ok(self):
        episode = extract_task_episode(trajectory("clean an apple")["trajectory"])
        self.assertTrue(episode.startswith("Human:\nYou are in a room."))
        self.assertNotIn("generic system instruction", episode)
        self.assertNotIn("\nOK\n", episode)
        self.assertIn("Assistant:\nThought: act", episode)
        self.assertIn("Human:\nObservation: done", episode)

    @unittest.skipUnless(
        importlib.util.find_spec("langchain_community") is not None,
        "LangChain/FAISS dependencies are not installed",
    )
    def test_index_build_reload_and_optional_threshold(self):
        trajectories = [
            trajectory("put a clean potato in microwave"),
            trajectory("heat an apple and put it in fridge"),
            trajectory("cool a mug and put it in cabinet"),
        ]
        manifest = build_edge_subset_manifest(
            trajectories,
            source_count=3,
            capacities=(2,),
            seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = root / "trajectories.json"
            subset_path = root / "subsets.json"
            memory_dir = root / "edge"
            trajectory_path.write_text(json.dumps(trajectories), encoding="utf-8")
            subset_path.write_text(json.dumps(manifest), encoding="utf-8")

            memory = RawTrajectoryMemory(
                trajectory_file=trajectory_path,
                subset_manifest=subset_path,
                capacity=2,
                memory_dir=memory_dir,
                embedding=FakeEmbeddings(),
            )
            self.assertEqual(memory.document_count, 2)
            self.assertTrue((memory_dir / "faiss" / "index.faiss").is_file())
            retrieved = memory.retrieve("put a clean potato in microwave")
            self.assertEqual(len(retrieved), 1)
            self.assertIn("trajectory", retrieved[0][0].metadata)

            reloaded = RawTrajectoryMemory(
                trajectory_file=trajectory_path,
                subset_manifest=subset_path,
                capacity=2,
                memory_dir=memory_dir,
                embedding=FakeEmbeddings(),
            )
            self.assertEqual(len(reloaded.retrieve("put a clean potato in microwave")), 1)
            self.assertEqual(
                reloaded.retrieve("completely unrelated task", score_threshold=0.0),
                [],
            )


class EdgeP0SummaryTests(unittest.TestCase):
    def test_summary_combines_capacities_and_counts_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rewards = {
                "edge_raw_50": [False, True],
                "edge_raw_100": [True, True],
                "edge_raw_150": [True, False],
            }
            for condition, values in rewards.items():
                condition_dir = root / condition
                condition_dir.mkdir()
                results = [
                    {
                        "task_id": f"task-{index}",
                        "task_type": "clean_then_place",
                        "reward": reward,
                    }
                    for index, reward in enumerate(values)
                ]
                (condition_dir / "results.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in results),
                    encoding="utf-8",
                )
                success_count = sum(values)
                (condition_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "task_count": 2,
                            "success_count": success_count,
                            "success_rate": success_count / 2,
                            "average_steps": 4.0,
                            "average_success_steps": 3.0,
                            "error_count": 0,
                        }
                    ),
                    encoding="utf-8",
                )

            summary = summarize_edge_p0(root)
            self.assertEqual(len(summary["overview"]), 3)
            first_transition = summary["edge_transitions"][0]
            self.assertEqual(first_transition["failure_to_success"], 1)
            self.assertEqual(first_transition["both_success"], 1)
            self.assertTrue((root / "capacity_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
