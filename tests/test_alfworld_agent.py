import json
import unittest
from pathlib import Path

from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.alfworld_agent import (
    parse_action,
    resolve_litellm_model,
    run_alfworld_batch,
)


EXAMPLES_PATH = (
    Path(__file__).resolve().parents[1]
    / "ProcedureMem"
    / "Alfworld"
    / "alfworld_examples.json"
)


class ScriptedBatchEnv:
    def __init__(self):
        self.actions = []

    def step(self, actions):
        self.actions.append(actions)
        if len(self.actions) == 1:
            return ["first done", "continue"], [0, 0], [True, False], {
                "won": [True, False]
            }
        return ["first unchanged", "second done"], [0, 0], [True, True], {
            "won": [True, True]
        }


class ImmediateSuccessEnv:
    def __init__(self):
        self.actions = []

    def step(self, actions):
        self.actions.append(actions)
        size = len(actions)
        return ["done"] * size, [0] * size, [True] * size, {"won": [True] * size}


class AlfworldActionParserTests(unittest.TestCase):
    def test_openai_compatible_endpoint_gets_litellm_provider_prefix(self):
        self.assertEqual(
            resolve_litellm_model("Qwen/Qwen3-4B", "http://localhost:8000/v1"),
            "openai/Qwen/Qwen3-4B",
        )
        self.assertEqual(
            resolve_litellm_model("openai/Qwen/Qwen3-4B", "http://localhost:8000/v1"),
            "openai/Qwen/Qwen3-4B",
        )
        self.assertEqual(resolve_litellm_model("gpt-4o", None), "gpt-4o")

    def test_parses_common_format_variations(self):
        cases = {
            "Thought: inspect\naction:   GO   TO fridge 1": "go to fridge 1",
            "Thought: place it\nAction: move apple 1 to table 1\nExtra text": (
                "move apple 1 to table 1"
            ),
            "```\nAction: use desklamp 1\n```": "use desklamp 1",
            'Action: "go to fridge 1."': "go to fridge 1",
            "Action: inventory": "inventory",
            "Action: look": "look",
            "Action: examine microwave 1": "examine microwave 1",
            "take mug 1 from counter 1": "take mug 1 from counter 1",
            "Action: dance\nAction: move apple 1 to table 1": (
                "move apple 1 to table 1"
            ),
            "Action: put apple 1 in/on table 1": "move apple 1 to table 1",
            "Action: toggle desklamp 1": "use desklamp 1",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(parse_action(response), expected)

    def test_forwards_unsupported_or_empty_actions_to_alfworld(self):
        self.assertEqual(parse_action("Action: task completed"), "task completed")
        self.assertEqual(parse_action("Action: dance"), "dance")
        self.assertEqual(parse_action("Thought only"), "")
        self.assertEqual(parse_action(""), "")

    def test_prompt_and_examples_use_the_same_action_vocabulary(self):
        self.assertIn("move {obj} to {recep}", alfworld_system_prompt)
        self.assertIn("use {obj}", alfworld_system_prompt)
        self.assertNotIn("put {obj}", alfworld_system_prompt)
        self.assertNotIn("toggle {obj}", alfworld_system_prompt)

        with EXAMPLES_PATH.open("r", encoding="utf-8") as reader:
            serialized = json.dumps(json.load(reader))
        self.assertNotIn("Action: put ", serialized)
        self.assertNotIn("Action: toggle ", serialized)


class AlfworldBatchRunnerTests(unittest.TestCase):
    def test_clean_trajectory_excludes_injected_prompt_context(self):
        env = ImmediateSuccessEnv()
        results = run_alfworld_batch(
            env=env,
            observations=["task with retrieved workflow"],
            trajectory_observations=["clean task observation"],
            names=["task"],
            llm_fn=lambda _: "Thought: act\nAction: go to table 1",
            system_prompt="system",
            few_shot=False,
        )

        serialized = json.dumps(results[0]["trajectory"])
        self.assertIn("clean task observation", serialized)
        self.assertNotIn("retrieved workflow", serialized)
        self.assertNotIn("system", serialized)
        self.assertEqual(results[0]["trajectory"][0]["from"], "human")
        self.assertEqual(results[0]["trajectory"][1]["from"], "gpt")

    def test_tasks_complete_independently_and_keep_per_task_steps(self):
        env = ScriptedBatchEnv()

        def llm(messages):
            if "first task" in messages[1]["content"]:
                return "Thought: finish first\nAction: go to table 1"
            if len([message for message in messages if message["role"] == "assistant"]) == 0:
                return "Thought: start second\nAction: go to counter 1"
            return "Thought: finish second\nAction: move apple 1 to table 1"

        results = run_alfworld_batch(
            env=env,
            observations=["first task", "second task"],
            names=["task-one", "task-two"],
            llm_fn=llm,
            system_prompt="system",
            few_shot=False,
            max_steps=3,
        )

        self.assertEqual(env.actions[1][0], "look")
        self.assertEqual(results[0]["termination_reason"], "success")
        self.assertEqual(results[0]["steps"], 1)
        self.assertEqual(results[1]["termination_reason"], "success")
        self.assertEqual(results[1]["steps"], 2)
        self.assertEqual(len(results), 2)

    def test_llm_failure_is_isolated_from_another_task(self):
        env = ImmediateSuccessEnv()

        def llm(messages):
            if "bad task" in messages[1]["content"]:
                raise RuntimeError("API retries exhausted")
            return "Thought: proceed\nAction: go to table 1"

        results = run_alfworld_batch(
            env=env,
            observations=["bad task", "good task"],
            names=["bad", "good"],
            llm_fn=llm,
            system_prompt="system",
            few_shot=False,
        )

        self.assertEqual(env.actions, [["look", "go to table 1"]])
        self.assertEqual(results[0]["termination_reason"], "llm_error")
        self.assertEqual(results[0]["steps"], 0)
        self.assertIn("retries exhausted", results[0]["error"])
        self.assertEqual(results[1]["termination_reason"], "success")

    def test_missing_action_is_sent_to_the_environment(self):
        class InvalidActionEnv:
            def __init__(self):
                self.actions = []

            def step(self, actions):
                self.actions.append(actions)
                return ["Nothing happened."], [0], [False], {"won": [False]}

        env = InvalidActionEnv()
        results = run_alfworld_batch(
            env=env,
            observations=["task"],
            names=["task"],
            llm_fn=lambda _: "Thought: uncertain",
            system_prompt="system",
            few_shot=False,
            max_steps=3,
        )

        self.assertEqual(env.actions, [[""], [""], [""]])
        self.assertEqual(results[0]["termination_reason"], "max_steps")
        self.assertEqual(results[0]["steps"], 3)
        self.assertFalse(results[0]["reward"])

    def test_premature_completion_gets_native_environment_feedback(self):
        class NativeFeedbackEnv:
            def __init__(self):
                self.actions = []

            def step(self, actions):
                self.actions.append(actions)
                if len(self.actions) == 1:
                    return ["Nothing happened."], [0], [False], {"won": [False]}
                return ["done"], [0], [True], {"won": [True]}

        env = NativeFeedbackEnv()

        def llm(messages):
            if any("Nothing happened" in message["content"] for message in messages):
                return "Thought: continue\nAction: go to table 1"
            return "Thought: finished\nAction: task completed"

        results = run_alfworld_batch(
            env=env,
            observations=["task"],
            names=["task"],
            llm_fn=llm,
            system_prompt="system",
            few_shot=False,
            max_steps=3,
        )

        self.assertEqual(env.actions, [["task completed"], ["go to table 1"]])
        self.assertEqual(results[0]["termination_reason"], "success")
        self.assertEqual(results[0]["steps"], 2)
        self.assertIsNone(results[0]["error"])
        self.assertIn("Nothing happened", results[0]["messages"][-3]["content"])

    def test_active_task_ends_at_max_steps(self):
        class NeverDoneEnv:
            def step(self, actions):
                return ["continue"], [0], [False], {"won": [False]}

        results = run_alfworld_batch(
            env=NeverDoneEnv(),
            observations=["task"],
            names=["task"],
            llm_fn=lambda _: "Action: go to table 1",
            system_prompt="system",
            few_shot=False,
            max_steps=2,
        )

        self.assertEqual(results[0]["termination_reason"], "max_steps")
        self.assertEqual(results[0]["steps"], 2)

    def test_environment_failure_is_recorded_for_each_stepped_task(self):
        class BrokenEnv:
            def step(self, actions):
                raise RuntimeError("TextWorld process stopped")

        results = run_alfworld_batch(
            env=BrokenEnv(),
            observations=["first", "second"],
            names=["first", "second"],
            llm_fn=lambda _: "Action: go to table 1",
            system_prompt="system",
            few_shot=False,
        )

        self.assertEqual(
            [result["termination_reason"] for result in results],
            ["environment_error", "environment_error"],
        )
        self.assertTrue(all("TextWorld" in result["error"] for result in results))


if __name__ == "__main__":
    unittest.main()
