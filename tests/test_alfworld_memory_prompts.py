import unittest

from ProcedureMem.Alfworld.memory_prompts import (
    PROMPT_APPROVAL,
    PROMPT_SOURCE,
    PROMPT_VERSION,
    build_prompt_manifest,
    generate_events_from_trajectory_prompt,
    generate_workflow_from_events_prompt,
    generate_workflow_from_trajectory_prompt,
    get_prompt_spec,
    prompt_manifest_mismatches,
)


class AlfworldMemoryPromptTests(unittest.TestCase):
    def test_direct_prompt_is_alfworld_specific(self):
        messages = generate_workflow_from_trajectory_prompt(
            "put a clean apple on the table",
            "Thought: find apple\nAction: go to fridge 1",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        prompt = messages[1]["content"]
        self.assertIn("put a clean apple on the table", prompt)
        self.assertIn("go to fridge 1", prompt)
        self.assertIn("ALFWorld", prompt)
        self.assertNotIn("FlightSearch", prompt)
        self.assertNotIn("TravelPlanner", prompt)

    def test_round_templates_are_available_but_have_a_distinct_hash(self):
        events_prompt = generate_events_from_trajectory_prompt("task", "trajectory")
        workflow_prompt = generate_workflow_from_events_prompt("task", [{"step": 1}])

        self.assertIn("Output only the JSON", events_prompt[1]["content"])
        self.assertIn("critical step ids", workflow_prompt[1]["content"])
        self.assertNotEqual(
            get_prompt_spec("direct").sha256,
            get_prompt_spec("round").sha256,
        )

    def test_manifest_records_prompt_provenance(self):
        spec = get_prompt_spec("direct")
        manifest = build_prompt_manifest(
            "direct",
            build_model="builder-model",
            trajectory_file="/data/alfworld_format_traj.json",
            trajectory_count=300,
        )

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["build_model"], "builder-model")
        self.assertEqual(manifest["trajectory_file"], "/data/alfworld_format_traj.json")
        self.assertEqual(manifest["trajectory_count"], 300)
        self.assertEqual(manifest["prompt"]["version"], PROMPT_VERSION)
        self.assertEqual(manifest["prompt"]["source"], PROMPT_SOURCE)
        self.assertEqual(manifest["prompt"]["approval"], PROMPT_APPROVAL)
        self.assertEqual(prompt_manifest_mismatches(manifest, spec), [])

        manifest["prompt"]["sha256"] = "legacy"
        self.assertEqual(prompt_manifest_mismatches(manifest, spec), ["sha256"])


if __name__ == "__main__":
    unittest.main()
