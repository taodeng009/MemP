import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from ProcedureMem import llm_api
except ModuleNotFoundError:
    llm_api = None


@unittest.skipIf(llm_api is None, "LLM API dependencies are not installed")
class MemoryBuildGenerationSettingsTests(unittest.TestCase):
    def test_defaults_are_temperature_zero_and_seed_42(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_api.resolve_memory_build_temperature(), 0.0)
            self.assertEqual(llm_api.resolve_memory_build_seed(), 42)

    def test_environment_overrides_defaults(self):
        values = {
            "MEMORY_BUILD_TEMPERATURE": "0.25",
            "MEMORY_BUILD_SEED": "7",
        }
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(llm_api.resolve_memory_build_temperature(), 0.25)
            self.assertEqual(llm_api.resolve_memory_build_seed(), 7)

    def test_explicit_values_override_environment(self):
        values = {
            "MEMORY_BUILD_TEMPERATURE": "0.25",
            "MEMORY_BUILD_SEED": "7",
        }
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(llm_api.resolve_memory_build_temperature(0), 0.0)
            self.assertEqual(llm_api.resolve_memory_build_seed(42), 42)

    def test_request_includes_resolved_temperature_and_seed(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="workflow"))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response,
                )
            )
        )
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return response

        client.chat.completions.create = create
        with patch.object(llm_api, "_get_client", return_value=client):
            result = llm_api.get_response(
                [{"role": "user", "content": "build"}],
                model="builder",
                temperature=0,
                seed=42,
            )

        self.assertEqual(result, "workflow")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["seed"], 42)

    def test_invalid_environment_values_are_rejected(self):
        with patch.dict(
            os.environ,
            {"MEMORY_BUILD_TEMPERATURE": "nan", "MEMORY_BUILD_SEED": "4.2"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "MEMORY_BUILD_TEMPERATURE"):
                llm_api.resolve_memory_build_temperature()
            with self.assertRaisesRegex(ValueError, "MEMORY_BUILD_SEED"):
                llm_api.resolve_memory_build_seed()


if __name__ == "__main__":
    unittest.main()
