import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from ProcedureMem.runtime_config import configure_runtime, load_environment
except ModuleNotFoundError:
    configure_runtime = None
    load_environment = None


@unittest.skipIf(load_environment is None, "runtime configuration dependencies are not installed")
class RuntimeConfigTests(unittest.TestCase):
    def test_dotenv_does_not_override_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "MODEL_NAME=dotenv-model\nOPENAI_API_KEY=dotenv-key\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MODEL_NAME": "process-model"}, clear=True):
                loaded_path = load_environment(env_file)

                self.assertEqual(loaded_path, env_file.resolve())
                self.assertEqual(os.environ["MODEL_NAME"], "process-model")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "dotenv-key")

    def test_agent_role_variables_populate_legacy_aliases(self):
        values = {
            "AGENT_MODEL_NAME": "openai/local-model",
            "AGENT_API_KEY": "local-key",
            "AGENT_API_BASE_URL": "http://127.0.0.1:8000/v1",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = configure_runtime(require_llm=True)

            self.assertEqual(settings.model_name, "openai/local-model")
            self.assertEqual(os.environ["MODEL_NAME"], "openai/local-model")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local-key")
            self.assertEqual(
                os.environ["OPENAI_API_BASE"],
                "http://127.0.0.1:8000/v1",
            )


if __name__ == "__main__":
    unittest.main()
