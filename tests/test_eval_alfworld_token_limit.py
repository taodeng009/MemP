import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from ProcedureMem.eval_alfworld import _make_llm, build_parser
except ModuleNotFoundError:
    _make_llm = None
    build_parser = None


@unittest.skipIf(build_parser is None, "ALFWorld evaluation dependencies are not installed")
class AgentTokenLimitTests(unittest.TestCase):
    def test_parser_defaults_agent_max_tokens_to_1024(self):
        args = build_parser().parse_args(["--condition", "no_memory"])
        self.assertEqual(args.agent_max_tokens, 1024)

    def test_make_llm_passes_agent_completion_limit(self):
        captured = {}

        def completion(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Action: look")
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=3,
                    total_tokens=13,
                ),
                model="agent-model",
                id="request-1",
            )

        fake_litellm = SimpleNamespace(completion=completion)
        with patch.dict(sys.modules, {"litellm": fake_litellm}), patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ):
            llm, _ = _make_llm(
                "agent-model",
                temperature=0,
                seed=42,
                max_tokens=321,
            )
            result = llm([{"role": "user", "content": "test"}])

        self.assertEqual(captured["max_tokens"], 321)
        self.assertEqual(result.content, "Action: look")
        self.assertEqual(result.usage.total_tokens, 13)


if __name__ == "__main__":
    unittest.main()
