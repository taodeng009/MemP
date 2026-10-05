import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alfworld_capability_adapter', ROOT / 'llm_verifier/alfworld_capability.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CapabilityVerifierTests(unittest.TestCase):
    def test_prompt_allowlist_does_not_leak_metadata(self):
        cases = runner.load_cases(ROOT / 'llm_verifier/smoke_prefixes.json')
        self.assertEqual(len(cases), 6)
        case = dict(cases[0], final_reward='SECRET_OUTCOME', future_actions='SECRET_FUTURE', total_length='SECRET_LENGTH')
        text = runner.prompt_for(case)
        self.assertIn(runner.CRITERION, text)
        self.assertIn(case['task'], text)
        self.assertNotIn('SECRET_', text)
        self.assertNotIn(case['id'], text)
        self.assertNotIn('expected_group', text)
        self.assertIn('<score_A>', text)

    def test_official_decoder_used_and_no_missing_logprob_fallback(self):
        called = []
        api = types.SimpleNamespace(
            call_verifier=lambda *a, **kw: ('<score_A> A </score_A>', ['tag', 'A'], [[('A', 0.)]]),
            _find_tag_logprobs=lambda *a: [(' A', -.5), ('T', -1.), ('invalid', -2.)],
            SCALE={'valid_tokens': {'A': 20., 'T': 1.}},
            extract_score=lambda *a: called.append(a) or .73)
        record = runner.official_score(api, 'prompt', object(), 'model')
        self.assertEqual(record['score'], .73)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(record['recognized_score_alternatives']), 2)
        api._find_tag_logprobs = lambda *a: None
        with self.assertRaises(RuntimeError):
            runner.official_score(api, 'prompt', object(), 'model')


if __name__ == '__main__':
    unittest.main()
