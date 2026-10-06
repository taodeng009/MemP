import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_difficulty', ROOT / 'experiments/vdar_edge_capability/generate_difficulty.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DifficultyTests(unittest.TestCase):
    def test_selects_five_types_and_does_not_carry_outcomes(self):
        rows = runner.select_tasks(ROOT / 'experiments/vdar_edge_capability/outputs/edge_capability_dataset.csv')
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({r['task_type'] for r in rows}), 5)
        self.assertTrue(all(set(r) == {'task_id', 'task_type', 'task_instruction'} for r in rows))

    def test_frozen_prompt_is_exact_official_v2(self):
        official = ROOT.parent / 'VDAR-Router/expert-5k/difficulty_aware_router/agents/difficulty_analysis.py'
        if not official.exists():
            self.skipTest('Official reference repository not available on this host')
        tree = ast.parse(official.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'V2_SYSTEM_PROMPT' for t in n.targets))
        expected = textwrap.dedent(ast.literal_eval(node.value.func.value.args[0])).strip()
        actual = (ROOT / 'experiments/vdar_edge_capability/prompts/vdar_v2_system.txt').read_text(encoding='utf-8').strip()
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
