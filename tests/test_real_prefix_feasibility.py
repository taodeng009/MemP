import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load('prefix_smoke_adapter', ROOT / 'llm_verifier/alfworld_capability.py')
with patch.dict(sys.modules, {'alfworld_capability': adapter}):
    runner = load('real_prefix_adapter', ROOT / 'llm_verifier/real_prefix_feasibility.py')


class RealPrefixTests(unittest.TestCase):
    def test_label_hash_allows_only_line_ending_conversion(self):
        lf = b'[{"sample_id":"s1","final_success":1}]\n'
        crlf = lf.replace(b'\n', b'\r\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.json'
            for data, original in [(lf, lf), (lf, crlf), (crlf, lf)]:
                path.write_bytes(data)
                result = runner.verify_label_hash(path, hashlib.sha256(original).hexdigest())
                self.assertEqual(result, 'exact_bytes' if data == original else 'line_endings_only')
            path.write_bytes(lf.replace(b'"final_success":1', b'"final_success":0'))
            with self.assertRaises(ValueError):
                runner.verify_label_hash(path, hashlib.sha256(crlf).hexdigest())

    def test_only_first_three_action_observation_pairs(self):
        record = {'query': 'put a mug on desk.', 'actions': ['a1', 'a2', 'a3'],
                  'reward': 'SECRET_LABEL', 'termination_reason': 'SECRET_TERMINATION',
                  'trajectory': [{'from': 'human', 'value': 'You see a desk.\nYour task is to: put a mug on desk.\n\nHere are some guidelines for solving similar tasks:\nSECRET_MEMORY'}]}
        for j in range(1, 4):
            record['trajectory'] += [{'from': 'gpt', 'value': f'Thought: SECRET_THOUGHT\nAction: a{j}'},
                                     {'from': 'human', 'value': f'Observation: o{j}'}]
        record['trajectory'].append({'from': 'gpt', 'value': 'SECRET_FUTURE'})
        initial, steps = runner.first_three(record)
        self.assertEqual(initial, 'You see a desk.')
        self.assertEqual(steps[-1], {'action': 'a3', 'observation': 'o3'})
        for length in range(4):
            case = {'task': record['query'], 'initial_observation': initial, 'prefix': steps[:length]}
            prompt = adapter.prompt_for(case)
            self.assertNotIn('SECRET_', prompt)
            self.assertEqual(prompt.count('Observed action:'), length)
            self.assertIn(adapter.CRITERION, prompt)

    def test_schema_rejects_label_and_non_nested_prefix(self):
        rows = [{'sample_id': f's{i}', 'L': length, 'task': 'task',
                 'initial_observation': 'initial',
                 'prefix': [{'action': f'a{j}', 'observation': f'o{j}'} for j in range(length)]}
                for i in range(30) for length in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inputs.json'
            path.write_text(json.dumps(rows), encoding='utf-8')
            self.assertEqual(len(runner.load_inputs(path)), 120)
            rows[0]['final_success'] = 1
            path.write_text(json.dumps(rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_inputs(path)
            del rows[0]['final_success']
            rows[1]['prefix'][0]['observation'] = 'WRONG'
            path.write_text(json.dumps(rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_inputs(path)


if __name__ == '__main__':
    unittest.main()
