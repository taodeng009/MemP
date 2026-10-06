import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/vdar_edge_capability'))
import online_support_smoke as smoke


def make_source(path, invert=False):
    specs = {index: (family, query) for family, group, index, query, reason in smoke.CASES}
    rows = []
    for i in range(134):
        family, query = specs.get(i, ('pick_and_place_simple', 'put a cup in cabinet.'))
        count = 0 if i == 13 else 1
        rows.append({'task_id': f'json_2.1.1/valid_unseen/{family}-Object-None-Receptacle-{i}/trial/game.tw-pddl',
                     'task_index': i, 'query': query, 'condition': 'online_construction_fifo_shortest_first',
                     'interval_id': i//20, 'policy': 'fifo_shortest_first', 'available_memory_count': 30,
                     'retrieved_count': count, 'retrieved_memory_ids': [f'memory_{i}'] if count else [],
                     'retrieved_memories': [{'rank': 1, 'workflow': 'Locate the object and execute the applicable procedure.',
                                            'score': 'FORBIDDEN_SCORE'}] if count else [],
                     'reward': bool(i % 2) != invert, 'trajectory': 'FORBIDDEN_TRAJECTORY'})
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')


class OnlineSupportTests(unittest.TestCase):
    def test_outcome_blind_selection_and_separate_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.jsonl'
            make_source(source)
            tasks = smoke.select_inputs(source)
            self.assertEqual(len(tasks), 12)
            self.assertEqual([(r['task_family'], r['support_group']) for r in tasks],
                             [(f, g) for f, g, *_ in smoke.CASES])
            self.assertNotIn('reward', json.dumps(tasks))
            self.assertNotIn('FORBIDDEN', json.dumps(tasks))
            make_source(source, invert=True)
            self.assertEqual(smoke.select_inputs(source), tasks)
            with contextlib.redirect_stdout(io.StringIO()):
                smoke.prepare(source, Path(directory), tasks)
            inputs = json.loads((Path(directory)/'inputs.json').read_text(encoding='utf-8'))
            self.assertNotIn('labels', inputs)
            self.assertNotIn('actual_outcome', json.dumps(inputs))
            self.assertEqual(len(json.loads((Path(directory)/'display_labels.json').read_text())['labels']), 12)
            bad = json.loads(source.read_text().splitlines()[94])
            bad['query'] = 'different task'
            rows = source.read_text().splitlines(); rows[94] = json.dumps(bad)
            source.write_text('\n'.join(rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                smoke.select_inputs(source)

    def test_score_twelve_only_resume_and_display_after_completion(self):
        text = '<summary>\noverall_difficulty: medium\nprimary_dimensions: [object_localization]\ndifficulty_profile: Locate the object using partial procedural support.\n</summary>'
        payload = json.dumps({'choices': [{'message': {'content': text}}]}).encode()
        env = {'MEMORY_BUILD_MODEL_NAME': 'test-model', 'MEMORY_BUILD_API_KEY': 'test-key',
               'MEMORY_BUILD_API_BASE_URL': 'http://example.invalid/v1'}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); source = directory/'source.jsonl'
            make_source(source); tasks = smoke.select_inputs(source)
            args = SimpleNamespace(output_dir=directory, env_file=directory/'absent.env', timeout=5)
            with patch.dict('os.environ', env, clear=True), contextlib.redirect_stdout(io.StringIO()):
                smoke.prepare(source, directory, tasks)
                # Scoring cannot depend on this label file.
                labels = (directory/'display_labels.json').read_bytes()
                (directory/'display_labels.json').unlink()
                with patch.object(smoke.urllib.request, 'urlopen', side_effect=[io.BytesIO(payload), OSError('offline')]):
                    with self.assertRaises(RuntimeError):
                        smoke.score(args, tasks)
                with self.assertRaises(ValueError):
                    smoke.display(directory, tasks)
                with patch.object(smoke.urllib.request, 'urlopen', side_effect=lambda *a, **kw: io.BytesIO(payload)) as request:
                    smoke.score(args, tasks)
                    self.assertEqual(request.call_count, 11)
                    for call in request.call_args_list:
                        messages = json.loads(call.args[0].data)['messages']
                        user = messages[1]['content']
                        for field in ['support_group', 'memory_state', 'actual_outcome', 'FORBIDDEN']:
                            self.assertNotIn(field, user)
                with patch.object(smoke.urllib.request, 'urlopen') as request:
                    smoke.score(args, tasks)
                    request.assert_not_called()
                (directory/'display_labels.json').write_bytes(labels)
                smoke.display(directory, tasks)
                results = json.loads((directory/'display_results.json').read_text(encoding='utf-8'))
                self.assertEqual(len(results['cases']), 12)
                self.assertTrue(all(r['actual_outcome'] in ['success', 'failure'] for r in results['cases']))


if __name__ == '__main__':
    unittest.main()
