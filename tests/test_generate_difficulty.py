import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
import sys
import io
import contextlib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_difficulty', ROOT / 'experiments/vdar_edge_capability/generate_difficulty.py')
runner = importlib.util.module_from_spec(spec)
prompt_spec = importlib.util.spec_from_file_location('difficulty_prompts', ROOT / 'experiments/vdar_edge_capability/difficulty_prompts.py')
prompts = importlib.util.module_from_spec(prompt_spec)
prompt_spec.loader.exec_module(prompts)
with patch.dict(sys.modules, {'difficulty_prompts': prompts}):
    batch_spec = importlib.util.spec_from_file_location('difficulty_batch', ROOT / 'experiments/vdar_edge_capability/difficulty_batch.py')
    batch = importlib.util.module_from_spec(batch_spec)
    batch_spec.loader.exec_module(batch)
    with patch.dict(sys.modules, {'difficulty_batch': batch}):
        spec.loader.exec_module(runner)


class DifficultyTests(unittest.TestCase):
    def test_alfworld_registered_and_v2_preserved(self):
        self.assertEqual(prompts.PROMPT_REGISTRY['alfworld'], prompts.ALFWORLD_SYSTEM_PROMPT)
        self.assertEqual(prompts.PROMPT_REGISTRY['v2'], prompts.V2_SYSTEM_PROMPT)
        self.assertIn('object_localization', prompts.ALFWORLD_SYSTEM_PROMPT)
        self.assertIn('Do not assume a particular environment layout', prompts.ALFWORLD_SYSTEM_PROMPT)

    def test_loads_all_tasks_without_outcome_fields(self):
        rows = runner.load_tasks(ROOT / 'experiments/vdar_edge_capability/outputs/edge_capability_dataset.csv')
        self.assertEqual(len(rows), 134)
        self.assertEqual(len({r['task_id'] for r in rows}), 134)
        self.assertTrue(all(set(r) == {'task_id', 'task_instruction'} for r in rows))

    def test_validation_resume_and_statistics(self):
        text = '<summary>\noverall_difficulty: medium\nprimary_dimensions: [navigation, object_localization]\ndifficulty_profile: Find the object and navigate to the destination.\n</summary>'
        valid = batch.parse_summary(text)
        self.assertEqual(valid['status'], 'success')
        self.assertEqual(batch.parse_summary('')['status'], 'empty')
        self.assertEqual(batch.parse_summary('<summary></summary>')['status'], 'empty')
        self.assertEqual(batch.parse_summary(text.replace('navigation', 'coding'))['status'], 'invalid')
        self.assertEqual(batch.parse_summary(text.replace('medium', 'extreme'))['status'], 'invalid')
        config = {'model': 'test', 'prompt_version': 'alfworld'}
        tasks = [{'task_id': str(i), 'task_instruction': 'query'} for i in range(3)]
        records = [{**t, **valid, 'response_text': text, 'configuration': config} for t in tasks[:2]]
        records.append({**tasks[2], **batch.parse_summary('bad'), 'response_text': 'bad', 'configuration': config})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.jsonl'
            batch.save_records(path, records)
            self.assertEqual(batch.load_existing(path, tasks, config), records)
            with self.assertRaises(ValueError):
                batch.load_existing(path, tasks, {'model': 'other'})
            batch.save_records(path, records + records[:1])
            with self.assertRaises(ValueError):
                batch.load_existing(path, tasks, config)
        stats = batch.summarize(records)
        self.assertEqual(stats['successful_generation_count'], 2)
        self.assertEqual(stats['invalid_summary_count'], 1)
        self.assertEqual(stats['overall_difficulty']['medium'], 2)
        self.assertEqual(stats['primary_dimensions']['navigation'], 2)
        self.assertEqual(stats['identical_summary_duplicate_extra_records'], 1)

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

    def test_interrupted_api_batch_resumes_and_completed_batch_makes_no_calls(self):
        text = '<summary>\noverall_difficulty: low\nprimary_dimensions: [navigation]\ndifficulty_profile: Navigate to the target receptacle.\n</summary>'
        payload = json.dumps({'choices': [{'message': {'content': text}}]}).encode()
        environment = {'MEMORY_BUILD_MODEL_NAME': 'test-model', 'MEMORY_BUILD_API_KEY': 'test-key',
                       'MEMORY_BUILD_API_BASE_URL': 'http://example.invalid/v1'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'profiles.jsonl'
            command = ['generate', '--env-file', str(Path(directory) / 'absent.env'), '--output', str(output)]
            with patch.dict('os.environ', environment, clear=True), patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(batch.urllib.request, 'urlopen', side_effect=[io.BytesIO(payload), io.BytesIO(payload), OSError('interrupted')]):
                    with self.assertRaises(RuntimeError):
                        runner.main()
                self.assertEqual(len(output.read_text(encoding='utf-8').splitlines()), 2)
                with patch.object(batch.urllib.request, 'urlopen', side_effect=lambda *a, **kw: io.BytesIO(payload)) as request:
                    runner.main()
                    self.assertEqual(request.call_count, 132)
                    body = json.loads(request.call_args.args[0].data)
                    self.assertEqual(set(body['messages'][1]), {'role', 'content'})
                    self.assertTrue(body['messages'][1]['content'].startswith('**Query to Analyze:**\n'))
                    self.assertNotIn('p_edge', body['messages'][1]['content'])
                with patch.object(batch.urllib.request, 'urlopen') as request:
                    runner.main()
                    request.assert_not_called()
                stats = json.loads(output.with_suffix('.stats.json').read_text())
                self.assertEqual(stats['successful_generation_count'], 134)
                self.assertEqual(stats['identical_summary_duplicate_extra_records'], 133)

                records = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
                original = records[104]
                bad = {**original, **batch.parse_summary('bad'), 'response_text': 'bad'}
                for field in ['overall_difficulty', 'primary_dimensions', 'difficulty_profile']:
                    bad.pop(field, None)
                records[104] = bad
                batch.save_records(output, records)
                before = output.read_bytes()
                with patch.object(batch.urllib.request, 'urlopen') as request:
                    runner.main()
                    request.assert_not_called()
                self.assertEqual(output.read_bytes(), before)
                with patch.object(sys, 'argv', command + ['--retry-invalid', '--dry-run']), patch.object(batch.urllib.request, 'urlopen') as request:
                    runner.main()
                    request.assert_not_called()
                self.assertEqual(output.read_bytes(), before)
                with patch.object(sys, 'argv', command + ['--retry-invalid']):
                    with patch.object(batch.urllib.request, 'urlopen', side_effect=OSError('offline')):
                        with self.assertRaises(RuntimeError):
                            runner.main()
                    self.assertEqual(output.read_bytes(), before)
                    with patch.object(batch.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as request:
                        runner.main()
                        self.assertEqual(request.call_count, 1)
                updated = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(updated), 134)
                self.assertEqual(updated[:104], records[:104])
                self.assertEqual(updated[105:], records[105:])
                self.assertEqual(updated[104]['status'], 'success')
                self.assertEqual(updated[104]['previous_attempts'], [bad])
                stats = json.loads(output.with_suffix('.stats.json').read_text())
                self.assertEqual(stats['successful_generation_count'], 134)
                self.assertEqual(stats['invalid_summary_count'], 0)
                with patch.object(sys, 'argv', command + ['--retry-invalid']), patch.object(batch.urllib.request, 'urlopen') as request:
                    runner.main()
                    request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
