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
import hashlib
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
        self.assertEqual(hashlib.sha256(prompts.ALFWORLD_SYSTEM_PROMPT.encode()).hexdigest(),
                         'eaafa216a5fc5736df37f241223d813725ad6aded214e1c19cab4bfd1e27fb4a')
        self.assertEqual(prompts.PROMPT_REGISTRY['alfworld_memory'], prompts.ALFWORLD_MEMORY_SYSTEM_PROMPT)
        self.assertIn('Do not use any historical outcome labels.', prompts.ALFWORLD_MEMORY_SYSTEM_PROMPT)

    def test_memory_input_allowlist_no_hit_and_cache_validation(self):
        tasks = [{'task_id': 'a', 'task_instruction': 'put a cup in cabinet.'},
                 {'task_id': 'b', 'task_instruction': 'put two pillow in sofa.'}]
        log = [{'task_id': t['task_id'], 'query': t['task_instruction'], 'condition': 'memory',
                'reward': 'FORBIDDEN_OUTCOME', 'trajectory': 'FORBIDDEN_TRAJECTORY',
                'retrieved_count': 3 if i == 0 else 0,
                'retrieved_memories': [{'rank': rank, 'workflow': f'Procedure body {rank}.',
                                        'score': 'FORBIDDEN_SCORE', 'task_name': 'FORBIDDEN_METADATA'}
                                       for rank in range(1, 4)] if i == 0 else []}
               for i, t in enumerate(tasks)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'logs.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in log), encoding='utf-8')
            inputs = batch.attach_retrieved_memories(tasks, path)
            messages = batch.build_messages(inputs[0], 'alfworld_memory')
            self.assertEqual(messages[1]['content'], '**Task Instruction:**\nput a cup in cabinet.\n\n'
                             '**Retrieved Procedural Memories:**\nMemory 1:\nProcedure body 1.\n\n'
                             'Memory 2:\nProcedure body 2.\n\nMemory 3:\nProcedure body 3.')
            self.assertNotIn('FORBIDDEN', json.dumps(messages))
            self.assertEqual(inputs[1]['retrieved_memories'], '(No procedural memories were retrieved.)')
            self.assertEqual(batch.build_messages(tasks[0], 'alfworld')[1]['content'],
                             '**Query to Analyze:**\nput a cup in cabinet.')
            text = '<summary>\noverall_difficulty: low\nprimary_dimensions: [object_localization]\ndifficulty_profile: Locate the object.\n</summary>'
            config = {'prompt_version': 'alfworld_memory', 'model': 'test'}
            records = [{**inputs[0], **batch.parse_summary(text), 'response_text': text,
                        'configuration': config, 'messages': messages}]
            cache = Path(directory) / 'cache.jsonl'
            batch.save_records(cache, records)
            self.assertEqual(batch.load_existing(cache, inputs, config), records)
            changed = [dict(t) for t in inputs]
            changed[0]['retrieved_memories'] = 'Changed memory support'
            with self.assertRaises(ValueError):
                batch.load_existing(cache, changed, config)
            log[0]['retrieved_count'] = 2
            path.write_text('\n'.join(json.dumps(r) for r in log), encoding='utf-8')
            with self.assertRaises(ValueError):
                batch.attach_retrieved_memories(tasks, path)

    def test_memory_mode_api_resume_without_official_agent(self):
        tasks = runner.load_tasks(ROOT / 'experiments/vdar_edge_capability/outputs/edge_capability_dataset.csv')
        logs = [{'task_id': t['task_id'], 'query': t['task_instruction'], 'condition': 'memory',
                 'reward': 'FORBIDDEN_OUTCOME', 'retrieved_count': 1,
                 'retrieved_memories': [{'rank': 1, 'workflow': 'Locate, acquire and place the object.'}]}
                for t in tasks]
        text = '<summary>\noverall_difficulty: low\nprimary_dimensions: [object_localization]\ndifficulty_profile: Locate the object.\n</summary>'
        payload = json.dumps({'choices': [{'message': {'content': text}}]}).encode()
        env = {'MEMORY_BUILD_MODEL_NAME': 'test-model', 'MEMORY_BUILD_API_KEY': 'test-key',
               'MEMORY_BUILD_API_BASE_URL': 'http://example.invalid/v1'}
        with tempfile.TemporaryDirectory() as directory:
            log_path, output = Path(directory)/'logs.jsonl', Path(directory)/'memory_profiles.jsonl'
            log_path.write_text('\n'.join(json.dumps(r) for r in logs), encoding='utf-8')
            cmd = ['generate', '--prompt-version', 'alfworld_memory', '--memory-log', str(log_path),
                   '--env-file', str(Path(directory)/'absent.env'), '--output', str(output)]
            with patch.dict('os.environ', env, clear=True), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', cmd + ['--dry-run']), patch.object(batch.urllib.request, 'urlopen') as request:
                    runner.main()
                    request.assert_not_called()
                    self.assertFalse(output.exists())
                with patch.object(sys, 'argv', cmd), patch.object(batch.urllib.request, 'urlopen',
                        side_effect=[io.BytesIO(payload), OSError('offline')]) as request:
                    with self.assertRaises(RuntimeError):
                        runner.main()
                    body = json.loads(request.call_args_list[0].args[0].data)
                    self.assertEqual(body['messages'][0]['content'], prompts.ALFWORLD_MEMORY_SYSTEM_PROMPT)
                    self.assertNotIn('FORBIDDEN', json.dumps(body))
                records = [json.loads(r) for r in output.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['configuration']['prompt_version'], 'alfworld_memory')
                with patch.object(sys, 'argv', cmd), patch.object(batch.urllib.request, 'urlopen', side_effect=OSError('offline')) as request:
                    with self.assertRaises(RuntimeError):
                        runner.main()
                    message = json.loads(request.call_args.args[0].data)['messages'][1]['content']
                    self.assertIn(tasks[1]['task_instruction'], message)

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
