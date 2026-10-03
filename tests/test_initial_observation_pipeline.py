import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ProcedureMem import build_initial_observation_embeddings as builder
from ProcedureMem.prepare_initial_observation_inputs import extract_initial_observation, MEMORY_MARKER


class InitialObservationTests(unittest.TestCase):
    def test_extract_uses_initial_environment_and_removes_memory_and_goal(self):
        record = {
            'task_id': 'task', 'query': 'put a mug on desk.',
            'trajectory': [
                {'from': 'human', 'value': 'You see a desk 1.\nYour task is to: put a mug on desk.'
                 + MEMORY_MARKER + '[{"guidelines": "Do not include this"}]'},
                {'from': 'gpt', 'value': 'Action: go to desk 1'},
                {'from': 'human', 'value': 'Observation: task completed'},
            ],
        }
        self.assertEqual(extract_initial_observation(record), 'You see a desk 1.')
        record['query'] = 'different task'
        with self.assertRaises(ValueError):
            extract_initial_observation(record)

    def test_server_builder_resumes_and_final_cache_matches_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, env = root / 'inputs.jsonl', root / 'embeddings.npz', root / '.env'
            rows = []
            for i in range(134):
                query = f'task {i % 3}'
                text = f'Task: {query}\nInitial observation: You see a desk.'
                rows.append({'task_id': f'task-{i}', 'task_index': i, 'query': query,
                             'initial_observation': 'You see a desk.', 'embedding_text': text,
                             'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'format_version': builder.FORMAT_VERSION})
            inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            env.write_text(f'EMBEDDING_MODEL_NAME={builder.MODEL}\nEMBEDDING_MODEL_KEY=test-key\nEMBEDDING_MODEL_BASE_URL=http://example.invalid/v1\n', encoding='utf-8')
            fake_tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: list(range(20)))
            transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_tokenizer))
            command = ['builder', '--inputs', str(inputs), '--output', str(output),
                       '--env-file', str(env), '--tokenizer-path', str(root), '--batch-size', '2']
            with patch.object(sys, 'argv', command), patch.dict(sys.modules, {'transformers': transformers}), patch.dict('os.environ', {}, clear=True):
                with patch.object(builder, 'request_batch', side_effect=[[[1.0] * 768, [2.0] * 768], RuntimeError('interrupted')]):
                    with self.assertRaises(RuntimeError):
                        builder.main()
                checkpoint = json.loads(output.with_name(output.name + '.checkpoint.json').read_text())
                self.assertEqual(len(checkpoint['vectors']), 2)
                with patch.object(builder, 'request_batch', return_value=[[3.0] * 768]) as request:
                    builder.main()
                    self.assertEqual(request.call_count, 1)
                    self.assertEqual(len(request.call_args.args[2]), 1)
                with np.load(output, allow_pickle=False) as cache:
                    self.assertEqual(cache['embeddings'].shape, (134, 768))
                    self.assertEqual(list(cache['task_ids']), [row['task_id'] for row in rows])
                    self.assertEqual(list(cache['text_sha256']), [row['text_sha256'] for row in rows])
                    np.testing.assert_array_equal(cache['embeddings'][:3, 0], [1, 2, 3])
                with patch.object(builder, 'request_batch') as request:
                    builder.main()
                    request.assert_not_called()
                # Default endpoint-only mode must not load transformers/tokenizer.
                endpoint_output = root / 'endpoint_only.npz'
                endpoint_command = ['builder', '--inputs', str(inputs), '--output', str(endpoint_output),
                                    '--env-file', str(env)]
                forbidden_transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Tokenizer must be optional'))))
                with patch.object(sys, 'argv', endpoint_command), patch.dict(sys.modules, {'transformers': forbidden_transformers}), \
                        patch.object(builder, 'request_batch', return_value=[[1.0] * 768] * 3):
                    builder.main()
                with np.load(endpoint_output, allow_pickle=False) as cache:
                    self.assertTrue(np.all(cache['token_counts'] == -1))
                audit = json.loads(endpoint_output.with_name(endpoint_output.name + '.tokens.json').read_text())
                self.assertFalse(audit['local_token_check_performed'])


if __name__ == '__main__':
    unittest.main()
