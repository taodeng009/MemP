import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ProcedureMem import build_top1_memory_embeddings as builder
from ProcedureMem.analyze_top1_memory_semantic_probe import task_memory_matrix


class Top1MemoryTests(unittest.TestCase):
    def test_mean_uses_record_ids_and_actual_run_multiplicity(self):
        inputs = [{'record_id': 'a::run3'}, {'record_id': 'b::run1'},
                  {'record_id': 'a::run1'}, {'record_id': 'b::run3'},
                  {'record_id': 'a::run2'}, {'record_id': 'b::run2'}]
        vectors = np.array([[8., 2.], [3., 9.], [2., 2.],
                            [3., 9.], [2., 2.], [3., 9.]])
        result = task_memory_matrix([{'task_id': 'b'}, {'task_id': 'a'}], inputs, vectors)
        np.testing.assert_allclose(result, [[3., 9.], [4., 2.]])

    def test_server_deduplicates_resumes_and_validates_text_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output, env = root / 'inputs.jsonl', root / 'vectors.npz', root / '.env'
            rows = []
            for task in range(134):
                for run in (1, 2, 3):
                    text = f'Workflow body {task % 3}'
                    rows.append({'record_id': f't{task}::run{run}', 'task_id': f't{task}',
                                 'run': run, 'workflow': text, 'embedding_text': text,
                                 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                 'format_version': builder.FORMAT_VERSION})
            inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            env.write_text(f'EMBEDDING_MODEL_NAME={builder.MODEL}\nEMBEDDING_MODEL_KEY=test\n'
                           'EMBEDDING_MODEL_BASE_URL=http://example.invalid/v1\n', encoding='utf-8')
            command = ['builder', '--inputs', str(inputs), '--output', str(output),
                       '--env-file', str(env), '--batch-size', '2']
            with patch.object(sys, 'argv', command), patch.dict('os.environ', {}, clear=True):
                with patch.object(builder, 'request_batch', side_effect=[
                        [[1.] * 768, [2.] * 768], RuntimeError('interrupted')]):
                    with self.assertRaises(RuntimeError):
                        builder.main()
                with patch.object(builder, 'request_batch', return_value=[[3.] * 768]) as request:
                    builder.main()
                    self.assertEqual(request.call_count, 1)
                    self.assertEqual(request.call_args.args[2], ['Workflow body 2'])
                with np.load(output, allow_pickle=False) as cache:
                    self.assertEqual(cache['embeddings'].shape, (402, 768))
                    np.testing.assert_array_equal(cache['embeddings'][:9, 0], [1, 1, 1, 2, 2, 2, 3, 3, 3])
                with patch.object(builder, 'request_batch') as request:
                    builder.main()
                    request.assert_not_called()
            changed_rows = [dict(row) for row in rows]
            changed_rows[0]['embedding_text'] = 'Wrong text'
            with self.assertRaises(ValueError):
                builder.validate_cache(output, changed_rows, hashlib.sha256(inputs.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
