"""Offline tests: no real embedding or generation services are called."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/vdar_edge_capability'))
import vdar_feasibility as vdar


def summary(index=0):
    return ('<summary>\noverall_difficulty: medium\n'
            'primary_dimensions: [object_localization, receptacle_interaction]\n'
            f'difficulty_profile: Locate object type {index} and place it in the receptacle.\n</summary>')


def fixture(directory):
    directory = Path(directory)
    tasks, profiles = [], []
    for i in range(134):
        instruction = f'put object {i//2} in shelf.'
        s = i % 4
        tasks.append({'task_id': f'task_{i:03}', 'task_instruction': instruction,
                      'canonical_query': instruction[:-1], 'success_run1': int(s >= 1),
                      'success_run2': int(s >= 2), 'success_run3': int(s >= 3), 'p_edge': s/3})
        raw = summary(i % 8)
        profiles.append({'task_id': tasks[-1]['task_id'], 'task_instruction': instruction,
                         'response_text': raw, **vdar.parse_summary(raw)})
    raw = profiles[104]['response_text'].replace('[object_localization, receptacle_interaction]',
                                                 'object_localization, receptacle_interaction')
    profiles[104].update(response_text=raw, **vdar.parse_summary(raw))
    dataset, profile_path = directory/'dataset.csv', directory/'profiles.jsonl'
    vdar.write_csv(dataset, tasks)
    profile_path.write_text('\n'.join(json.dumps(r) for r in profiles), encoding='utf-8')
    return dataset, profile_path


class VdarTests(unittest.TestCase):
    def test_only_bracket_repair_is_allowed(self):
        raw = summary().replace('[object_localization, receptacle_interaction]',
                                'object_localization, receptacle_interaction')
        record = {'task_id': 'x', 'response_text': raw, **vdar.parse_summary(raw)}
        text, normalized = vdar.embedding_text(record)
        self.assertTrue(normalized)
        self.assertEqual(text, vdar.parse_summary(summary())['difficulty_summary'])
        self.assertEqual(record['response_text'], raw)
        for bad in [raw.replace('object_localization', 'coding'), raw.replace('medium', 'extreme'), '',
                    raw + '\nextra text']:
            record = {'task_id': 'x', 'response_text': bad, **vdar.parse_summary(bad)}
            with self.assertRaises(ValueError):
                vdar.embedding_text(record)

    def test_load_alignment_and_label_free_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, profiles = fixture(directory)
            tasks, identity = vdar.load_inputs(dataset, profiles)
            self.assertEqual(len(tasks), 134)
            self.assertEqual(sum(r['bracket_normalized'] for r in identity['inputs']), 1)
            self.assertNotIn('p_edge', json.dumps(identity))
            tasks[0]['success_run1'] = tasks[0]['success_run2'] = tasks[0]['success_run3'] = 1
            tasks[0]['p_edge'] = 1
            vdar.write_csv(dataset, tasks)
            self.assertEqual(vdar.load_inputs(dataset, profiles)[1], identity)
            tasks[0]['canonical_query'] = 'wrong'
            vdar.write_csv(dataset, tasks)
            with self.assertRaises(ValueError):
                vdar.load_inputs(dataset, profiles)

    def test_retrieval_formula_oof_and_no_query_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, profiles = fixture(directory)
            tasks, _ = vdar.load_inputs(dataset, profiles)
            matrix = np.zeros((134, 768)); matrix[:, 0] = 1
            assignments = np.array([(i//2) % 5 for i in range(134)])
            splits = [(np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)) for fold in range(5)]
            predictions, fold_ids, neighbors, folds = vdar.retrieve_oof(tasks, matrix, splits)
            self.assertEqual(len(folds), 5)
            self.assertEqual(len(neighbors), 134*13)
            self.assertTrue(np.all(fold_ids >= 1))
            for i, task in enumerate(tasks):
                for k in vdar.KS:
                    evidence = [r for r in neighbors if r['task_id'] == task['task_id'] and r['k'] == k]
                    self.assertEqual(len(evidence), k)
                    self.assertAlmostEqual(predictions[k][i], sum(r['neighbor_p_edge'] for r in evidence)/k)
                    self.assertTrue(all(r['neighbor_canonical_query'] != task['canonical_query'] for r in evidence))
            # Test labels may affect evaluation, never their own prediction.
            tasks[0]['p_edge'] = 1 - tasks[0]['p_edge']
            changed = vdar.retrieve_oof(tasks, matrix, splits)[0]
            self.assertEqual(predictions[3][0], changed[3][0])
            # Nonzero squared-L2: orthogonal test vector has distance 2, similarity 1/3.
            matrix[0] = 0; matrix[0, 1] = 1
            altered = vdar.retrieve_oof(tasks, matrix, splits)
            evidence = [r for r in altered[2] if r['task_id'] == tasks[0]['task_id'] and r['k'] == 3]
            self.assertTrue(all(r['distance'] == 2 and abs(r['similarity']-1/3) < 1e-12 for r in evidence))
            self.assertAlmostEqual(altered[0][3][0], sum(r['neighbor_p_edge']/3 for r in evidence)/3)
            with self.assertRaises(ValueError):
                vdar.retrieve_oof(tasks, matrix, [(np.arange(1, 134), np.array([0]))])

    def test_curves_and_metrics(self):
        tasks = [{'task_id': str(i), 'p_edge': p} for i, p in enumerate([0, 1/3, 2/3, 1])]
        predictions = {k: np.array([0, 1/3, 2/3, 1]) for k in vdar.KS}
        curves = vdar.failure_curves(tasks, predictions)
        self.assertAlmostEqual(curves['total_failure_mass'], 2)
        self.assertEqual(curves['captured_failure_mass']['random_expectation'], [0, .5, 1, 1.5, 2])
        self.assertEqual(curves['captured_failure_mass']['vdar_k3'], curves['captured_failure_mass']['oracle'])
        for curve in curves['captured_failure_mass'].values():
            self.assertEqual(curve[0], 0)
            self.assertAlmostEqual(curve[-1], 2)
        p = predictions[3]
        self.assertAlmostEqual(vdar.metrics(p, p)['spearman'], 1)
        self.assertEqual(vdar.metrics(p, p)['mae'], 0)
        self.assertIsNone(vdar.metrics(p, np.zeros(4))['spearman'])

    def test_embedding_checkpoint_resume_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, profiles = fixture(directory)
            _, identity = vdar.load_inputs(dataset, profiles)
            args = SimpleNamespace(output_dir=Path(directory), env_file=Path(directory)/'absent.env',
                                   batch_size=3, timeout=5, retries=0)
            env = {'EMBEDDING_MODEL_NAME': vdar.MODEL, 'EMBEDDING_MODEL_BASE_URL': 'http://example.invalid/v1',
                   'EMBEDDING_MODEL_KEY': 'test-key'}
            def vectors(*args):
                return [[1.0] + [0.0]*767 for _ in args[2]]
            with patch.dict('os.environ', env, clear=True):
                with patch.object(vdar, 'request_embeddings', side_effect=[vectors(None, None, [1, 2, 3]), RuntimeError('offline')]):
                    with self.assertRaises(RuntimeError):
                        vdar.embed(args, identity)
                self.assertFalse((Path(directory)/'difficulty_embeddings.npy').exists())
                with patch.object(vdar, 'request_embeddings', side_effect=vectors) as request:
                    vdar.embed(args, identity)
                    self.assertEqual(sum(len(call.args[2]) for call in request.call_args_list), 5)
                matrix = vdar.load_embeddings(Path(directory)/'difficulty_embeddings.npy', identity)
                self.assertEqual(matrix.shape, (134, 768))
                with patch.object(vdar, 'request_embeddings') as request:
                    vdar.embed(args, identity)
                    request.assert_not_called()
                identity['inputs'][0]['text'] = 'changed'
                with self.assertRaises(ValueError):
                    vdar.load_embeddings(Path(directory)/'difficulty_embeddings.npy', identity)

    def test_embedding_response_indices_and_validation(self):
        vector = [1.0] + [0.0]*767
        payload = {'model': vdar.MODEL, 'data': [{'index': 1, 'embedding': vector}, {'index': 0, 'embedding': vector}]}
        with patch.object(vdar.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as request:
            values = vdar.request_embeddings('http://example.invalid/embeddings', 'key', ['a', 'b'], 5, 0)
            self.assertEqual(len(values), 2)
            self.assertEqual(json.loads(request.call_args.args[0].data), {'model': vdar.MODEL, 'input': ['a', 'b']})
        payload['data'][1]['index'] = 1
        with patch.object(vdar.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(ValueError):
                vdar.request_embeddings('http://example.invalid/embeddings', 'key', ['a', 'b'], 5, 0)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn not installed on this host')
    def test_real_groupkfold_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, profiles = fixture(directory)
            tasks, identity = vdar.load_inputs(dataset, profiles)
            path = Path(directory)/'difficulty_embeddings.npy'
            matrix = np.zeros((134, 768), dtype=np.float32); matrix[:, 0] = 1
            np.save(path, matrix)
            vdar.atomic_json(path.with_suffix('.manifest.json'), {'identity': identity, 'npy_sha256': vdar.digest(path.read_bytes())})
            args = SimpleNamespace(output_dir=Path(directory), dataset=dataset, profiles=profiles)
            vdar.evaluate(args, tasks, identity)
            report = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(len(report['folds']), 5)
            self.assertEqual(report['task_total'], 134)
            self.assertEqual(len((Path(directory)/'oof_predictions.csv').read_text().splitlines()), 135)
            self.assertEqual(len((Path(directory)/'neighbors.csv').read_text().splitlines()), 134*13+1)
            self.assertTrue((Path(directory)/'failure_capture_curve.svg').exists())


if __name__ == '__main__':
    unittest.main()
