"""Check label denominators, immutable inputs, recovery and archive-only downloads."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/kaggle'))
import behavior_summary as behavior
import ingest_scale8 as ingest
import run_artifacts as artifacts


def synthetic_run():
    responses = [{'response_id': str(i), 'prompt_id': 'p', 'status': 'success',
                  'output_truncated': i < 2, 'output_tokens': 10 + i} for i in range(4)]
    labels = [{'response_id': str(i), 'labeler_id': 'rules', 'refusal': i % 2 == 0,
               'error': None} for i in range(4)]
    return {'context': {'prompts': [{'prompt_id': 'p', 'prompt_group_id': 'g', 'source_id': 'test',
                                    'source_labels': {'prompt_intent': 'unharmful'}, 'variant_type': 'vanilla'}],
                        'prompt_manifests': [{'group_splits': {'g': 'validation'}}]},
            'responses': responses, 'labels': labels}


class BehaviorTest(unittest.TestCase):
    def test_missing_and_error_labels_are_not_nonrefusals(self):
        run = synthetic_run()
        run['labels'][1]['error'] = 'malformed'
        run['labels'].pop()
        with patch.object(behavior, 'describe', return_value={'model': 'test'}):
            rows, associations = behavior.summarize_run(run)
        self.assertEqual(rows[0]['n'], 4)
        self.assertEqual(rows[0]['n_valid_labels'], 2)
        self.assertEqual(rows[0]['refusal_rate'], 1)
        self.assertEqual(rows[0]['truncation_rate'], .5)
        self.assertEqual(rows[0]['median_output_tokens'], 11.5)
        self.assertIsNone(associations[0]['truncation_given_nonrefused'])

    def test_conditional_truncation_and_failed_response_denominators(self):
        run = synthetic_run()
        run['responses'][3]['status'] = 'failed'
        with patch.object(behavior, 'describe', return_value={'model': 'test'}):
            rows, associations = behavior.summarize_run(run)
        self.assertEqual(rows[0]['n_failed'], 1)
        self.assertAlmostEqual(rows[0]['refusal_rate'], 2 / 3)
        self.assertEqual(associations[0]['truncation_given_refused'], .5)
        self.assertEqual(associations[0]['truncation_given_nonrefused'], 1)

    def test_checksum_corruption_is_rejected_without_rewriting_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'responses-000000.jsonl').write_text('{"response_id":"r"}\n')
            journal = folder / 'responses.checksums.json'
            journal.write_text(json.dumps({'responses-000000.jsonl': '0' * 64}))
            original = journal.read_bytes()
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                artifacts.shards(folder, 'responses', 'response_id')
            self.assertEqual(journal.read_bytes(), original)

    def test_changed_validation_export_is_still_checked(self):
        from test_inference_labels import InferenceTest
        fixture = InferenceTest()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = fixture.generate(Path(tmp))
            artifacts.load_run(folder)
            saved = folder / 'validation.bundle.json'
            bundle = artifacts.read_json(saved)
            bundle['responses'][0]['response_id'] = 'r_' + '0' * 64
            saved.write_text(json.dumps(bundle), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid response identity'):
                artifacts.load_run(folder)


class IngestionTest(unittest.TestCase):
    def test_scratch_cache_checks_exact_run_and_rejoins_fresh_labels(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'response_id': 'r', 'model_id': 'm', 'prompt_id': 'p',
                   'status': 'success', 'text': 'answer', 'output_tokens': 2, 'output_truncated': False}
            frame = pd.DataFrame([{**{k: row[k] for k in [
                'response_id', 'model_id', 'prompt_id', 'output_tokens', 'output_truncated']},
                'text_chars': 6, 'vector': [1.] + [0.] * 383, 'refusal': False}])
            cache = root / 'cache.parquet'
            frame.to_parquet(cache, index=False)
            run = {'responses': [row], 'labels': [{'response_id': 'r', 'labeler_id': 'wildguard',
                   'refusal': True, 'response_harmful': False, 'prompt_harmful': True}]}
            ingest.reuse_scratch_vectors(cache, run, root / 'stage')
            with (root / 'stage/responses.parquet').open('rb') as stream:
                saved = pd.read_parquet(stream)
            self.assertTrue(saved.iloc[0]['refusal'])
            self.assertTrue(pd.isna(saved.iloc[0]['refusal_rules']))
            run['responses'][0]['model_id'] = 'different-model'
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                ingest.reuse_scratch_vectors(cache, run, root / 'wrong')

    def test_scratch_cache_is_not_a_production_shortcut(self):
        with self.assertRaisesRegex(ValueError, 'only for an offline scratch'):
            ingest.ingest(Path('unused'), Path('unused'), archive=Path('unused'),
                          reuse_flat=Path('unused'))

    def test_download_requests_only_archive_and_exact_version(self):
        api = MagicMock()
        response = MagicMock()
        response.iter_content.return_value = [b'\x1f\x8bexample']
        client = api.build_kaggle_client.return_value.__enter__.return_value
        endpoint = client.kernels.kernels_api_client.download_kernel_output
        endpoint.return_value = response
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'pilot_outputs.tar.gz'
            with patch('kaggle.api.kaggle_api_extended.KaggleApi', return_value=api):
                ingest.download('owner/notebook', 7, target)
            request = endpoint.call_args.args[0]
            self.assertEqual(request.file_path, 'pilot_outputs.tar.gz')
            self.assertEqual(request.version_number, 7)
            self.assertEqual(target.read_bytes(), b'\x1f\x8bexample')
            response.close.assert_called_once()

    def test_existing_human_audit_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / 'manual.csv', root / 'pilot_audit.csv'
            source.write_text('fresh audit')
            destination.write_text('owner annotations')
            saved = Path(ingest.copy_audit(source, destination))
            self.assertEqual(destination.read_text(), 'owner annotations')
            self.assertEqual(saved.read_text(), 'fresh audit')

    def test_partial_archive_is_validated_and_only_complete_runs_staged(self):
        def extract(archive, destination):
            for name in ['complete', 'partial']:
                folder = destination / name
                folder.mkdir(parents=True)
                (folder / 'run.bundle.json').write_text('{}')

        def load(folder):
            return {'folder': folder, 'stats': {'completed': folder.name == 'complete'},
                    'context': {'run_manifests': [{'run_id': 'run_' + folder.name,
                        'runtime': {'configuration': {'model_id': folder.name}}}]},
                    'responses': [], 'labels': []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive';archive.write_bytes(b'local')
            with patch('summarize_pilot.extract_archive', side_effect=extract), \
                    patch.object(ingest, 'load_run', side_effect=load), \
                    patch.object(ingest, 'flatten') as flatten, \
                    patch('behavior_summary.build'), redirect_stdout(io.StringIO()):
                result = ingest.ingest(root / 'out', root, archive, scratch=True)
            self.assertEqual(flatten.call_count, 1)
            self.assertEqual(result['runs'][1]['staging'], 'skipped_incomplete_generation')
            self.assertEqual(result['sha256'], hashlib.sha256(b'local').hexdigest())
            self.assertFalse((root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
