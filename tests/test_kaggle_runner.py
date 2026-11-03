"""Check staged Kaggle payload hashes and extraction boundaries without network calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PATH = Path(__file__).resolve().parents[1] / 'scripts/kaggle/runner.py'
SPEC = importlib.util.spec_from_file_location('kaggle_runner', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class KagglePayloadTest(unittest.TestCase):
    def test_python_environment_is_pinned_independently_of_the_kaggle_frontend(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(RUNNER.subprocess, 'run') as run:
                result = RUNNER.create_inference_environment(Path(tmp))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn('uv==' + RUNNER.UV_VERSION, commands[0])
            self.assertEqual(commands[1][-3:], ['python', 'install', '3.11.14'])
            self.assertIn('3.11.14', commands[2])
            self.assertEqual(result, Path(tmp) / 'inference-env/bin/python')

    def test_secret_lookup_accepts_the_existing_mixed_case_label(self):
        def secret(name):
            if name != 'HF_Token':
                raise KeyError(name)
            return 'synthetic-token'
        value, source = RUNNER.read_hf_token({}, secret)
        self.assertEqual(value, 'synthetic-token')
        self.assertEqual(source, 'Kaggle secret:HF_Token')

    def test_secret_lookup_accepts_big_data_project_and_environment(self):
        secret = lambda name: 'synthetic-token' if name == 'big_data_project' else None
        self.assertEqual(RUNNER.read_hf_token({}, secret)[1], 'Kaggle secret:big_data_project')
        self.assertEqual(RUNNER.read_hf_token({'HF_TOKEN': 'env-token'}, secret),
                         ('env-token', 'environment:HF_TOKEN'))

    def test_missing_secret_error_does_not_include_a_token(self):
        with self.assertRaisesRegex(ValueError, 'Kaggle secret retrieval failed'):
            RUNNER.read_hf_token({}, lambda name: None)

    def test_secret_failure_preserves_error_type_and_redacts_token_values(self):
        def secret(name):
            raise RuntimeError('backend failed for hf_examplePrivateValue123456')
        with self.assertRaises(ValueError) as raised:
            RUNNER.read_hf_token({}, secret)
        self.assertIn('RuntimeError', str(raised.exception))
        self.assertNotIn('examplePrivateValue', str(raised.exception))

    def payload(self, dataset, name='data/processed/prompts/pilot_v5/example.json'):
        archive = dataset / 'pilot_v5.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr(name, 'fixture')
        metadata = {'archive_sha256': RUNNER.checksum(archive),
                    'files': {name: hashlib.sha256(b'fixture').hexdigest()}}
        (dataset / 'payload.json').write_text(json.dumps(metadata))

    def test_valid_payload_and_changed_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, project = Path(tmp) / 'dataset', Path(tmp) / 'project'
            dataset.mkdir()
            project.mkdir()
            self.payload(dataset)
            RUNNER.install_payload(dataset, project)
            self.assertEqual((project / 'data/processed/prompts/pilot_v5/example.json').read_text(), 'fixture')
            with (dataset / 'pilot_v5.zip').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'ZIP checksum'):
                RUNNER.install_payload(dataset, project)

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, project = Path(tmp) / 'dataset', Path(tmp) / 'project'
            dataset.mkdir()
            project.mkdir()
            self.payload(dataset, '../outside.json')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                RUNNER.install_payload(dataset, project)


if __name__ == '__main__':
    unittest.main()
