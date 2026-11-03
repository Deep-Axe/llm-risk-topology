"""Check staged Kaggle payload hashes and extraction boundaries without network calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

PATH = Path(__file__).resolve().parents[1] / 'scripts/kaggle/runner.py'
SPEC = importlib.util.spec_from_file_location('kaggle_runner', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class KagglePayloadTest(unittest.TestCase):
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
