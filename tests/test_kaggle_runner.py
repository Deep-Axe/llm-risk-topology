"""Check staged Kaggle payload hashes and extraction boundaries without network calls."""
import hashlib
import io
import importlib.util
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PATH = Path(__file__).resolve().parents[1] / 'scripts/kaggle/runner.py'
SPEC = importlib.util.spec_from_file_location('kaggle_runner', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
PREPARE_SPEC = importlib.util.spec_from_file_location('kaggle_prepare', PATH.with_name('prepare.py'))
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


class KagglePayloadTest(unittest.TestCase):
    def test_subprocess_progress_heartbeat_and_final_output_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = io.StringIO()
            program = ('import sys,time; print("loading weights",flush=True); '
                       'time.sleep(0.3); print("batch 1 committed",file=sys.stderr,flush=True)')
            with redirect_stdout(captured):
                RUNNER.stream_command(sys.executable, ['-c', program], root, root,
                                      'generation', heartbeat_seconds=0.05)
            text = captured.getvalue()
            for message in ['START', 'loading weights', 'RUNNING:', 'batch 1 committed', 'COMPLETE']:
                self.assertIn(message, text)
            log = (root / 'generation.log').read_text()
            self.assertIn('loading weights', log)
            self.assertIn('batch 1 committed', log)

    def test_subprocess_failure_is_visible_and_preserves_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = io.StringIO()
            with redirect_stdout(captured):
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    RUNNER.stream_command(sys.executable,
                                          ['-c', 'import sys; print("fixture failure"); sys.exit(2)'],
                                          root, root, 'failure', heartbeat_seconds=0.05)
            self.assertEqual(raised.exception.returncode, 2)
            self.assertIn('fixture failure', captured.getvalue())
            self.assertIn('FAILED', captured.getvalue())
            self.assertNotIn('COMPLETE', captured.getvalue())

    def test_notebook_payload_is_json_and_preserves_executable_source(self):
        source = 'print("synthetic notebook")\n'
        notebook = json.loads(json.dumps(PREPARE.make_notebook(source)))
        self.assertEqual(notebook['nbformat'], 4)
        self.assertEqual(''.join(notebook['cells'][0]['source']), source)
        compile(''.join(notebook['cells'][0]['source']), 'runner.ipynb', 'exec')
        self.assertEqual(notebook['cells'][0]['outputs'], [])

    def test_network_failure_is_reported_before_secret_lookup(self):
        with patch.object(RUNNER.urllib.request, 'urlopen', side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, 'Outbound network'):
                RUNNER.check_network()

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

    def test_extracted_payload_keeps_all_file_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, project = Path(tmp) / 'dataset', Path(tmp) / 'project'
            dataset.mkdir()
            self.payload(dataset)
            with zipfile.ZipFile(dataset / 'pilot_v5.zip') as archive:
                archive.extractall(dataset / 'pilot_v5')
            (dataset / 'pilot_v5.zip').unlink()
            RUNNER.install_payload(dataset, project)
            name = 'data/processed/prompts/pilot_v5/example.json'
            self.assertEqual((project / name).read_text(), 'fixture')
            (dataset / 'pilot_v5' / name).write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'member checksum'):
                RUNNER.install_payload(dataset, Path(tmp) / 'untouched')
            self.assertFalse((Path(tmp) / 'untouched').exists())
            (dataset / 'pilot_v5' / name).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing extracted'):
                RUNNER.install_payload(dataset, project)

    def test_extracted_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, project = Path(tmp) / 'dataset', Path(tmp) / 'project'
            dataset.mkdir()
            self.payload(dataset, '../outside.json')
            (dataset / 'pilot_v5.zip').unlink()
            with self.assertRaisesRegex(ValueError, 'escapes'):
                RUNNER.install_payload(dataset, project)

    def test_both_assistants_finish_before_missing_secret_stops_wildguard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working, inputs = root / 'working', root / 'input'
            working.mkdir()
            inputs.mkdir()
            dataset = inputs / 'dataset'
            dataset.mkdir()
            self.payload(dataset)
            events = []

            def fake_run(args, **kwargs):
                if 'python/generate_responses.py' in args:
                    model = args[args.index('--model') + 1]
                    events.append(('generate', model))
                    run = working / 'project/data/processed/responses/run' / str(len(events))
                    run.mkdir(parents=True)
                    (run / 'run.bundle.json').write_text(json.dumps({'run_manifests': [{
                        'runtime': {'engine': 'vllm', 'configuration': {'model_id': model}}
                    }]}))
                if '--labeler' in args:
                    events.append(('label', args[args.index('--labeler') + 1]))

            def missing_token():
                events.append(('secret', None))
                raise ValueError('unavailable')

            def mapped_path(value):
                return {'/kaggle/working': working, '/kaggle/input': inputs}.get(value, Path(value))

            with patch.object(RUNNER, 'Path', side_effect=mapped_path), \
                    patch.object(RUNNER, 'CODE_REVISION', 'synthetic'), \
                    patch.object(RUNNER, 'check_network'), \
                    patch.object(RUNNER, 'create_inference_environment', return_value=root / 'python'), \
                    patch.object(RUNNER.subprocess, 'run', side_effect=fake_run), \
                    patch.object(RUNNER, 'load_wildguard_token', side_effect=missing_token), \
                    patch.dict(RUNNER.os.environ, {'HF_HOME': str(root / 'hf')}):
                with self.assertRaisesRegex(RuntimeError, 'Public assistant generation'):
                    RUNNER.main()
            self.assertEqual(events, [('secret', None),
                                      ('generate', RUNNER.MODELS[0]), ('label', 'refusal_rules'),
                                      ('generate', RUNNER.MODELS[1]), ('label', 'refusal_rules'),
                                      ('secret', None)])
            self.assertTrue((working / 'pilot_outputs.tar.gz').is_file())
            status = json.loads((working / 'pilot_outputs/pipeline_status.json').read_text())
            self.assertEqual(status['status'], 'awaiting_wildguard_token')


if __name__ == '__main__':
    unittest.main()
