"""Exercise the labels-only handoff with real contract/shard fixtures and no GPU."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_inference_labels as fixtures
from data_contracts import manifest_identity
from generate_responses import Shards, run_generation
from evaluate_responses import evaluate

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/kaggle' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


WORKER = module('wildguard_only', 'wildguard_runner.py')
PREPARE = module('prepare_wildguard', 'prepare_wildguard.py')


class WildGuardHandoffTest(unittest.TestCase):
    def build(self, root):
        case = fixtures.InferenceTest()
        case.setUp()
        models = deepcopy(case.models)
        panel = models['model_manifests'][0]
        template = deepcopy(panel['models'][0])
        panel['models'] = [{**template, 'model_id': name} for name in WORKER.MODELS]
        panel['manifest_id'] = manifest_identity('model_manifests', panel)
        source, project = root / 'source', root / 'project'
        (source / 'pilot_outputs').mkdir(parents=True)
        (source / 'pilot_outputs/code_revision.txt').write_text(WORKER.UPSTREAM_REVISION)
        runs = []
        for name in WORKER.MODELS:
            run, _ = run_generation(case.prompts, models, name, fixtures.FakeBackend(),
                                    case.settings, source / 'responses',
                                    splits=('development', 'validation'), batch_size=2)
            evaluate(run, 'refusal_rules')
            runs.append(run)
        return case.prompts, panel, source, project, runs

    def test_both_runs_copy_unchanged_and_only_wildguard_is_invoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts, panel, source, project, originals = self.build(root)
            runs, provenance = WORKER.copy_runs(source, project, prompts, expected_model_manifest=panel)
            copied_by_name = {p.name: p for p in runs}
            for original in originals:
                copied = copied_by_name[original.name]
                self.assertEqual((original / 'stats.json').read_bytes(), (copied / 'stats.json').read_bytes())
                self.assertEqual(set(Shards(original, 'responses', 'response_id').rows),
                                 set(Shards(copied, 'responses', 'response_id').rows))
            self.assertEqual({r['model_id'] for r in provenance}, set(WORKER.MODELS))
            commands = []

            def command(args, name):
                commands.append(args)
                if '--labeler' in args:
                    run = Path(args[args.index('--run') + 1])
                    folder = run / 'labels/wildguard-fixture'
                    folder.mkdir(parents=True)
                    (folder / 'validation.bundle.json').write_text('{}')

            WORKER.label_runs(runs, root, command)
            self.assertEqual(sum('--labeler' in c for c in commands), 2)
            self.assertTrue(all(c[c.index('--labeler') + 1] == 'wildguard'
                                for c in commands if '--labeler' in c))
            self.assertFalse(any('python/generate_responses.py' in c for c in commands))
            self.assertEqual(commands[-1][1], 'audit')

    def test_incomplete_wrong_revision_and_corrupt_shards_are_rejected(self):
        for problem in ['incomplete', 'revision', 'checksum', 'selection']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                prompts, panel, source, project, runs = self.build(Path(tmp))
                if problem == 'revision':
                    (source / 'pilot_outputs/code_revision.txt').write_text('0' * 40)
                elif problem == 'checksum':
                    shard = next(runs[0].glob('responses-*.jsonl'))
                    shard.write_bytes(shard.read_bytes() + b'\n')
                elif problem == 'selection':
                    path = runs[0] / 'run.bundle.json'
                    data = json.loads(path.read_text())
                    data['run_manifests'][0]['runtime']['configuration']['selection']['splits'] = ['test']
                    path.write_text(json.dumps(data))
                else:
                    path = runs[0] / 'stats.json'
                    data = json.loads(path.read_text());data['completed'] = False
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    WORKER.copy_runs(source, project, prompts, expected_model_manifest=panel)

    def test_archive_layout_and_traversal_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts, panel, source, project, _ = self.build(root)
            inputs = root / 'input';inputs.mkdir()
            archive = inputs / 'pilot_outputs.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                stream.add(source / 'pilot_outputs', arcname='pilot_outputs')
                stream.add(source / 'responses', arcname='responses')
            restored, evidence = WORKER.source_tree(inputs, root / 'working')
            resumed, same_evidence = WORKER.source_tree(inputs, root / 'working')
            self.assertEqual(restored, resumed)
            self.assertEqual(evidence, same_evidence)
            runs, _ = WORKER.copy_runs(restored, project, prompts, expected_model_manifest=panel)
            self.assertEqual(len(runs), 2)
            self.assertEqual(evidence['archive_sha256'], WORKER.file_hash(archive))
            with tarfile.open(archive, 'w:gz') as stream:
                member = tarfile.TarInfo('../outside');member.size = 1
                stream.addfile(member, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                WORKER.extract_source(archive, root / 'rejected')
            self.assertFalse((root / 'rejected').exists())

    def test_staged_notebook_is_private_has_kernel_input_and_does_not_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(PREPARE.subprocess, 'run'):
            output = PREPARE.prepare(Path(tmp) / 'stage', 'a' * 40, 'deepamahuja')
            meta = json.loads((output / 'kernel-metadata.json').read_text())
            notebook = json.loads((output / 'wildguard.ipynb').read_text(encoding='utf-8'))
            self.assertTrue(meta['is_private'])
            self.assertEqual(meta['kernel_sources'], ['deepamahuja/llm-risk-topology-two-model-pilot'])
            self.assertTrue(meta['dataset_sources'])
            source = ''.join(notebook['cells'][-1]['source'])
            self.assertIn("CODE_REVISION = '" + 'a' * 40 + "'", source)
            compile(source, 'wildguard.ipynb', 'exec')
            self.assertEqual(notebook['cells'][-1]['outputs'], [])

    def test_staged_frontend_hands_pinned_code_to_managed_python_without_file_global(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(PREPARE.subprocess, 'run'):
            root = Path(tmp)
            output = PREPARE.prepare(root / 'stage', 'a' * 40, 'deepamahuja')
            notebook = json.loads((output / 'wildguard.ipynb').read_text(encoding='utf-8'))
            staged = ModuleType('notebook_cell')
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), staged.__dict__)
            working, inputs = root / 'working', root / 'input'
            project = working / 'project'
            scripts = project / 'scripts/kaggle';scripts.mkdir(parents=True)
            (scripts / 'wildguard_runner.py').write_bytes((ROOT / 'scripts/kaggle/wildguard_runner.py').read_bytes())
            dataset = inputs / 'dataset';dataset.mkdir(parents=True)
            (dataset / 'pilot_v5').mkdir()
            (dataset / 'payload.json').write_text('{}')
            helpers = SimpleNamespace(check_network=lambda: None, load_wildguard_token=lambda: None,
                                      install_payload=lambda *args: None,
                                      create_inference_environment=lambda path: working / 'inference-env/bin/python')
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            mapped = lambda value: {'/kaggle/working': working, '/kaggle/input': inputs}.get(value, Path(value))
            with patch.object(staged, 'Path', side_effect=mapped), \
                    patch.object(staged.importlib.util, 'spec_from_file_location', return_value=spec), \
                    patch.object(staged.importlib.util, 'module_from_spec', return_value=helpers), \
                    patch.object(staged, 'source_tree'), \
                    patch.object(staged.sys, 'argv', ['notebook']), \
                    patch.dict(staged.os.environ), \
                    patch.object(staged.subprocess, 'run') as run:
                staged.main()
            worker = ModuleType('managed_worker')
            exec((working / 'wildguard_worker.py').read_text(), worker.__dict__)
            self.assertEqual(worker.CODE_REVISION, 'a' * 40)
            self.assertEqual(run.call_args.args[0][-1], '--label-worker')


if __name__ == '__main__':
    unittest.main()
