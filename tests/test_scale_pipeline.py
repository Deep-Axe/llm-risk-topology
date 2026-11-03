"""Check registered settings, incremental archives, and descriptive validation reports."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import test_inference_labels as fixtures
from data_contracts import manifest_identity
from generate_responses import run_generation
from evaluate_responses import evaluate

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/kaggle' / file)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


SCALE = load('scale_worker', 'wildguard_runner.py')
REPORT = load('model_report', 'report_model.py')


class ScaleTest(unittest.TestCase):
    def test_only_registered_models_and_validation_are_allowed(self):
        self.assertEqual(SCALE.run_config()['models'], list(SCALE.EXTRA_MODELS))
        for bad in [{'splits': ['test']}, {'splits': ['development']},
                    {'models': [SCALE.MODELS[0]]}, {'models': [SCALE.EXTRA_MODELS[0]] * 2}]:
            with self.assertRaises(ValueError):
                SCALE.run_config(bad)

    def test_checkpoint_and_cache_release_after_each_extra_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);project = root / 'project';outputs = root / 'pilot_outputs';outputs.mkdir()
            models = [SCALE.EXTRA_MODELS[0], SCALE.EXTRA_MODELS[2]]
            manifest = project / 'manifests/models/scale6.bundle.json';manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'model_manifests': [{'status': 'frozen', 'manifest_id': 'panel',
                                                                'models': [{'model_id': m} for m in models]}]}))
            events = [];commands = []

            def command(args, name):
                commands.append(args)
                if 'python/generate_responses.py' in args:
                    model = args[args.index('--model') + 1];events.append(('generate', model))
                    folder = project / 'data/processed/responses/run' / model.replace('/', '__')
                    folder.mkdir(parents=True)
                    (folder / 'run.bundle.json').write_text(json.dumps({'run_manifests': [{
                        'model_manifest_id': 'panel', 'runtime': {'configuration': {'model_id': model}}}]}))
                    (folder / 'stats.json').write_text('{"completed":true}')
                    cache = root / 'hf/hub' / ('models--' + model.replace('/', '--'));cache.mkdir(parents=True)
                    (cache / 'weights').write_text('fixture')
                else:
                    events.append(('rules', name))

            def checkpoint(state):
                events.append(('checkpoint', state['completed_model']))
                SCALE.save_checkpoint(root, project, outputs, state)

            with patch.dict(SCALE.os.environ, {'HF_HOME': str(root / 'hf')}):
                runs = SCALE.generate_extra_runs(project, outputs, SCALE.run_config({'models': models}),
                                                command, checkpoint, lambda *args: None)
            self.assertEqual(len(runs), 2)
            self.assertEqual([x[0] for x in events], ['generate', 'rules', 'checkpoint'] * 2)
            self.assertFalse(list((root / 'hf/hub').glob('models--*')))
            with tarfile.open(root / 'pilot_outputs.tar.gz') as archive:
                self.assertEqual(sum(n.endswith('/run.bundle.json') for n in archive.getnames()), 2)
            for args in commands:
                if '--backend' in args:
                    for flag, expected in [('--max-new-tokens', '512'), ('--seed', '42'),
                                           ('--batch-size', '64'), ('--max-input-tokens', '4096')]:
                        self.assertEqual(args[args.index(flag) + 1], expected)
                    self.assertEqual(args[args.index('--split') + 1:], ['validation'])

    def test_validation_report_uses_rule_labels_without_inventing_harm(self):
        case = fixtures.InferenceTest();case.setUp()
        manifest = case.prompts['prompt_manifests'][0]
        manifest['group_splits'] = {group: 'validation' for group in manifest['group_splits']}
        manifest['manifest_id'] = manifest_identity('prompt_manifests', manifest)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, _ = run_generation(case.prompts, case.models, case.model_id, fixtures.FakeBackend(),
                                       case.settings, root / 'responses', splits=('validation',), batch_size=2)
            evaluate(folder, 'refusal_rules')
            pairs = root / 'data/processed/prompts/pilot_v5/pairs.jsonl';pairs.parent.mkdir(parents=True)
            ids = [p['prompt_id'] for p in case.prompts['prompts']]
            pairs.write_text(json.dumps({'original_prompt_id': ids[0], 'variant_prompt_id': ids[1]}) + '\n')
            with patch.object(REPORT, 'ROOT', root):
                result = REPORT.report_model(folder, root / 'report')
            self.assertEqual(result['responses'], 2)
            self.assertEqual(result['failures'], 0)
            self.assertEqual(result['validation_refusal_flips'][0]['valid_pairs'], 1)
            self.assertEqual(result['validation_refusal_flips'][0]['flip_rate'], 0)
            self.assertTrue(all(row['harmful_response_rate'] is None for row in result['rates']))
            self.assertTrue(list((root / 'report').glob('*_rates.csv')))


if __name__ == '__main__':
    unittest.main()
