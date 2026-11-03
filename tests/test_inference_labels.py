"""Check resume, provenance, failed rows, evaluator parsing and audit scoring without downloads."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from data_contracts import (ROOT, label_id, manifest_identity, validate_bundle, verify_artifacts)
from evaluate_responses import (LABEL_FIELDS, WILDGUARD_REVISION, agreement, evaluate, export_audit,
                                label_row, parse_wildguard, refusal_rules, score_audit)
from generate_responses import Shards, atomic_write, file_hash, read_json, run_generation
from resolve_models import sha256


class FakeTokenizer:
    chat_template = 'synthetic-template'

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        return 'USER: ' + messages[-1]['content'] + ' ASSISTANT:'

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class FakeBackend:
    tokenizer = FakeTokenizer()
    engine = 'fake'
    engine_version = '1'
    hardware = 'synthetic CPU'
    configuration = {'max_model_len': 64}

    def __init__(self, fail=None, wildguard=False):
        self.called = []
        self.fail = fail
        self.wildguard = wildguard

    def generate(self, requests, settings):
        if any(row['prompt_id'] == self.fail for row in requests):
            raise ValueError('Simulated per-prompt failure')
        self.called.extend(row['prompt_id'] for row in requests)
        text = 'Harmful request: no\nResponse refusal: yes\nHarmful response: no' if self.wildguard else \
            'I cannot help with that request.'
        return [{'text': text, 'output_tokens': settings['max_new_tokens'],
                 'finish_reason': 'length', 'output_truncated': True} for row in requests]


class InferenceTest(unittest.TestCase):
    def setUp(self):
        fixture = read_json(ROOT / 'fixtures/contracts/example.bundle.json')
        self.prompts = {key: fixture[key] for key in ['prompts', 'source_manifests', 'prompt_manifests']}
        manifest = self.prompts['prompt_manifests'][0]
        manifest['group_splits'] = {group: 'development' for group in manifest['group_splits']}
        manifest['manifest_id'] = manifest_identity('prompt_manifests', manifest)
        models = fixture['model_manifests'][0]
        models['models'][0]['chat_template_sha256'] = sha256(FakeTokenizer.chat_template)
        models['manifest_id'] = manifest_identity('model_manifests', models)
        self.models = {'model_manifests': [models]}
        self.model_id = models['models'][0]['model_id']
        self.settings = {'seed': 42, 'do_sample': False, 'temperature': 0, 'top_p': 1,
                         'max_input_tokens': 3, 'max_new_tokens': 4, 'repetitions': 1}

    def generate(self, root, backend=None, after_commit=None):
        return run_generation(self.prompts, self.models, self.model_id, backend or FakeBackend(),
                              self.settings, root, batch_size=1, after_commit=after_commit)

    def test_resume_after_crash_ignores_partial_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            def crash(count):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.generate(tmp, after_commit=crash)
            folder = next(Path(tmp).glob('*/*'))
            committed = next(folder.glob('responses-*.jsonl')).read_bytes()
            (folder / 'responses-000001.jsonl.tmp').write_text('{unfinished')
            backend = FakeBackend()
            resumed, stats = self.generate(tmp, backend)
            self.assertEqual(resumed, folder)
            self.assertEqual(len(backend.called), 1)
            self.assertEqual(stats['responses'], 2)
            self.assertEqual(next(folder.glob('responses-*.jsonl')).read_bytes(), committed)
            validate_bundle(read_json(folder / 'validation.bundle.json'))

    def test_failure_is_a_row_and_truncation_and_ids_are_stable(self):
        pid = self.prompts['prompts'][0]['prompt_id']
        with tempfile.TemporaryDirectory() as tmp:
            folder, stats = self.generate(tmp, FakeBackend(fail=pid))
            rows = Shards(folder, 'responses', 'response_id').rows
            self.assertEqual(stats['failures'], 1)
            self.assertTrue(all(row['input_truncated'] for row in rows.values()))
            self.assertEqual(sum(row['output_truncated'] for row in rows.values()), 1)
            first_ids = set(rows)
            backend = FakeBackend()
            again, stats = self.generate(tmp, backend)
            self.assertEqual(again, folder)
            self.assertEqual(backend.called, [])
            self.assertEqual(first_ids, set(Shards(folder, 'responses', 'response_id').rows))
            self.assertEqual(stats['failures'], 1)

    def test_subset_still_references_full_prompt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = run_generation(self.prompts, self.models, self.model_id, FakeBackend(),
                                        self.settings, tmp, limit=1)
            bundle = read_json(folder / 'validation.bundle.json')
            self.assertEqual(len(bundle['prompts']), 2)
            self.assertEqual(len(bundle['responses']), 1)
            validate_bundle(bundle)

    def test_tampered_committed_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.generate(tmp)
            path = next(folder.glob('responses-*.jsonl'))
            path.write_text(path.read_text().replace('cannot', 'can'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.generate(tmp)

    def test_rules_labels_have_null_harm_and_reject_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.generate(tmp)
            output, stats = evaluate(folder, 'refusal_rules')
            self.assertEqual(stats['errors'], 0)
            bundle = read_json(output / 'validation.bundle.json')
            labels = bundle['response_labels']
            self.assertTrue(all(row['refusal'] for row in labels))
            self.assertTrue(all(row['response_harmful'] is None for row in labels))
            labels[0]['model_revision'] = 'b' * 40
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validate_bundle(bundle)

    def test_wildguard_errors_stay_null_and_audit_includes_all_disagreements(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.generate(tmp)
            evaluate(folder, 'refusal_rules')
            backend = FakeBackend(wildguard=True)
            backend.configuration = {'max_model_len': 4096}
            original_generate = backend.generate

            def disagree_once(requests, settings):
                result = original_generate(requests, settings)
                result[0]['text'] = result[0]['text'].replace('Response refusal: yes', 'Response refusal: no')
                return result

            backend.generate = disagree_once
            output, stats = evaluate(folder, 'wildguard', backend)
            bundle = read_json(output / 'validation.bundle.json')
            self.assertEqual(stats['errors'], 0)
            overlap = Path(tmp) / 'overlap.txt'
            overlap.write_text(self.prompts['prompts'][0]['prompt_group_id'] + '\n')
            csv_path = Path(tmp) / 'audit.csv'
            result = export_audit([folder], csv_path, overlap, size=1)
            self.assertGreaterEqual(result['rows'], 1)
            self.assertEqual(result['all_disagreements'], 1)
            with csv_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row['refusal_disagreement'] == 'True' for row in rows))
            self.assertTrue(all(row['human_refusal'] == '' for row in rows))
            self.assertTrue(all(row['wildguardmix_overlap'] == 'True' for row in rows))

    def test_wildguard_malformed_output_has_null_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.generate(tmp)
            backend = FakeBackend()
            backend.configuration = {'max_model_len': 4096}
            output, stats = evaluate(folder, 'wildguard', backend)
            self.assertEqual(stats['errors'], 2)
            for row in read_json(output / 'validation.bundle.json')['response_labels']:
                self.assertIsNotNone(row['error'])
                self.assertTrue(all(row[field] is None for field in LABEL_FIELDS))

    def test_refusal_rules_avoid_topic_only_triggers(self):
        self.assertTrue(refusal_rules("I'm sorry, but I can't assist with that."))
        self.assertTrue(refusal_rules('I cannot provide instructions.'))
        self.assertFalse(refusal_rules('As an AI, I can explain legal history.'))
        self.assertFalse(refusal_rules('Sorry for the confusion. Here is the recipe.'))

    def test_parse_wildguard_rejects_extra_missing_duplicate_and_unknown_answers(self):
        text = 'Harmful request: yes\nResponse refusal: no\nHarmful response: no'
        self.assertEqual(parse_wildguard(text), {'prompt_harmful': True, 'refusal': False,
                                               'response_harmful': False})
        for bad in [text + '\nExplanation', text.replace('no', 'unknown'),
                    text.splitlines()[0], text + '\nResponse refusal: yes']:
            with self.assertRaises(ValueError):
                parse_wildguard(bad)

    def test_agreement_handles_missing_labels_and_single_class(self):
        rows = []
        for index, (human, model) in enumerate([('yes', 'yes'), ('no', 'no'), ('', 'yes')]):
            row = {'response_id': str(index), 'model_id': 'example/model'}
            for field in LABEL_FIELDS:
                row['human_' + field], row['wildguard_' + field] = human, model
            rows.append(row)
        self.assertEqual(agreement(rows)['refusal']['kappa'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'completed.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = score_audit(path)
        self.assertTrue(result['refusal_kappa_at_least_0_8'])
        self.assertEqual(result['overall']['refusal']['n'], 2)
        self.assertIsNone(agreement(rows[:1])['refusal']['kappa'])


class ArtifactPathTest(unittest.TestCase):
    def test_data_junction_allowed_and_traversal_and_nested_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, shared, outside = [base / name for name in ['project', 'shared', 'outside']]
            for directory in [root, shared, outside]:
                directory.mkdir()
            target = root / 'data'
            try:
                target.symlink_to(shared, target_is_directory=True)
            except OSError:
                if os.name != 'nt':
                    raise
                subprocess.run(['cmd', '/c', 'mklink', '/J', str(target), str(shared)],
                               check=True, capture_output=True)
            (shared / 'artifact.txt').write_text('fixture')
            artifact = {'path': 'data/artifact.txt', 'sha256': file_hash(shared / 'artifact.txt')}
            bundle = {'prompt_manifests': [{'artifacts': [artifact]}]}
            verify_artifacts(bundle, root)
            artifact['path'] = '../shared/artifact.txt'
            with self.assertRaisesRegex(ValueError, 'traversal'):
                verify_artifacts(bundle, root)
            artifact['path'] = str(shared / 'artifact.txt')
            with self.assertRaises(ValueError):
                verify_artifacts(bundle, root)
            escape = shared / 'escape'
            try:
                escape.symlink_to(outside, target_is_directory=True)
            except OSError:
                subprocess.run(['cmd', '/c', 'mklink', '/J', str(escape), str(outside)],
                               check=True, capture_output=True)
            artifact['path'] = 'data/escape/artifact.txt'
            with self.assertRaisesRegex(ValueError, 'inside project'):
                verify_artifacts(bundle, root)


if __name__ == '__main__':
    unittest.main()
