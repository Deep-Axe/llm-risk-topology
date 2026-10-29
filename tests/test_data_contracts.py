"""Contract tests focus on provenance, leakage and reproducibility failures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from data_contracts import ROOT, assign_groups, normalize_messages, prompt_hash, manifest_identity, validate_bundle


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads((ROOT/'fixtures/contracts/example.bundle.json').read_text(encoding='utf-8'))

    def test_valid_fixture_and_draft_templates(self):
        validate_bundle(self.bundle)
        validate_bundle(json.loads((ROOT/'manifests/templates/experiment.bundle.json').read_text(encoding='utf-8')))

    def test_normalization_preserves_meaningful_whitespace_and_case(self):
        a = [{'role': 'user', 'content': 'Cafe\u0301\r\nHi '}]
        b = [{'role': 'user', 'content': 'Caf\u00e9\nHi '}]
        self.assertEqual(prompt_hash(a), prompt_hash(b))
        self.assertNotEqual(prompt_hash(b), prompt_hash([{'role': 'user', 'content': 'Caf\u00e9\nHi'}]))

    def test_grouping_transitive_and_order_independent(self):
        anchors = {'p1': ['a', 'b'], 'p2': ['c'], 'p3': ['b', 'c']}
        forward = assign_groups(anchors)
        self.assertEqual(forward, assign_groups(dict(reversed(list(anchors.items())))))
        self.assertEqual(len(set(forward.values())), 1)

    def test_reject_tampered_prompt_and_response_revision(self):
        self.bundle['prompts'][0]['messages'][0]['content'] += ' altered'
        with self.assertRaises(ValueError): validate_bundle(self.bundle)
        self.setUp()
        self.bundle['responses'][0]['model_revision'] = 'b'*40
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_reject_parent_cycle_and_wrong_group(self):
        self.bundle['prompts'][0]['parent_prompt_id'] = self.bundle['prompts'][1]['prompt_id']
        with self.assertRaises(ValueError): validate_bundle(self.bundle)
        self.setUp()
        self.bundle['prompts'][1]['prompt_group_id'] = 'g_'+'0'*64
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_reject_frozen_unresolved_models(self):
        self.bundle['model_manifests'][0]['models'][0]['revision'] = None
        m = self.bundle['model_manifests'][0]
        m['manifest_id'] = manifest_identity('model_manifests', m)
        with self.assertRaises(ValueError): validate_bundle(self.bundle)

    def test_reject_missing_split_and_duplicate_response(self):
        m = self.bundle['prompt_manifests'][0]
        m['group_splits'] = {}
        m['manifest_id'] = manifest_identity('prompt_manifests', m)
        with self.assertRaises(ValueError): validate_bundle(self.bundle)
        self.setUp()
        self.bundle['responses'].append(deepcopy(self.bundle['responses'][0]))
        with self.assertRaises(ValueError): validate_bundle(self.bundle)


if __name__ == '__main__':
    unittest.main()
