"""Check paired-label exclusions and refusal agreement in downloaded pilot summaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import test_inference_labels as inference_fixtures

PATH = Path(__file__).resolve().parents[1] / 'scripts/kaggle/summarize_pilot.py'
SPEC = importlib.util.spec_from_file_location('pilot_summary', PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
from evaluate_responses import evaluate
from generate_responses import Shards


class PilotSummaryTest(unittest.TestCase):
    def build(self, root, malformed=False):
        fixture = inference_fixtures.InferenceTest()
        fixture.setUp()
        folder, _ = fixture.generate(root)
        responses = Shards(folder, 'responses', 'response_id').rows
        variant = next(p for p in fixture.prompts['prompts'] if p['parent_prompt_id'])
        original = variant['parent_prompt_id']
        variant_response = next(r['response_id'] for r in responses.values()
                                if r['prompt_id'] == variant['prompt_id'])
        evaluate(folder, 'refusal_rules')
        backend = inference_fixtures.FakeBackend(wildguard=True)
        backend.configuration = {'max_model_len': 4096}

        def output(requests, settings):
            result = []
            for request in requests:
                text = 'Harmful request: no\nResponse refusal: yes\nHarmful response: no'
                if request['prompt_id'] == variant_response:
                    text = 'unknown' if malformed else text.replace('Response refusal: yes',
                                                                    'Response refusal: no')
                result.append({'text': text, 'output_tokens': 12,
                               'finish_reason': 'stop', 'output_truncated': False})
            return result

        backend.generate = output
        evaluate(folder, 'wildguard', backend)
        pairs = [{'original_prompt_id': original, 'variant_prompt_id': variant['prompt_id']}]
        return SUMMARY.analyze_run(folder, pairs)

    def test_flip_direction_and_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.build(tmp)
        self.assertEqual(result['development_refusal_flips']['valid_pairs'], 1)
        self.assertEqual(result['development_refusal_flips']['refused_to_complied'], 1)
        self.assertEqual(result['development_refusal_flips']['flip_prevalence'], 1)
        self.assertEqual(result['refusal_agreement_rate'], .5)
        self.assertEqual(result['wildguard_parse_errors'], 0)

    def test_unparseable_label_excludes_the_pair_without_becoming_a_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.build(tmp, malformed=True)
        self.assertEqual(result['wildguard_parse_errors'], 1)
        self.assertEqual(result['wildguard_parse_error_rate'], .5)
        self.assertEqual(result['development_refusal_flips']['excluded_pairs'], 1)
        self.assertIsNone(result['development_refusal_flips']['flip_prevalence'])


if __name__ == '__main__':
    unittest.main()
