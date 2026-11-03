import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from prepare_prompts import prepare, json_records
from data_contracts import canonical, validate_bundle


def source(name, rows, config='train'):
    return {'source_id': name, 'revision': 'a'*40, 'config': config, 'split': 'train',
            'rows': [{'row_index': i, 'row': row} for i,row in enumerate(rows)]}


class PromptPreparationTest(unittest.TestCase):
    def test_cross_source_duplicate_promotes_entire_pair_to_reserved_test(self):
        inputs = [source('OpenSafetyLab/Salad-Data', [{'qid': 1, 'baseq': 'Shared original', 'augq': 'Existing variant'}], 'attack_enhanced_set'),
                  source('paul-rottger/xstest', [{'id': 'x1', 'prompt': 'Shared original', 'label': 'safe'}])]
        prompts, pairs, aliases, splits, quality, unchanged = prepare(inputs)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(aliases), 3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(splits.values()), {'test'})
        self.assertEqual(len({p['prompt_group_id'] for p in prompts}), 1)
        self.assertEqual(quality['cross_source_duplicate_inputs'], 1)

    def test_dedup_originals_retains_multiple_actual_variants(self):
        inputs = [source('OpenSafetyLab/Salad-Data', [{'qid': 1, 'baseq': 'Original', 'augq': 'Variant one'},
                  {'qid': 1, 'baseq': 'Original', 'augq': 'Variant two'}], 'attack_enhanced_set')]
        prompts, pairs, aliases, splits, quality, unchanged = prepare(inputs)
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({p['prompt_group_id'] for p in prompts}), 1)
        self.assertEqual(quality['duplicate_occurrences_removed'], 1)
        validate_bundle({'prompts': prompts})

    def test_unchanged_variant_is_not_a_behavior_change_pair(self):
        inputs = [source('OpenSafetyLab/Salad-Data', [{'baseq': 'Same text', 'augq': 'Same text'}], 'attack_enhanced_set')]
        prompts, pairs, aliases, splits, quality, unchanged = prepare(inputs)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(pairs, [])
        self.assertEqual(len(unchanged), 1)
        self.assertIsNone(prompts[0]['parent_prompt_id'])

    def test_cap_never_slices_a_parent_group(self):
        inputs = [source('OpenSafetyLab/Salad-Data', [{'baseq': 'Original', 'augq': 'Variant'}], 'attack_enhanced_set')]
        prompts, pairs, aliases, splits, quality, unchanged = prepare(inputs, maximum=1)
        self.assertEqual(prompts, [])
        self.assertEqual(pairs, [])

    def test_reordering_inputs_is_deterministic(self):
        a = source('allenai/wildjailbreak', [{'vanilla': 'An original', 'adversarial': 'A variant', 'data_type': 'adversarial_harmful', 'completion': 'HISTORICAL ANSWER'}])
        b = source('paul-rottger/xstest', [{'id': 'x1', 'prompt': 'Another prompt', 'label': 'safe'}])
        one, two = prepare([a,b]), prepare([b,a])
        for index in range(4): self.assertEqual(one[index], two[index])
        self.assertNotIn('HISTORICAL ANSWER', canonical(one[:4]))

    def test_json_array_and_jsonl_multiline_records(self):
        rows = [{'baseq': 'Text\nwith newline', 'augq': 'Other'}, {'baseq': 'Unicode \u00e9', 'augq': 'End'}]
        for text in [json.dumps(rows, indent=2), '\n'.join(json.dumps(r) for r in rows)]:
            self.assertEqual(list(json_records(io.StringIO(text))), rows)

    def test_missing_fields_are_reported(self):
        result = prepare([source('OpenSafetyLab/Salad-Data', [{'baseq': 'No variant field'}], 'attack_enhanced_set')])
        self.assertEqual(len(result[4]['rejected_rows']), 1)
        self.assertEqual(result[0], [])

    def test_declared_parent_id_keeps_different_base_wordings_together(self):
        inputs = [source('OpenSafetyLab/Salad-Data', [{'qid': 7, 'baseq': 'Original wording', 'augq': 'Variant one'},
                  {'qid': 7, 'baseq': 'Reworded original', 'augq': 'Variant two'}], 'attack_enhanced_set')]
        prompts, pairs, aliases, splits, quality, unchanged = prepare(inputs)
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(splits), 1)
        self.assertEqual(quality['salad_qids_with_multiple_base_texts'], 1)


if __name__ == '__main__':
    unittest.main()
