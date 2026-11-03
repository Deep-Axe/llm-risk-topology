import io
import json
from pathlib import Path
import sys
import csv
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from prepare_prompts import (prepare, json_records, sample_wild_file, extend_prepared,
                             near_matches, load_prepared, word_grams, containment, estimate_tokens)
from data_contracts import canonical, validate_bundle, digest


def source(name, rows, config='train'):
    return {'source_id': name, 'revision': 'a'*40, 'config': config, 'split': 'train',
            'rows': [{'row_index': i, 'row': row} for i,row in enumerate(rows)]}


class PromptPreparationTest(unittest.TestCase):
    def test_token_estimate_uses_the_frozen_tokenizer_revision(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.apply_chat_template.return_value = [1, 2]
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer) as load:
            estimate_tokens([{'prompt_id': 'synthetic', 'messages': [{'role': 'user', 'content': 'Hi'}]}])
        load.assert_called_once_with('mistralai/Mistral-7B-Instruct-v0.1',
                                     revision='ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca')

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

    def test_adversarial_intent_rows_keep_parents_and_exclude_completions(self):
        rows = [{'vanilla': 'A direct request', 'adversarial': 'A rewritten request',
                 'data_type': kind, 'completion': 'STORED RESPONSE'}
                for kind in ['adversarial_harmful', 'adversarial_benign']]
        result = prepare([source('allenai/wildjailbreak', rows)])
        self.assertEqual(len(result[1]), 2)
        intents = {alias['source_labels']['prompt_intent'] for alias in result[2]}
        self.assertEqual(intents, {'harmful', 'unharmful'})
        self.assertNotIn('STORED RESPONSE', canonical(result))

    def test_full_file_sampling_matches_hash_rank_and_repeats(self):
        rows = [{'vanilla': f'Original {i}', 'adversarial': f'Variant {i}',
                 'data_type': kind, 'completion': 'STORED RESPONSE'}
                for kind in ['vanilla_harmful', 'adversarial_harmful', 'adversarial_benign']
                for i in range(9)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.tsv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows)
            spec = source('allenai/wildjailbreak', [])
            first, audit = sample_wild_file(path, spec, per_intent=3, seed=42)
            second, repeated = sample_wild_file(path, spec, per_intent=3, seed=42)
        expected = []
        for kind in ['adversarial_harmful', 'adversarial_benign']:
            eligible = [i for i, row in enumerate(rows) if row['data_type'] == kind]
            expected.extend(sorted(eligible, key=lambda i: digest([42, 'a' * 40, 'wild-row', i]))[:3])
        self.assertEqual([r['row_index'] for r in first['rows']], sorted(expected))
        self.assertEqual((first, audit), (second, repeated))
        self.assertEqual(audit['file_row_counts']['adversarial_benign'], 9)
        self.assertNotIn('STORED RESPONSE', canonical(first))

    def test_near_duplicate_joins_frozen_group_without_changing_it(self):
        text = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
        base = prepare([source('allenai/wildjailbreak', [
            {'vanilla': text, 'adversarial': '', 'data_type': 'vanilla_benign'}])])
        new = source('allenai/wildjailbreak', [
            {'vanilla': text + ' nu', 'adversarial': 'A separate rewritten version',
             'data_type': 'adversarial_benign'}])
        new['rows'][0]['row_index'] = 20
        result = extend_prepared(base, [new])
        lookup = {p['prompt_id']: p for p in result[0]}
        self.assertEqual(lookup[base[0][0]['prompt_id']], base[0][0])
        self.assertEqual(result[3], base[3])
        self.assertEqual(len({p['prompt_group_id'] for p in result[0]}), 1)
        self.assertEqual(len(result[4]['near_duplicate_audit']['merges']), 1)
        validate_bundle({'prompts': result[0]})

    def test_new_near_groups_share_anchor_and_split(self):
        text = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
        new = source('allenai/wildjailbreak', [
            {'vanilla': text, 'adversarial': 'First rewritten version', 'data_type': 'adversarial_benign'},
            {'vanilla': text + ' nu', 'adversarial': 'Second rewritten version',
             'data_type': 'adversarial_benign'}])
        result = extend_prepared(prepare([]), [new])
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(len(result[3]), 1)
        self.assertEqual(result[4]['near_duplicate_audit']['final_cross_split_question_matches'], 0)

    def test_new_bridge_between_frozen_groups_is_excluded(self):
        left = 'alpha beta gamma delta epsilon zeta eta theta iota'
        right = 'one two three four five six seven eight nine ten'
        base = prepare([source('allenai/wildjailbreak', [
            {'vanilla': text, 'adversarial': '', 'data_type': 'vanilla_benign'}
            for text in [left, right]])])
        new = source('allenai/wildjailbreak', [
            {'vanilla': left + ' ' + right, 'adversarial': 'A rewritten bridge',
             'data_type': 'adversarial_benign'}])
        new['rows'][0]['row_index'] = 20
        result = extend_prepared(base, [new])
        self.assertEqual(result[0], base[0])
        self.assertEqual(result[3], base[3])
        self.assertEqual(len(result[4]['near_duplicate_audit']['excluded_bridge_components']), 1)

    def test_shared_variant_template_does_not_merge_different_questions(self):
        wrapper = ' '.join(f'wrapper{i}' for i in range(65))
        inputs = [source('allenai/wildjailbreak', [
            {'vanilla': core, 'adversarial': wrapper + ' ' + core, 'data_type': 'adversarial_benign'}
            for core in ['alpha beta gamma delta epsilon zeta eta theta iota',
                         'one two three four five six seven eight nine']])]
        result = extend_prepared(prepare([]), inputs)
        hits = near_matches(result[0], result[1])
        self.assertTrue(any(hit['template_only'] for hit in hits))
        self.assertEqual(len(result[3]), 2)

    def test_containment_is_symmetric_shorter_text_fraction(self):
        short = word_grams('one two three four five six seven eight nine')
        long = word_grams('prefix one two three four five six seven eight nine suffix')
        self.assertEqual(containment(short, long), 1)
        self.assertEqual(containment(long, short), 1)

    def test_frozen_v4_saved_input_replays_byte_for_byte(self):
        folder = Path(__file__).resolve().parents[1] / 'data/processed/prompts/pilot_v4'
        if not folder.exists():
            self.skipTest('Frozen local pilot_v4 data is required for the replay check')
        inputs = json.loads((folder / 'inputs.json').read_text(encoding='utf-8'))
        records = prepare(inputs, seed=42, maximum=2500)[0]
        replay = ''.join(canonical(row) + '\n' for row in records).encode('utf-8')
        self.assertEqual(replay, (folder / 'prompts.jsonl').read_bytes())

    def test_real_v5_preserves_v4_and_meets_test_group_targets(self):
        folder = Path(__file__).resolve().parents[1] / 'data/processed/prompts'
        if not (folder / 'pilot_v5/experiment.bundle.json').exists():
            self.skipTest('Frozen local pilot_v5 data is required for the extension check')
        old = load_prepared(folder / 'pilot_v4')
        new = load_prepared(folder / 'pilot_v5')
        lookup = {p['prompt_id']: p for p in new[0]}
        for row in old[0]:
            self.assertEqual(lookup[row['prompt_id']], row)
        for group, split in old[3].items():
            self.assertEqual(new[3][group], split)
        test_groups = {lookup[p['original_prompt_id']]['prompt_group_id'] for p in new[1]
                       if new[3][lookup[p['original_prompt_id']]['prompt_group_id']] == 'test'}
        benign_ids = {a['canonical_prompt_id'] for a in new[2]
                      if a['source_labels'].get('prompt_intent') == 'unharmful'}
        benign_test = {lookup[p['original_prompt_id']]['prompt_group_id'] for p in new[1]
                       if p['original_prompt_id'] in benign_ids and
                       new[3][lookup[p['original_prompt_id']]['prompt_group_id']] == 'test'}
        self.assertLessEqual(len(new[0]), 6000)
        self.assertGreaterEqual(len(test_groups), 300)
        self.assertGreaterEqual(len(benign_test), 100)
        validate_bundle({'prompts': new[0]})


if __name__ == '__main__':
    unittest.main()
