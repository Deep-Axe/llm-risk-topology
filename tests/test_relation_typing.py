"""Test evidence conflicts, parent-specific tags and preservation of declared relations."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from relation_typing import checksum, infer, type_snapshot, write_json
from data_contracts import validate_bundle


class RelationTypingTest(unittest.TestCase):
    def test_parent_tag_must_match_the_actual_edge(self):
        metadata = {'tags': ['base_model:quantized:other/parent', 'base_model:finetune:base/model']}
        result = infer('child/model', 'base/model', {}, metadata)
        self.assertEqual(result[:3], ('fine_tune', 'api_parent_finetune', 'medium'))

    def test_quantized_artifact_overrides_the_hub_default_finetune_inference(self):
        metadata = {'tags': ['base_model:finetune:base/model'],
                    'siblings': [{'rfilename': 'model-Q4_K_M.gguf'}]}
        result = infer('child/model', 'base/model', {}, metadata)
        self.assertEqual(result[:3], ('quantization', 'quantized_artifact', 'high'))

    def test_multiple_transformations_are_left_unresolved(self):
        metadata = {'siblings': [{'rfilename': 'model.gguf'}, {'rfilename': 'adapter_config.json'}]}
        self.assertEqual(infer('child/model', 'base/model', {}, metadata)[:3],
                         (None, 'conflicting_transformations', 'unresolved'))

    def test_repository_pattern_does_not_guess_from_instruct_or_owner(self):
        self.assertEqual(infer('quantized-owner/Qwen-Instruct', 'base/model', {}, {})[0], None)
        self.assertEqual(infer('owner/model-AWQ', 'base/model', {}, {})[:3],
                         ('quantization', 'repository_name', 'low'))

    def test_adapter_library_and_merge_tags(self):
        self.assertEqual(infer('owner/model', 'base/model', {'library_name': 'peft'}, {})[:3],
                         ('adapter', 'peft_library', 'medium'))
        self.assertEqual(infer('owner/model', 'base/model', {'tags': ['mergekit']}, {})[0], 'merge')

    def test_snapshot_keeps_declared_types_and_emits_separate_inferred_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, output = Path(tmp) / 'snapshot', Path(tmp) / 'output'
            snapshot.mkdir()
            rows = [
                {'id': 'child/declared', 'sha': 'a' * 40,
                 'cardData': {'base_model': 'base/model', 'base_model_relation': 'finetune'},
                 'tags': ['base_model:quantized:base/model']},
                {'id': 'child/inferred', 'sha': 'b' * 40, 'cardData': {'base_model': 'base/model'},
                 'tags': ['base_model:quantized:base/model']},
            ]
            with gzip.open(snapshot / 'models.jsonl.gz', 'wt', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            with (snapshot / 'models.tsv').open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write('\t'.join([row['id'], row['sha'], json.dumps(row['cardData'])]) + '\n')
            write_json(snapshot / 'metadata_snapshot.json', {
                'status': 'frozen', 'artifacts': [{'path': name, 'sha256': checksum(snapshot / name)}
                                                  for name in ['models.tsv', 'models.jsonl.gz']]})
            source_hashes = {path.name: checksum(path) for path in snapshot.iterdir()}
            summary = type_snapshot(snapshot, output, audit_size=1)
            self.assertEqual(summary['declared_edges'], 2)
            self.assertEqual(summary['inferred_edges'], 1)
            annotations = [json.loads(line) for line in (output / 'annotations.jsonl').read_text().splitlines()]
            declared = next(row for row in annotations if row['child_model_id'] == 'child/declared')
            self.assertEqual(declared['declared_relation'], 'fine_tune')
            self.assertIsNone(declared['inferred_relation'])
            inferred = [json.loads(line) for line in (output / 'inferred_edges.jsonl').read_text().splitlines()]
            validate_bundle({'metadata_edges': inferred})
            self.assertEqual(inferred[0]['lineage_status'], 'inferred')
            self.assertEqual(source_hashes, {path.name: checksum(path) for path in snapshot.iterdir()})


if __name__ == '__main__':
    unittest.main()
