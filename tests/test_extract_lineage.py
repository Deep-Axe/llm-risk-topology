import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from extract_lineage import extract


class LineageTest(unittest.TestCase):
    def test_multiple_parents_and_unknown_relation(self):
        result = extract('test/child', {'base_model': ['test/a', 'test/b']})
        self.assertEqual([x[2] for x in result], ['test/a', 'test/b'])
        self.assertTrue(all(x[3] == 'other' for x in result))
        self.assertEqual(json.loads(result[0][5]), ['test/a', 'test/b'])

    def test_quantization_is_not_fine_tune(self):
        row = extract('test/child', {'base_model': 'test/base', 'base_model_relation': 'quantized'})[0]
        self.assertEqual(row[3], 'quantization')

    def test_reject_malformed_and_self_parents(self):
        rows = extract('test/child', {'base_model': ['test/child', {'name': 'test/base'}, 'https://huggingface.co/test/base']})
        self.assertTrue(all(x[0] == 'rejected' for x in rows))

    def test_missing_parent_is_not_inferred(self):
        self.assertEqual(extract('test/child', {})[0][0], 'no_lineage')


if __name__ == '__main__':
    unittest.main()
