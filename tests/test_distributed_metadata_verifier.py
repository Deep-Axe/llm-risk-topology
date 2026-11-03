import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from data_contracts import ROOT
from verify_metadata_pipeline import verify_distributed, main

class DistributedVerifierTest(unittest.TestCase):
    def test_equal_edge_counts_do_not_hide_missing_and_unexpected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            sample=json.loads((ROOT/'fixtures/contracts/example.bundle.json').read_text())['metadata_edges'][0]
            (p/'edges_sample.jsonl').write_text(json.dumps(sample)+'\n')
            (p/'summary.json').write_text(json.dumps({'typed_edges':10}))
            (p/'distributed_validation.jsonl').write_text(json.dumps({'passed':True,'missing_edges':1,'unexpected_edges':1,'invalid_contracts':0})+'\n')
            with self.assertRaisesRegex(ValueError,'missing_edges'):
                verify_distributed(p,p,{'record_count':50},'a'*64)
    def test_failed_spark_report_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'summary.json').write_text('{}')
            (p/'distributed_validation.jsonl').write_text('{"passed":false}\n')
            with self.assertRaisesRegex(ValueError,'did not pass'):
                verify_distributed(p,p,{'record_count':50},'a'*64)
    def test_large_snapshot_cannot_use_in_memory_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'metadata_snapshot.json').write_text(json.dumps({'status':'frozen','record_count':50000,'requested_records':50000,'artifacts':[]}))
            with patch.object(sys,'argv',['verify_metadata_pipeline.py',str(p),str(p),'--mode','python']), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error: main()
            self.assertEqual(error.exception.code,2)

if __name__ == '__main__': unittest.main()
