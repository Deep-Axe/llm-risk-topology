import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from collect_metadata import initialize,append_page,load_checkpoint,finalize,atomic_json,ENDPOINT

class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.state = initialize(self.path,3)
    def first(self):
        append_page(self.path,self.state,set(),lambda u,p:([{'id':'test/a'},{'id':'test/b'}],ENDPOINT+'?cursor=2'))
    def test_resume_deduplicates_and_freezes_only_complete(self):
        self.first()
        state,seen = load_checkpoint(self.path,3)
        self.assertFalse((self.path/'metadata_snapshot.json').exists())
        with self.assertRaises(ValueError): finalize(self.path,state)
        append_page(self.path,state,seen,lambda u,p:([{'id':'test/b'},{'id':'test/c'}],None))
        result = finalize(self.path,state)
        self.assertEqual(result['record_count'],3)
        with gzip.open(self.path/'models.jsonl.gz','rt') as stream:
            self.assertEqual(len(list(stream)),3)
        with self.assertRaises(ValueError): load_checkpoint(self.path)
    def test_torn_checkpoint_adopts_written_page_without_refetch(self):
        original = json.loads((self.path/'collection_checkpoint.json').read_text())
        self.first()
        atomic_json(self.path/'collection_checkpoint.json',original)
        state,seen = load_checkpoint(self.path)
        append_page(self.path,state,seen,lambda u,p:self.fail('must adopt orphan'))
        self.assertEqual(seen,{'test/a','test/b'})
    def test_page_corruption_and_changed_target_rejected(self):
        self.first()
        with self.assertRaises(ValueError): load_checkpoint(self.path,4)
        (self.path/'pages/000000.json').write_text('{}')
        with self.assertRaises(ValueError): load_checkpoint(self.path)
    def test_unpublished_partial_page_is_ignored(self):
        (self.path/'pages/000000.json.partial').write_text('{broken')
        state,seen = load_checkpoint(self.path)
        self.assertEqual(seen,set())
        append_page(self.path,state,seen,lambda u,p:([{'id':'test/a'}],None))
        self.assertEqual(len(seen),1)
    def test_failure_before_page_commit_retains_prior_cursor(self):
        with self.assertRaises(RuntimeError):
            append_page(self.path,self.state,set(),lambda u,p:(_ for _ in ()).throw(RuntimeError('interrupted')))
        state,seen = load_checkpoint(self.path)
        self.assertEqual(state['next_url'],ENDPOINT)
        self.assertFalse(seen)

if __name__ == '__main__': unittest.main()
