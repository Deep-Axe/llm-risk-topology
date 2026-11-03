"""A failed cluster stage must never appear in the analysis-ready registry."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import test_inference_labels as fixtures

SPEC = importlib.util.spec_from_file_location('steward', Path(__file__).resolve().parents[1] /
                                            'scripts/kaggle/stage_finished_run.py')
STEWARD = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(STEWARD)


class StewardTest(unittest.TestCase):
    def test_all_cluster_steps_and_metrics_required_including_utf16_shell_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);evidence = root / 'evidence';evidence.mkdir()
            timings = evidence / 'timings.tsv'
            timings.write_text('stage\tseconds\tstatus\nhdfs_upload\t1\tPASS\n'
                               'spark_mllib\t1\tPASS\nhive_queries\t1\tPASS\n')
            (evidence / 'models.txt').write_text('LAYER2_MODEL_DONE fixture recall=1 silhouette=0.2\n')
            log = root / 'pipeline.log'
            log.write_text('LAYER2_PIPELINE_OK\nEvidence: ' + str(evidence) + '\n', encoding='utf-16')
            path, metrics = STEWARD.cluster_evidence(log)
            self.assertEqual(path, evidence)
            self.assertEqual(len(metrics), 1)
            timings.write_text(timings.read_text().replace('hive_queries\t1\tPASS', 'hive_queries\t1\tFAIL'))
            with self.assertRaisesRegex(ValueError, 'All HDFS'):
                STEWARD.cluster_evidence(log)

    def test_failed_pipeline_does_not_publish_ready_row(self):
        case = fixtures.InferenceTest();case.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            run, _ = case.generate(project / 'responses')
            log = project / 'failed.log';log.write_text('RUN spark_mllib\nFAIL spark_mllib\n')
            with self.assertRaisesRegex(ValueError, 'success marker'):
                STEWARD.stage(project, run, 'fixture', log)
            self.assertFalse((project / 'report/pilot/staged_runs.tsv').exists())


if __name__ == '__main__':
    unittest.main()
