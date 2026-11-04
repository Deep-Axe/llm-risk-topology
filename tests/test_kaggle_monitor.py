"""Check live-progress parsing and replay after a broken log connection without Kaggle calls."""
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('monitor_run', Path(__file__).resolve().parents[1] /
                                            'scripts/kaggle/monitor_run.py')
MONITOR = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MONITOR)


class MonitorTest(unittest.TestCase):
    def test_duplicate_and_replayed_counts_do_not_inflate_progress(self):
        p = MONITOR.Progress('owner/notebook', 6, 365504304)
        p.record('Verified upstream run: mistralai/Mistral-7B-Instruct-v0.1 4575 responses')
        p.record('[extra_generate_0] Qwen/Qwen2.5-7B-Instruct: 128/790 committed')
        p.record('[extra_generate_0] Qwen/Qwen2.5-7B-Instruct: 64/790 committed')
        self.assertEqual(p.data['models'][MONITOR.MODELS[2]]['responses_committed'], 128)
        self.assertTrue(p.data['models'][MONITOR.MODELS[0]]['reused_from_v7'])
        p.record("Checkpoint saved: /kaggle/working/pilot_outputs.tar.gz {'phase': 'generation', 'completed_extra_models': 1}")
        self.assertEqual(p.data['checkpoint']['completed_extra_models'], 1)

    def test_wildguard_progress_is_tied_to_its_model_and_secrets_are_redacted(self):
        p = MONITOR.Progress('owner/notebook', 6, 1)
        p.record('[wildguard_1] START')
        p.record('wildguard: 64/4575 labels committed')
        self.assertEqual(p.data['models'][MONITOR.MODELS[1]]['wildguard_labels'], 64)
        p.record('RuntimeError: failed with hf_privateToken123456789 https://example.com/file?signature=private')
        error = p.data['recent_errors'][0]['message']
        self.assertNotIn('privateToken', error)
        self.assertNotIn('signature=private', error)
        self.assertNotEqual(p.data['status'], 'ERROR')  # A log error is not proof of terminal job failure.

    def test_reconnect_skips_old_events_and_preserves_new_progress(self):
        events = [{'data': f'Qwen/Qwen2.5-7B-Instruct: {count}/790 committed'} for count in [64, 128, 192]]

        class FakeApi:
            calls = 0
            def kernels_logs_stream(self, kernel):
                self.calls += 1
                for event in events[:2] if self.calls == 1 else events:
                    yield event
                if self.calls == 1:
                    raise ConnectionError('fixture stream dropped')

        with tempfile.TemporaryDirectory() as tmp, patch.object(MONITOR, 'api_client', return_value=FakeApi()):
            p = MONITOR.Progress('owner/notebook', 6, 1)
            stop = SimpleNamespace(is_set=lambda: False, wait=lambda delay: False)
            MONITOR.stream_logs(p, threading.Lock(), stop, Path(tmp))
            self.assertEqual(p.data['events_seen'], 3)
            self.assertEqual(p.data['reconnections'], 1)
            self.assertEqual(p.data['models'][MONITOR.MODELS[2]]['responses_committed'], 192)
            self.assertEqual((Path(tmp) / 'live.log').read_text().count('committed'), 3)


if __name__ == '__main__':
    unittest.main()
