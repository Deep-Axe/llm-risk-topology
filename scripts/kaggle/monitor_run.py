"""Continuously stream one saved Kaggle run, reconnect, record progress and recover its final archive."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

MODELS = ['mistralai/Mistral-7B-Instruct-v0.1', 'HuggingFaceH4/zephyr-7b-beta',
          'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',
          'NousResearch/Hermes-3-Llama-3.1-8B', 'dphn/dolphin-2.9.4-llama3.1-8b',
          'allenai/Llama-3.1-Tulu-3-8B', 'teknium/OpenHermes-2.5-Mistral-7B']
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def redact(text):
    text = re.sub(r'\bhf_[A-Za-z0-9]{12,}\b', '[REDACTED]', text)
    text = re.sub(r'\bKGAT[A-Za-z0-9_-]{12,}\b', '[REDACTED]', text)
    return re.sub(r'(https?://[^\s?]+)\?\S+', r'\1?[REDACTED]', text)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def monitor_lock(folder):
    with (folder / 'monitor.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Progress:
    def __init__(self, kernel, version, session_id):
        self.data = {'kernel': kernel, 'version': version, 'session_id': session_id,
                     'monitor_pid': os.getpid(), 'started_at': utc(), 'status': 'CONNECTING',
                     'stream': 'connecting', 'events_seen': 0, 'reconnections': 0,
                     'models': {}, 'recent_errors': [], 'stage': None, 'archive': None}
        self.label_model = None

    def record(self, text):
        text = redact(text)
        self.data['last_log_at'] = utc()
        stage = re.search(r'\[(extra_generate|extra_rules|wildguard)_(\d+)\]', text)
        if stage:
            kind, index = stage.group(1), int(stage.group(2))
            model_index = index if kind == 'wildguard' else index + 2
            if model_index < len(MODELS):
                self.data['stage'] = {'kind': kind, 'model': MODELS[model_index]}
                if kind == 'wildguard':
                    self.label_model = MODELS[model_index]
        for match in re.finditer(r'([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+):\s*(\d+)/(\d+) committed', text):
            model, done, total = match.groups()
            record = self.data['models'].setdefault(model, {})
            record['responses_committed'] = max(record.get('responses_committed', 0), int(done))
            record['responses_total'] = int(total)
            self.data['last_progress_at'] = utc()
        for match in re.finditer(r'Verified upstream run:\s*(\S+)\s+(\d+) responses', text):
            model, count = match.groups()
            self.data['models'].setdefault(model, {}).update({
                'responses_committed': int(count), 'responses_total': int(count), 'reused_from_v7': True})
        if self.label_model:
            for match in re.finditer(r'wildguard:\s*(\d+)/(\d+) labels committed', text):
                done, total = map(int, match.groups())
                record = self.data['models'].setdefault(self.label_model, {})
                record['wildguard_labels'] = max(record.get('wildguard_labels', 0), done)
                record['wildguard_total'] = total
                self.data['last_progress_at'] = utc()
        marker = 'Checkpoint saved: /kaggle/working/pilot_outputs.tar.gz '
        for line in text.splitlines():
            if marker in line:
                try:
                    self.data['checkpoint'] = ast.literal_eval(line.split(marker, 1)[1])
                    self.data['checkpoint_at'] = utc()
                except (ValueError, SyntaxError):
                    pass
        if 'Traceback (most recent call last)' in text or re.search(r'\b\w*(?:Error|Exception):', text):
            item = {'time': utc(), 'message': text[-1500:]}
            if not any(r['message'] == item['message'] for r in self.data['recent_errors']):
                self.data['recent_errors'] = [*self.data['recent_errors'][-19:], item]
        return text


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi();api.authenticate()
    return api


def stream_logs(progress, lock, stop, folder):
    api = api_client()
    delay = 2
    while not stop.is_set():
        try:
            with lock:
                progress.data['stream'] = 'connected'
            count = 0
            with (folder / 'live.log').open('a', encoding='utf-8') as log:
                for event in api.kernels_logs_stream(progress.data['kernel'] + '/' + str(progress.data['version'])):
                    if stop.is_set():
                        return
                    count += 1
                    with lock:
                        if count <= progress.data['events_seen']:
                            continue  # The server replays prior events on reconnect.
                        progress.data['events_seen'] = count
                        text = progress.record(str(event.get('data', '')))
                    log.write(text + '\n');log.flush()
                    delay = 2
            with lock:
                progress.data['stream'] = 'ended'
            return
        except Exception as exc:
            with lock:
                progress.data['stream'] = 'reconnecting'
                progress.data['reconnections'] += 1
                progress.data['last_connection_error'] = {'time': utc(), 'type': type(exc).__name__}
            if stop.wait(delay):
                return
            delay = min(delay * 2, 60)


def recover_archive(api, progress, destination):
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    destination.mkdir(parents=True, exist_ok=True)
    name = 'pilot_outputs_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '.tar.gz'
    target = destination / name
    for attempt in range(5):
        try:
            with api.build_kaggle_client() as client:
                request = ApiDownloadKernelOutputRequest()
                request.owner_slug, request.kernel_slug = progress.data['kernel'].split('/', 1)
                request.version_number = progress.data['version']
                request.file_path = 'pilot_outputs.tar.gz'
                response = client.kernels.kernels_api_client.download_kernel_output(request)
                response.raise_for_status()
                temporary = target.with_suffix('.gz.download')
                try:
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                finally:
                    response.close()
                with temporary.open('rb') as stream:
                    if stream.read(2) != b'\x1f\x8b':
                        raise ValueError('The output endpoint did not return a gzip archive')
                temporary.replace(target)
            with target.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': digest,
                      'downloaded_at': utc(), 'validation_pending': True}
            atomic_json(target.with_suffix('.download.json'), result)
            return result
        except Exception as exc:
            progress.data['archive_download_error'] = {'time': utc(), 'type': type(exc).__name__,
                                                      'attempt': attempt + 1}
            if attempt < 4:
                time.sleep(30)  # Allow time for Kaggle to publish terminated-run outputs.
    return {'available': False, 'validation_pending': True}


def monitor(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with monitor_lock(args.output):
        progress = Progress(args.kernel, args.version, args.session_id)
        lock, stop = threading.Lock(), threading.Event()
        api = api_client()
        thread = None
        next_poll = 0
        try:
            while True:
                now = time.monotonic()
                if now >= next_poll:
                    try:
                        response = api.kernels_status(args.kernel + '/' + str(args.version))
                        status = str(response.status).rsplit('.', 1)[-1]
                        with lock:
                            progress.data['status'] = status
                            progress.data['status_checked_at'] = utc()
                            if response.failure_message:
                                progress.data['failure_message'] = redact(response.failure_message)
                        if status in TERMINAL:
                            stop.set()
                            with lock:
                                progress.data['archive_download'] = 'in_progress'
                                atomic_json(args.output / 'status.json', progress.data)
                            if args.download_to:
                                progress.data['archive'] = recover_archive(api, progress, args.download_to)
                            progress.data['monitor_finished_at'] = utc()
                            atomic_json(args.output / 'status.json', progress.data)
                            atomic_json(args.output / 'finished.json', progress.data)
                            print('Run finished:', status, flush=True)
                            return
                        if thread is None or not thread.is_alive():
                            thread = threading.Thread(target=stream_logs,
                                                      args=(progress, lock, stop, args.output), daemon=True)
                            thread.start()
                    except Exception as exc:
                        with lock:
                            progress.data['status_check_error'] = {'time': utc(), 'type': type(exc).__name__}
                    next_poll = now + 600  # Status API at most once every ten minutes; logs stream continuously.
                with lock:
                    progress.data['monitor_heartbeat_at'] = utc()
                    last_log = progress.data.get('last_log_at', progress.data['started_at'])
                    quiet = (datetime.now(timezone.utc) - datetime.fromisoformat(last_log)).total_seconds()
                    progress.data['seconds_since_log'] = round(quiet)
                    progress.data['attention'] = 'No streamed output for over 10 minutes' if quiet > 600 else None
                    atomic_json(args.output / 'status.json', progress.data)
                time.sleep(5)
        finally:
            stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--session-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-to', type=Path)
    monitor(parser.parse_args())
