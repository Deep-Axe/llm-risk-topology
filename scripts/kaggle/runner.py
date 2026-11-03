"""Kaggle entry point: clone pinned code, verify prompts, generate, label and pack outputs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

CODE_REVISION = 'SET_BY_PREPARE'
REPOSITORY = 'https://github.com/Deep-Axe/llm-risk-topology.git'
MODELS = ['mistralai/Mistral-7B-Instruct-v0.1', 'HuggingFaceH4/zephyr-7b-beta']
SPLITS = ['development', 'validation']
HF_SECRET_NAMES = ('HF_Token', 'HF_TOKEN', 'big_data_project')
INFERENCE_PYTHON = '3.11.14'
UV_VERSION = '0.9.18'


def check_network():
    results = {}
    for url in ['https://huggingface.co', 'https://github.com', 'https://www.kaggle.com']:
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                results[url] = 'HTTP ' + str(response.status)
        except urllib.error.HTTPError as exc:
            results[url] = 'HTTP ' + str(exc.code)
        except Exception as exc:
            results[url] = type(exc).__name__
    print('Network preflight:', json.dumps(results), flush=True)
    if not any(value.startswith('HTTP ') for value in results.values()):
        raise RuntimeError('Outbound network is unreachable; check this notebook Internet setting')
    return results


def read_hf_token(environ, get_secret):
    for name in HF_SECRET_NAMES:
        if environ.get(name):
            return environ[name], 'environment:' + name
    errors = []
    for name in HF_SECRET_NAMES:
        try:
            token = get_secret(name)
        except Exception as exc:
            message = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', str(exc))
            message = re.sub(r'(https?://[^\s?]+)\?\S+', r'\1?[REDACTED]', message)
            errors.append(name + ': ' + type(exc).__name__ + ': ' + message[:350])
            continue
        if token:
            return token, 'Kaggle secret:' + name
    details = '; '.join(errors) if errors else 'No value was returned for the accepted secret labels'
    raise ValueError('Kaggle secret retrieval failed: ' + details)


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install_payload(dataset, project):
    metadata = json.loads((dataset / 'payload.json').read_text())
    archive = dataset / 'pilot_v5.zip'
    targets = {}
    for name in metadata['files']:
        target = (project / name).resolve()
        if (not target.is_relative_to(project.resolve()) or
                PureWindowsPath(name).drive or '\\' in name or
                PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts):
            raise ValueError('Dataset member escapes the project')
        targets[name] = target
    if archive.is_file():
        if checksum(archive) != metadata['archive_sha256']:
            raise ValueError('Dataset ZIP checksum mismatch')
        with zipfile.ZipFile(archive) as stream:
            if (set(stream.namelist()) != set(metadata['files']) or
                    len(stream.namelist()) != len(metadata['files'])):
                raise ValueError('Dataset ZIP member list mismatch')
            # Verify every member before writing any files into the checkout.
            for name, expected in metadata['files'].items():
                with stream.open(name) as member:
                    if hashlib.file_digest(member, 'sha256').hexdigest() != expected:
                        raise ValueError('Dataset member checksum mismatch')
            for name, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(name) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
        layout = 'ZIP'
    else:
        root = (dataset / 'pilot_v5').resolve()
        sources = {}
        for name, expected in metadata['files'].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root):
                raise ValueError('Dataset member escapes the extracted folder')
            if not source.is_file():
                raise ValueError('Missing extracted dataset member: ' + name)
            if checksum(source) != expected:
                raise ValueError('Dataset member checksum mismatch')
            sources[name] = source
        for name, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[name], target)
        layout = 'extracted folder'
    for name, expected in metadata['files'].items():
        if checksum(project / name) != expected:
            raise ValueError('Dataset member checksum mismatch')
    print('Verified prompt payload:', layout, len(targets), 'files', flush=True)


def load_wildguard_token():
    get_secret = lambda name: None
    if not any(os.environ.get(name) for name in HF_SECRET_NAMES):
        from kaggle_secrets import UserSecretsClient
        get_secret = UserSecretsClient().get_secret
    token, token_source = read_hf_token(os.environ, get_secret)
    os.environ['HF_TOKEN'] = token
    print('Hugging Face credential source:', token_source, flush=True)


def create_inference_environment(working):
    environment = working / 'inference-env'
    if environment.exists():
        return environment / 'bin/python'
    bootstrap = working / 'uv-bootstrap'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(bootstrap),
                    'uv==' + UV_VERSION], check=True)
    bootstrap_env = dict(os.environ)
    bootstrap_env['PYTHONPATH'] = str(bootstrap)
    bootstrap_env['UV_PYTHON_INSTALL_DIR'] = str(working / 'managed-python')
    uv = [sys.executable, '-m', 'uv']
    subprocess.run([*uv, 'python', 'install', INFERENCE_PYTHON], env=bootstrap_env, check=True)
    subprocess.run([*uv, 'venv', '--python', INFERENCE_PYTHON, '--seed', str(environment)],
                   env=bootstrap_env, check=True)
    return environment / 'bin/python'


def stream_command(python, arguments, project, outputs, name, heartbeat_seconds=30):
    """Keep persistent subprocess logs visible while a model loads or generates."""
    log = outputs / (name + '.log')
    started = time.monotonic()
    finished = threading.Event()
    print(f'[{name}] START; live log: {log}', flush=True)
    with log.open('a', encoding='utf-8') as writer, log.open('r', encoding='utf-8', errors='replace') as reader:
        reader.seek(0, 2)

        def relay():
            last_message = time.monotonic()
            def emit(chunk):
                chunk = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', chunk)
                chunk = re.sub(r'(https?://[^\s?]+)\?\S+', r'\1?[REDACTED]', chunk)
                print(f'[{name}] ' + chunk, end='' if chunk.endswith('\n') else '\n', flush=True)
            while True:
                chunk = reader.read()
                if chunk:
                    emit(chunk)
                    last_message = time.monotonic()
                now = time.monotonic()
                if now - last_message >= heartbeat_seconds:
                    print(f'[{name}] RUNNING: {now - started:.0f}s elapsed; '
                          'waiting for subprocess output', flush=True)
                    last_message = now
                if finished.is_set():
                    remaining = reader.read()
                    if remaining:
                        emit(remaining)
                    return
                finished.wait(min(0.5, heartbeat_seconds))

        thread = threading.Thread(target=relay, daemon=True)
        thread.start()
        try:
            subprocess.run([python, '-u', *arguments], cwd=project, stdout=writer,
                           stderr=subprocess.STDOUT, check=True)
        except BaseException:
            print(f'[{name}] FAILED after {time.monotonic() - started:.1f}s; see {log}', flush=True)
            raise
        finally:
            # The subprocess has closed its file descriptor, so drain its last output.
            writer.flush()
            finished.set()
            thread.join()
    print(f'[{name}] COMPLETE in {time.monotonic() - started:.1f}s', flush=True)


def main():
    if CODE_REVISION == 'SET_BY_PREPARE':
        raise ValueError('Stage the kernel with scripts/kaggle/prepare.py first')
    working = Path('/kaggle/working')
    project = working / 'project'
    outputs = working / 'pilot_outputs'
    outputs.mkdir(exist_ok=True)
    check_network()
    try:
        load_wildguard_token()
    except Exception as exc:
        print('Hugging Face secret preflight:', type(exc).__name__,
              '- public assistants will run; WildGuard needs an attached secret.', flush=True)
    os.environ['HF_HOME'] = '/kaggle/temp/llm-risk-hf'
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if not project.exists():
        subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(project)], check=True)
    subprocess.run(['git', 'checkout', '--detach', CODE_REVISION], cwd=project, check=True)
    markers = list(Path('/kaggle/input').rglob('payload.json'))
    matches = [path.parent for path in markers
               if (path.parent / 'pilot_v5.zip').is_file() or (path.parent / 'pilot_v5').is_dir()]
    if len(matches) != 1:
        raise ValueError('Attach exactly one prepared pilot_v5 dataset')
    install_payload(matches[0], project)
    python = str(create_inference_environment(working))
    subprocess.run([python, '-c', 'import platform; print("Inference Python:", platform.python_version()); '
                    f'assert platform.python_version() == "{INFERENCE_PYTHON}"'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', '-r', 'scripts/kaggle/requirements.txt'],
                   cwd=project, check=True)
    with (outputs / 'pip_freeze.txt').open('w') as stream:
        subprocess.run([python, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    subprocess.run([python, '-c',
                    'import torch; assert torch.cuda.device_count() >= 2, "Select GPU T4 x2"'], check=True)
    (outputs / 'code_revision.txt').write_text(CODE_REVISION + '\n')
    runs = []

    def command(arguments, name):
        stream_command(python, arguments, project, outputs, name)

    try:
        command(['python/data_contracts.py', 'validate',
                 'data/processed/prompts/pilot_v5/experiment.bundle.json', '--verify-artifacts'], 'validate')
        for index, model in enumerate(MODELS):
            command(['python/generate_responses.py', '--prompts',
                     'data/processed/prompts/pilot_v5/experiment.bundle.json',
                     '--models', 'manifests/models/pilot.bundle.json', '--model', model,
                     '--backend', 'vllm', '--tensor-parallel', '2', '--max-model-len', '4096',
                     '--batch-size', '16', '--split', *SPLITS], f'generate_{index}')
            candidates = []
            for path in (project / 'data/processed/responses').glob('*/*/run.bundle.json'):
                bundle = json.loads(path.read_text())
                runtime = bundle['run_manifests'][0]['runtime']
                if runtime['engine'] == 'vllm' and runtime['configuration']['model_id'] == model:
                    candidates.append(path.parent)
            if len(candidates) != 1:
                raise ValueError('Expected one remote run folder per model')
            runs.append(candidates[0])
            command(['python/evaluate_responses.py', 'label', '--run', str(runs[-1]),
                     '--labeler', 'refusal_rules'], f'rules_{index}')
            # Release assistant weights before downloading the next 14.5 GB checkpoint.
            cache = Path(os.environ['HF_HOME']) / 'hub'
            checkpoint_cache = cache / ('models--' + model.replace('/', '--'))
            if checkpoint_cache.exists():
                if checkpoint_cache.resolve().parent != cache.resolve():
                    raise ValueError('Unexpected cache path')
                shutil.rmtree(checkpoint_cache)
        try:
            load_wildguard_token()
        except Exception as exc:
            (outputs / 'pipeline_status.json').write_text(json.dumps({
                'status': 'awaiting_wildguard_token',
                'public_assistant_runs': [str(run) for run in runs],
            }))
            raise RuntimeError('Public assistant generation and refusal-rule labels are saved. '
                               'Attach a Hugging Face secret before WildGuard labeling.') from exc
        for index, run in enumerate(runs):
            command(['python/evaluate_responses.py', 'label', '--run', str(run),
                     '--labeler', 'wildguard', '--tensor-parallel', '2'], f'wildguard_{index}')
        command(['python/evaluate_responses.py', 'audit', '--runs', *[str(run) for run in runs],
                 '--overlap-groups',
                 'data/processed/prompts/pilot_v5/wildguardmix_train_overlap_groups_v5.txt',
                 '--output', str(outputs / 'manual_audit.csv')], 'audit')
    finally:
        with tarfile.open(working / 'pilot_outputs.tar.gz', 'w:gz') as archive:
            archive.add(outputs, arcname='pilot_outputs')
            responses = project / 'data/processed/responses'
            if responses.exists():
                archive.add(responses, arcname='responses')
        print('Download /kaggle/working/pilot_outputs.tar.gz')


if __name__ == '__main__':
    main()
