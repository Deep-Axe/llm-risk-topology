"""Kaggle entry point: clone pinned code, verify prompts, generate, label and pack outputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

CODE_REVISION = 'SET_BY_PREPARE'
REPOSITORY = 'https://github.com/Deep-Axe/llm-risk-topology.git'
MODELS = ['mistralai/Mistral-7B-Instruct-v0.1', 'HuggingFaceH4/zephyr-7b-beta']
SPLITS = ['development', 'validation']
HF_SECRET_NAMES = ('HF_Token', 'HF_TOKEN', 'big_data_project')
INFERENCE_PYTHON = '3.11.14'
UV_VERSION = '0.9.18'


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
    if checksum(archive) != metadata['archive_sha256']:
        raise ValueError('Dataset ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist()) != set(metadata['files']):
            raise ValueError('Dataset ZIP member list mismatch')
        for member in stream.infolist():
            target = (project / member.filename).resolve()
            if not target.is_relative_to(project.resolve()) or member.filename.startswith('/'):
                raise ValueError('ZIP member escapes the project')
        stream.extractall(project)
    for name, expected in metadata['files'].items():
        if checksum(project / name) != expected:
            raise ValueError('Dataset member checksum mismatch')


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


def main():
    if CODE_REVISION == 'SET_BY_PREPARE':
        raise ValueError('Stage the kernel with scripts/kaggle/prepare.py first')
    working = Path('/kaggle/working')
    project = working / 'project'
    outputs = working / 'pilot_outputs'
    outputs.mkdir(exist_ok=True)
    get_secret = lambda name: None
    if not any(os.environ.get(name) for name in HF_SECRET_NAMES):
        from kaggle_secrets import UserSecretsClient
        get_secret = UserSecretsClient().get_secret
    token, token_source = read_hf_token(os.environ, get_secret)
    print('Hugging Face credential source:', token_source)
    os.environ['HF_TOKEN'] = token
    os.environ['HF_HOME'] = '/kaggle/temp/llm-risk-hf'
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if not project.exists():
        subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(project)], check=True)
    subprocess.run(['git', 'checkout', '--detach', CODE_REVISION], cwd=project, check=True)
    markers = list(Path('/kaggle/input').rglob('payload.json'))
    matches = [path.parent for path in markers if (path.parent / 'pilot_v5.zip').exists()]
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
        with (outputs / (name + '.log')).open('a', encoding='utf-8') as stream:
            subprocess.run([python, *arguments], cwd=project, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

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
