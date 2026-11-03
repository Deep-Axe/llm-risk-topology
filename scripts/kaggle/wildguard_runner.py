"""Reuse v7, generate the six validation assistants, and label all eight runs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tarfile

CODE_REVISION = 'SET_BY_PREPARE'
UPSTREAM_REVISION = '843b3d761170d20e60a6ad3113fd7710bd6ca0ef'
UPSTREAM_VERSION = 7
REPOSITORY = 'https://github.com/Deep-Axe/llm-risk-topology.git'
MODELS = ('mistralai/Mistral-7B-Instruct-v0.1', 'HuggingFaceH4/zephyr-7b-beta')
EXTRA_MODELS = ('Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct',
                'NousResearch/Hermes-3-Llama-3.1-8B', 'dphn/dolphin-2.9.4-llama3.1-8b',
                'allenai/Llama-3.1-Tulu-3-8B', 'teknium/OpenHermes-2.5-Mistral-7B')


def run_config(value=None):
    config = {'models': list(EXTRA_MODELS), 'splits': ['validation'],
              'model_manifest': 'manifests/models/scale6.bundle.json'}
    config.update(value or {})
    if (len(config['models']) != len(set(config['models'])) or
            not set(config['models']).issubset(EXTRA_MODELS)):
        raise ValueError('Extra models must be a unique subset of the registered six-model panel')
    if config['splits'] != ['validation']:
        raise ValueError('The registered scale-up generates validation only')
    if config['model_manifest'] != 'manifests/models/scale6.bundle.json':
        raise ValueError('Use the frozen six-model manifest')
    return config


def save_checkpoint(working, project, outputs, state):
    from generate_responses import atomic_write
    atomic_write(outputs / 'checkpoint.json', state)
    temporary = working / 'pilot_outputs.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as archive:
        archive.add(outputs, arcname='pilot_outputs')
        responses = project / 'data/processed/responses'
        if responses.exists():
            archive.add(responses, arcname='responses')
    temporary.replace(working / 'pilot_outputs.tar.gz')
    print('Checkpoint saved: /kaggle/working/pilot_outputs.tar.gz', state, flush=True)


def generate_extra_runs(project, outputs, config, command, checkpoint, report):
    panel = json.loads((project / config['model_manifest']).read_text())['model_manifests'][0]
    if panel['status'] != 'frozen' or not set(config['models']).issubset(m['model_id'] for m in panel['models']):
        raise ValueError('Extra-model manifest is not frozen or lacks a requested model')
    runs = []
    for index, model in enumerate(config['models']):
        command(['python/generate_responses.py', '--prompts',
                 'data/processed/prompts/pilot_v5/experiment.bundle.json', '--models', config['model_manifest'],
                 '--model', model, '--backend', 'vllm', '--tensor-parallel', '2', '--batch-size', '16',
                 '--max-model-len', '4096', '--max-input-tokens', '4096', '--max-new-tokens', '512',
                 '--seed', '42', '--split', *config['splits']], f'extra_generate_{index}')
        candidates = []
        for path in (project / 'data/processed/responses').glob('*/*/run.bundle.json'):
            bundle = json.loads(path.read_text());run = bundle['run_manifests'][0]
            if (run['runtime']['configuration']['model_id'] == model and
                    run['model_manifest_id'] == panel['manifest_id']):
                candidates.append(path.parent)
        if len(candidates) != 1 or not json.loads((candidates[0] / 'stats.json').read_text())['completed']:
            raise ValueError('Expected one complete validation run for ' + model)
        folder = candidates[0]
        command(['python/evaluate_responses.py', 'label', '--run', str(folder),
                 '--labeler', 'refusal_rules'], f'extra_rules_{index}')
        report(folder, outputs / 'model_reports')
        runs.append(folder)
        checkpoint({'phase': 'generation', 'completed_model': model,
                    'completed_extra_models': index + 1})
        cache = Path(os.environ['HF_HOME']) / 'hub'
        target = cache / ('models--' + model.replace('/', '--'))
        if target.exists():
            if target.resolve().parent != cache.resolve():
                raise ValueError('Unexpected checkpoint cache path')
            shutil.rmtree(target)
    return runs


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_source(archive, destination):
    """Extract only regular files/directories, after checking the entire member list."""
    if destination.exists():
        raise ValueError('Use a fresh source extraction folder')
    with tarfile.open(archive, 'r:gz') as stream:
        names = set()
        for member in stream.getmembers():
            name = member.name
            if (name in names or PureWindowsPath(name).drive or '\\' in name or
                    PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
                    not (destination / name).resolve().is_relative_to(destination.resolve()) or
                    not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe upstream archive member')
            names.add(name)
        destination.mkdir(parents=True)
        stream.extractall(destination)
    return destination


def source_tree(inputs, working):
    archives = sorted(inputs.rglob('pilot_outputs.tar.gz'))
    if len(archives) == 1:
        digest = file_hash(archives[0])
        destination = working / 'v7_source'
        receipt = working / 'v7_source.sha256'
        if destination.exists():
            if not receipt.is_file() or receipt.read_text().strip() != digest:
                raise ValueError('Existing source extraction is incomplete or came from a different archive')
        else:
            extract_source(archives[0], destination)
            receipt.write_text(digest + '\n')
        return destination, {'archive_path': str(archives[0]), 'archive_sha256': digest}
    if archives:
        raise ValueError('Attach exactly one upstream pilot output')
    markers = [p for p in inputs.rglob('code_revision.txt') if p.parent.name == 'pilot_outputs']
    if len(markers) != 1:
        raise ValueError('v7 outputs are not available. Wait for generation to finish and attach '
                         'the two-model pilot under Notebook inputs (kernel_sources).')
    return markers[0].parent.parent, {'extracted_source_path': str(markers[0].parent.parent)}


def copy_runs(source, project, prompt_bundle, expected_models=MODELS, expected_model_manifest=None):
    """Validate complete dev/validation runs before using their writable copies."""
    from data_contracts import validate_bundle, verify_artifacts
    from generate_responses import Shards
    revision = (source / 'pilot_outputs/code_revision.txt').read_text().strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError('Upstream code revision is not the pinned v7 generation code')
    roots = [p for p in (source / 'responses', source / 'project/data/processed/responses') if p.is_dir()]
    if len(roots) != 1:
        raise ValueError('Expected exactly one upstream response tree')
    manifest = prompt_bundle['prompt_manifests'][0]
    wanted = {p['prompt_id'] for p in prompt_bundle['prompts']
              if manifest['group_splits'][p['prompt_group_id']] in {'development', 'validation'}}
    folders, seen, provenance = [], set(), []
    for path in sorted(roots[0].glob('*/*/run.bundle.json')):
        context = json.loads(path.read_text())
        run = context['run_manifests'][0]
        config = run['runtime']['configuration']
        model = config['model_id']
        if model not in expected_models or model in seen:
            raise ValueError('Unexpected or duplicate upstream assistant run')
        stats = json.loads(path.with_name('stats.json').read_text())
        if not stats['completed']:
            raise ValueError('Upstream assistant generation is incomplete: ' + model)
        if context['prompt_manifests'][0] != manifest or context['prompts'] != prompt_bundle['prompts']:
            raise ValueError('Upstream prompt corpus differs from the frozen pilot')
        if expected_model_manifest is not None and context['model_manifests'][0] != expected_model_manifest:
            raise ValueError('Upstream model pins differ from the frozen pilot')
        if (set(config['selection']['splits']) != {'development', 'validation'} or
                set(config['selection']['prompt_ids']) != wanted or config['selection']['limit'] is not None):
            raise ValueError('Upstream selection must cover development and validation only')
        validate_bundle(context)
        original = path.parent
        if any(p.is_symlink() for p in original.rglob('*')):
            raise ValueError('Symlinks are not accepted in an upstream run')
        destination = project / 'data/processed/responses' / original.relative_to(roots[0])
        if destination.exists():
            # Keep committed labels for same-session resume; never overwrite changed generations.
            for item in [path, original / 'stats.json', original / 'responses.checksums.json',
                         *original.glob('responses-*.jsonl')]:
                if not (destination / item.name).is_file() or file_hash(item) != file_hash(destination / item.name):
                    raise ValueError('Existing generation copy differs from upstream')
        else:
            shutil.copytree(original, destination)
        journal = json.loads((destination / 'responses.checksums.json').read_text())
        shards = {p.name: p for p in destination.glob('responses-*.jsonl')}
        if set(journal) != set(shards) or any(file_hash(shards[name]) != value
                                              for name, value in journal.items()):
            raise ValueError('Upstream response checksum journal mismatch')
        rows = list(Shards(destination, 'responses', 'response_id').rows.values())
        if {r['prompt_id'] for r in rows} != wanted or len(rows) != len(wanted):
            raise ValueError('Upstream response coverage is incomplete or includes test prompts')
        if stats['run_id'] != run['run_id'] or stats['model_id'] != model or stats['responses'] != len(rows):
            raise ValueError('Upstream generation statistics disagree with the run')
        validate_bundle({**context, 'responses': rows})
        verify_artifacts(context)
        seen.add(model)
        folders.append(destination)
        provenance.append({'model_id': model, 'run_id': run['run_id'], 'responses': len(rows),
                           'run_bundle_sha256': file_hash(path), 'generation_stats_sha256':
                           file_hash(original / 'stats.json'), 'response_shards': {
                               p.name: file_hash(p) for p in sorted(original.glob('responses-*.jsonl'))}})
        print('Verified upstream run:', model, len(rows), 'responses', flush=True)
    if seen != set(expected_models):
        raise ValueError('Both completed v7 assistant runs are required')
    return folders, provenance


def label_runs(runs, outputs, command, after_model=None):
    """Only label stored responses and export the audit; no assistant generation command."""
    for index, run in enumerate(runs):
        command(['python/evaluate_responses.py', 'label', '--run', str(run), '--labeler',
                 'wildguard', '--tensor-parallel', '2'], f'wildguard_{index}')
        label_bundles = list((run / 'labels').glob('wildguard-*/validation.bundle.json'))
        if len(label_bundles) != 1:
            raise ValueError('Expected one validated WildGuard label bundle per assistant')
        command(['python/data_contracts.py', 'validate', str(label_bundles[0]),
                 '--verify-artifacts'], f'validate_labels_{index}')
        if after_model:
            after_model(run, index)
    command(['python/evaluate_responses.py', 'audit', '--runs', *map(str, runs),
             '--overlap-groups',
             'data/processed/prompts/pilot_v5/wildguardmix_train_overlap_groups_v5.txt',
             '--output', str(outputs / 'manual_audit.csv')], 'audit')


def main():
    if CODE_REVISION == 'SET_BY_PREPARE':
        raise ValueError('Stage with scripts/kaggle/prepare_wildguard.py first')
    working, inputs = Path('/kaggle/working'), Path('/kaggle/input')
    project, outputs = working / 'project', working / 'pilot_outputs'
    outputs.mkdir(exist_ok=True)
    config = run_config(json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
                        if '--config' in sys.argv else globals().get('RUN_CONFIG'))
    if not project.exists():
        subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(project)], check=True)
    subprocess.run(['git', 'checkout', '--detach', CODE_REVISION], cwd=project, check=True)
    spec = importlib.util.spec_from_file_location('pilot_helpers', project / 'scripts/kaggle/runner.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers.check_network()
    # Fail before dependency/model installation if the web secret is not attached.
    helpers.load_wildguard_token()
    os.environ.update({'HF_HOME': '/kaggle/temp/llm-risk-hf', 'VLLM_USE_V1': '0',
                       'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'NCCL_P2P_DISABLE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'})
    datasets = [p.parent for p in inputs.rglob('payload.json')
                if (p.parent / 'pilot_v5.zip').is_file() or (p.parent / 'pilot_v5').is_dir()]
    if len(datasets) != 1:
        raise ValueError('Attach the prepared pilot_v5 prompt dataset')
    helpers.install_payload(datasets[0], project)
    # Check that generation outputs are attached before installing GPU dependencies.
    source_tree(inputs, working)
    python = str(helpers.create_inference_environment(working))
    subprocess.run([python, '-m', 'pip', 'install', '-r', 'scripts/kaggle/requirements.txt'],
                   cwd=project, check=True)
    # Re-enter in the managed environment: validators require its pinned dependencies.
    if '--label-worker' not in sys.argv:
        worker = working / 'wildguard_worker.py'
        worker.write_text((project / 'scripts/kaggle/wildguard_runner.py').read_text().replace(
            "CODE_REVISION = 'SET_BY_PREPARE'", f"CODE_REVISION = '{CODE_REVISION}'", 1))
        config_path = working / 'scale_config.json'
        config_path.write_text(json.dumps(config))
        subprocess.run([python, '-u', str(worker), '--config', str(config_path), '--label-worker'],
                       cwd=project, check=True)
        return
    subprocess.run([python, '-c', 'import torch; assert torch.cuda.device_count() >= 2, "Select T4 x2"'],
                   check=True)
    sys.path.insert(0, str(project / 'python'))
    sys.path.insert(0, str(project / 'scripts/kaggle'))
    from generate_responses import atomic_write
    from report_model import report_model
    runs = []
    state = {'phase': 'initializing'}
    def checkpoint(status):
        state.update(status)
        save_checkpoint(working, project, outputs, dict(state))
    try:
        source, provenance = source_tree(inputs, working)
        prompt_bundle = json.loads((project / 'data/processed/prompts/pilot_v5/experiment.bundle.json').read_text())
        model_manifest = json.loads((project / 'manifests/models/pilot.bundle.json').read_text())['model_manifests'][0]
        runs, provenance['runs'] = copy_runs(source, project, prompt_bundle,
                                             expected_model_manifest=model_manifest)
        provenance.update({'upstream_version': UPSTREAM_VERSION, 'upstream_code_revision': UPSTREAM_REVISION,
                           'labeling_code_revision': CODE_REVISION})
        atomic_write(outputs / 'source_input.json', provenance)
        (outputs / 'code_revision.txt').write_text(CODE_REVISION + '\n')
        command = lambda args, name: helpers.stream_command(python, args, project, outputs, name)
        atomic_write(outputs / 'run_config.json', config)
        checkpoint({'phase': 'upstream_copied', 'upstream_runs': len(runs)})
        runs.extend(generate_extra_runs(project, outputs, config, command, checkpoint, report_model))
        def labeled(run, index):
            report_model(run, outputs / 'model_reports')
            checkpoint({'phase': 'wildguard', 'completed_labels_models': index + 1,
                        'run_folder': str(run)})
        label_runs(runs, outputs, command, after_model=labeled)
        # Reuse the pilot summary format for the downstream graph and manual audit.
        command(['scripts/kaggle/summarize_pilot.py', '--extracted', str(project / 'data/processed'),
                 '--output', str(outputs / 'summary')], 'summary')
        state.update({'phase': 'complete', 'models': len(runs)})
    except BaseException as exc:
        state.update({'phase': 'failed_or_interrupted', 'error_type': type(exc).__name__})
        raise
    finally:
        checkpoint(state)


if __name__ == '__main__':
    main()
