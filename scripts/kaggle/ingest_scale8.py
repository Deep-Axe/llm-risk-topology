"""Download only the result archive, validate it, and stage completed runs.

Use --archive PATH --scratch for an offline rehearsal without changing production data.
The production path also runs the existing HDFS/Spark/Hive stager; it never restarts Hadoop.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_artifacts import ROOT, file_hash, load_run


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def download(kernel, version, target):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    api = KaggleApi()
    api.authenticate()
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = kernel.split('/', 1)
    if version is not None:
        request.version_number = version
    request.file_path = 'pilot_outputs.tar.gz'
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        try:
            response.raise_for_status()
            temporary = target.with_suffix('.download')
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            with temporary.open('rb') as stream:
                if stream.read(2) != b'\x1f\x8b':
                    raise ValueError('The output endpoint did not return gzip')
            temporary.replace(target)
        finally:
            response.close()


def short_name(run):
    model = run['context']['run_manifests'][0]['runtime']['configuration']['model_id']
    import re
    name = re.sub('[^a-z0-9]+', '_', model.lower()).strip('_')
    # The run suffix prevents collisions between settings or prompt selections.
    return name + '_' + run['context']['run_manifests'][0]['run_id'][4:12]


def flatten(project, runs, target):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    bootstrap = ('import runpy,sys,torch; torch.set_num_threads(2); '
                 'assert not torch.cuda.is_available(); sys.argv=sys.argv[1:]; '
                 'runpy.run_path(sys.argv[0],run_name="__main__")')
    subprocess.run([sys.executable, '-X', 'utf8', '-u', '-c', bootstrap,
                    'python/stage_layer2.py', '--runs', *[str(r['folder']) for r in runs],
                    '--output', str(target), '--batch-size', '32'], cwd=project, env=env, check=True)


def copy_audit(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_hash(source) != file_hash(destination):
        # An existing CSV may contain the owner's annotations.
        destination = destination.with_name('pilot_audit_' + file_hash(source)[:12] + '.csv')
    if not destination.exists():
        shutil.copyfile(source, destination)
    return str(destination)


def ingest(output, project, archive=None, kernel=None, version=None, scratch=False):
    # A fresh folder keeps interrupted extraction/staging from looking complete.
    output.mkdir(parents=True, exist_ok=False)
    if archive is None:
        archive = output / 'pilot_outputs.tar.gz'
        download(kernel, version, archive)
    archive = archive.resolve()
    evidence = {'kernel': kernel, 'version': version, 'archive': str(archive),
                'bytes': archive.stat().st_size, 'sha256': file_hash(archive),
                'time': datetime.now(timezone.utc).isoformat(), 'scratch': scratch,
                'status': 'validation_pending', 'runs': []}
    save_json(output / 'ingestion.json', evidence)
    from summarize_pilot import extract_archive
    extracted = output / 'extracted'
    extract_archive(archive, extracted)
    # Validate every run before publishing any production staging.
    runs = []
    for path in sorted(extracted.rglob('run.bundle.json')):
        print('Validating archived run: ' + str(path.parent), flush=True)
        runs.append(load_run(path.parent))
    if not runs:
        raise ValueError('Archive contains no response runs')
    for run in runs:
        labels = Counter(r['labeler_id'] for r in run['labels'])
        item = {'model': run['context']['run_manifests'][0]['runtime']['configuration']['model_id'],
                'run_id': run['context']['run_manifests'][0]['run_id'],
                'completed': bool(run['stats'].get('completed')), 'responses': len(run['responses']),
                'labels': dict(labels), 'wildguard_labels': labels.get('wildguard', 0),
                'label_errors': sum(bool(r.get('error')) for r in run['labels'])}
        evidence['runs'].append(item)
        print(json.dumps(item), flush=True)
    evidence['status'] = 'validated'
    save_json(output / 'ingestion.json', evidence)
    for run, item in zip(runs, evidence['runs']):
        if not item['completed']:
            item['staging'] = 'skipped_incomplete_generation'
            continue
        name = short_name(run)
        if scratch:
            target = output / 'staged' / ('layer2_flat_' + name)
            flatten(project, [run], target)
            item['staging'] = str(target)
            item['cluster'] = 'not_run_scratch'
        else:
            from stage_finished_run import stage
            item['staging'] = stage(project, run['folder'], name)
        save_json(output / 'ingestion.json', evidence)
    pair = [r for r in runs if r['stats'].get('completed') and any(
        key in r['context']['run_manifests'][0]['runtime']['configuration']['model_id'].lower()
        for key in ['mistral-7b-instruct-v0.1', 'zephyr-7b-beta'])]
    if len(pair) == 2 and all(
        {r['response_id'] for r in run['labels'] if r['labeler_id'] == 'wildguard'} ==
        {r['response_id'] for r in run['responses']} for run in pair):
        target = (output / 'staged' if scratch else project / 'data/processed/analysis') / 'layer2_flat_v7'
        # Publish the combined replacement only after flattening succeeds.
        pending = output / 'combined_pending'
        flatten(project, pair, pending)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_name(target.name + '_before_' + output.name)
            target.rename(backup)
        pending.rename(target)
        evidence['combined_v7'] = str(target)
    audits = list(extracted.rglob('manual_audit.csv'))
    if len(audits) > 1:
        raise ValueError('Ambiguous manual audit files')
    report = output / 'report' if scratch else project / 'report'
    evidence['audit'] = copy_audit(audits[0], report / 'evaluator/pilot_audit.csv') if audits else 'not_in_archive'
    from behavior_summary import build
    from run_artifacts import discover_runs
    folders = [r['folder'] for r in runs if r['stats'].get('completed')] if scratch else discover_runs(
        project / 'data/processed/responses')
    # Include extracted runs even when the output folder is outside the default response tree.
    if not scratch:
        folders = list(dict.fromkeys([*folders, *[r['folder'] for r in runs if r['stats'].get('completed')]]))
        richest = {}
        for folder in folders:
            run = load_run(folder)
            key = run['context']['run_manifests'][0]['run_id']
            if key not in richest or len(run['labels']) > len(richest[key]['labels']):
                richest[key] = run
        folders = [r['folder'] for r in richest.values()]
    build(folders, report / 'pilot', {r['folder'].resolve(): r for r in runs})
    evidence['status'] = 'complete'
    save_json(output / 'ingestion.json', evidence)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--kernel')
    source.add_argument('--archive', type=Path)
    parser.add_argument('--version', type=int)
    parser.add_argument('--output', type=Path, required=True, help='New folder; never reuse a failed ingestion folder')
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--scratch', action='store_true', help='Stage locally; leave production reports and cluster alone')
    args = parser.parse_args()
    ingest(args.output.resolve(), args.project.resolve(), args.archive, args.kernel, args.version, args.scratch)
