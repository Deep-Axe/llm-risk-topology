"""Stage one completed run using main's unchanged embedding/WSL scripts and publish its evidence."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import os
import time

FIELDS = ['short_name', 'model', 'splits', 'responses', 'label_source',
          'staged_path', 'mllib_evidence_folder', 'time']


def wsl_path(path):
    value = Path(path).resolve().as_posix()
    if len(value) > 2 and value[1] == ':':
        return '/mnt/' + value[0].lower() + value[2:]
    return value


def windows_path(value):
    if re.match(r'^/mnt/[a-zA-Z]/', value):
        return Path(value[5].upper() + ':' + value[6:])
    return Path(value)


def log_text(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8', errors='replace')


def cluster_evidence(log):
    text = log_text(log)
    matches = re.findall(r'^Evidence:\s*(.+)$', text, re.MULTILINE)
    if 'LAYER2_PIPELINE_OK' not in text or len(matches) != 1:
        raise ValueError('Cluster pipeline did not publish a complete success marker')
    evidence = windows_path(matches[0].strip())
    with (evidence / 'timings.tsv').open(encoding='utf-8') as stream:
        timings = list(csv.DictReader(stream, delimiter='\t'))
    if {r['stage'] for r in timings} != {'hdfs_upload', 'spark_mllib', 'hive_queries'} or any(
            r['status'] != 'PASS' for r in timings):
        raise ValueError('All HDFS, Spark and Hive steps must pass before registering a run')
    lines = [r for r in (evidence / 'models.txt').read_text().splitlines() if 'LAYER2_MODEL_DONE' in r]
    if not lines:
        raise ValueError('Missing Spark model metrics')
    return evidence, lines


def wait_for_report(project, model):
    while True:
        for stats_path in (project / 'data/processed/responses').glob(
                '*/' + model.replace('/', '__') + '/stats.json'):
            stats = json.loads(stats_path.read_text())
            report = project / 'report/pilot' / (
                model.replace('/', '__') + '__' + stats['run_id'][4:20] + '.json')
            if stats['completed'] and report.is_file() and json.loads(report.read_text())['completed']:
                return stats_path.parent
        time.sleep(5)


def stage(project, run, short_name, existing_log=None):
    if not re.fullmatch('[a-z0-9_]+', short_name):
        raise ValueError('Use a simple lowercase staging name')
    sys.path.insert(0, str(project / 'python'))
    from data_contracts import validate_bundle, verify_artifacts
    from generate_responses import Shards, directory_lock, atomic_write
    import numpy as np
    import pandas as pd
    context = json.loads((run / 'run.bundle.json').read_text())
    stats = json.loads((run / 'stats.json').read_text())
    if not stats['completed']:
        raise ValueError('Generation must finish before staging')
    responses = list(Shards(run, 'responses', 'response_id').rows.values())
    validate_bundle({**context, 'responses': responses})
    verify_artifacts(context)
    output = project / 'data/processed/analysis' / ('layer2_flat_' + short_name)
    reports = project / 'report/pilot'
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    logs = reports / 'staging_logs';logs.mkdir(exist_ok=True)
    if existing_log is None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        bootstrap = ('import runpy,sys,torch; torch.set_num_threads(2); '
                     'assert not torch.cuda.is_available(); sys.argv=sys.argv[1:]; '
                     'runpy.run_path(sys.argv[0],run_name="__main__")')
        with (logs / f'{short_name}_{stamp}_embeddings.log').open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', '-c', bootstrap,
                            'python/stage_layer2.py', '--runs', str(run), '--output', str(output),
                            '--batch-size', '32'], cwd=project, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        pipeline_log = logs / f'{short_name}_{stamp}_cluster.log'
        with pipeline_log.open('w', encoding='utf-8') as stream:
            subprocess.run(['wsl.exe', '-e', 'bash', wsl_path(project / 'scripts/run_layer2_mllib_wsl.sh'),
                            wsl_path(output)], cwd=project, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        pipeline_log = existing_log
    evidence, metrics = cluster_evidence(pipeline_log)
    frame = pd.read_parquet(output / 'responses.parquet')
    successful = {r['response_id'] for r in responses if r['status'] == 'success'}
    if (set(frame['response_id']) != successful or frame['response_id'].duplicated().any() or
            set(frame['model_id']) != {stats['model_id']}):
        raise ValueError('Staged response coverage differs from the completed source run')
    vectors = np.vstack(frame['vector'].to_list())
    if (vectors.shape != (len(frame), 384) or not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
        raise ValueError('Expected normalized finite 384-dimensional MiniLM vectors')
    sources = sorted({json.loads(p.read_text())['labeler_id']
                      for p in (run / 'labels').glob('*/labeler.json')})
    row = dict(zip(FIELDS, [short_name, stats['model_id'], ','.join(sorted(frame['split'].unique())),
                           len(frame), ','.join(sources) or 'none', output.relative_to(project).as_posix(),
                           evidence.relative_to(project).as_posix(), datetime.now(timezone.utc).isoformat()]))
    with directory_lock(reports):
        registry = reports / 'staged_runs.tsv'
        if registry.exists():
            with registry.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                if reader.fieldnames != FIELDS:
                    raise ValueError('Existing staging registry has a different column contract')
                rows = list(reader)
        else:
            rows = []
        temporary = registry.with_suffix('.tsv.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
            writer.writeheader();writer.writerows([*rows, row])
        temporary.replace(registry)
    atomic_write(reports / f'{short_name}_staging.json', {
        **row, 'run_id': stats['run_id'], 'run_folder': str(run), 'metrics': metrics,
        'pipeline_log': str(pipeline_log), 'embedding_device': 'cpu',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'embedding_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'})
    with (project / 'project_notes/agent_rounds/round3/A_report.md').open('a', encoding='utf-8') as stream:
        stream.write(f"\nStaged {short_name}: {stats['model_id']}, {len(frame)} responses, labels {row['label_source']}. "
                     f"HDFS/Spark/Hive all passed. Evidence: {row['mllib_evidence_folder']}. "
                     f"Spark metrics: {'; '.join(metrics)}. Registered in report/pilot/staged_runs.tsv.\n")
    print(json.dumps(row), flush=True)
    for line in metrics:
        print(line, flush=True)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run', type=Path)
    selection.add_argument('--wait-for-model')
    parser.add_argument('--short-name', required=True)
    parser.add_argument('--existing-pipeline-log', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    run = args.run.resolve() if args.run else wait_for_report(project, args.wait_for_model)
    stage(project, run, args.short_name, args.existing_pipeline_log)
