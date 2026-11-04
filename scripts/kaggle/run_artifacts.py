"""Read and validate saved response/label shards without changing their journals."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from data_contracts import validate_bundle, verify_artifacts


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shards(folder, kind, key):
    paths = sorted(folder.glob(kind + '-*.jsonl'))
    if not paths:
        journal = folder / (kind + '.checksums.json')
        if journal.exists() and read_json(journal):
            raise ValueError('Shard journal references missing files: ' + str(folder))
        return []
    journal = read_json(folder / (kind + '.checksums.json'))
    if set(journal) != {p.name for p in paths}:
        raise ValueError('Shard journal/file coverage mismatch: ' + str(folder))
    rows = []
    for path in paths:
        if file_hash(path) != journal[path.name]:
            raise ValueError('Shard checksum mismatch: ' + str(path))
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    if len({r[key] for r in rows}) != len(rows):
        raise ValueError('Duplicate shard identities: ' + str(folder))
    return rows


def load_run(folder):
    folder = Path(folder)
    context = read_json(folder / 'run.bundle.json')
    responses = shards(folder, 'responses', 'response_id')
    labels = []
    for policy in sorted((folder / 'labels').glob('*/labeler.json')):
        labels.extend(shards(policy.parent, 'response_labels', 'label_id'))
    validate_bundle({**context, 'responses': responses, 'response_labels': labels})
    verify_artifacts(context)
    for path in [folder / 'validation.bundle.json', *sorted((folder / 'labels').glob('*/validation.bundle.json'))]:
        if path.is_file():
            validate_bundle(read_json(path))
    stats_path = folder / 'stats.json'
    stats = read_json(stats_path) if stats_path.exists() else {'completed': False}
    run = context['run_manifests'][0]
    if stats.get('completed'):
        selected = set(run['runtime']['configuration']['selection']['prompt_ids'])
        if {r['prompt_id'] for r in responses} != selected or len(responses) != len(selected):
            raise ValueError('Completed run has incomplete response coverage')
    return {'folder': folder, 'context': context, 'responses': responses, 'labels': labels, 'stats': stats}


def discover_runs(root):
    """Prefer the richest label copy of a run; reject conflicting generation replicas."""
    candidates = {}
    paths = []
    for folder, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in {
            'inference-env', 'site-packages', '.git', '__pycache__', 'project', 'node_modules'}]
        if 'run.bundle.json' in files:
            paths.append(Path(folder) / 'run.bundle.json')
            directories[:] = []
    for path in sorted(paths):
        if not (path.parent / 'stats.json').is_file():
            continue
        stats = read_json(path.parent / 'stats.json')
        if not stats.get('completed'):
            continue
        key = stats['run_id']
        fingerprint = read_json(path.parent / 'responses.checksums.json')
        label_count = sum(1 for p in (path.parent / 'labels').glob('*/response_labels-*.jsonl')
                          for line in p.open(encoding='utf-8') if line.strip())
        if key in candidates and candidates[key][0] != fingerprint:
            raise ValueError('Conflicting copies of generation ' + key)
        if key not in candidates or label_count > candidates[key][1]:
            candidates[key] = (fingerprint, label_count, path.parent)
    return [value[2] for _, value in sorted(candidates.items())]


def intent(prompt):
    labels = prompt['source_labels']
    value = labels.get('prompt_intent')
    if value:
        return 'benign' if value == 'unharmful' else value
    if prompt['source_id'] == 'OpenSafetyLab/Salad-Data':
        return 'harmful'
    if prompt['source_id'] == 'paul-rottger/xstest':
        return {'safe': 'benign', 'unsafe': 'harmful'}.get(labels.get('label'), 'unspecified')
    return 'unspecified'
