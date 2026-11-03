"""Stage a private Kaggle dataset and kernel without uploading or running either.

Run after committing: python scripts/kaggle/prepare.py --code-revision FULL_SHA
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from data_contracts import validate_bundle, verify_artifacts
from generate_responses import atomic_write, file_hash, read_json


def prepare(output, revision, owner='deepax'):
    if re.fullmatch('[0-9a-f]{40}', revision) is None:
        raise ValueError('Pass the reviewed full code commit SHA')
    subprocess.run(['git', 'cat-file', '-e', revision + ':python/generate_responses.py'],
                   cwd=ROOT, check=True)
    pilot = ROOT / 'data/processed/prompts/pilot_v5'
    bundle = read_json(pilot / 'experiment.bundle.json')
    validate_bundle(bundle)
    verify_artifacts(bundle)
    provenance = read_json(pilot / 'preparation.json')
    for item in provenance['artifacts']:
        if file_hash(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen pilot artifact checksum mismatch')
    output.mkdir(parents=True, exist_ok=False)
    dataset, kernel = output / 'dataset', output / 'kernel'
    dataset.mkdir()
    kernel.mkdir()
    overlap = sorted({json.loads(line)['prompt_group_id'] for line in
                      (pilot / 'wildguard_overlap.jsonl').read_text(encoding='utf-8').splitlines()
                      if json.loads(line)['overlap']})
    contents = {}
    payload = dataset / 'pilot_v5.zip'
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pilot.iterdir()):
            if path.is_file():
                name = 'data/processed/prompts/pilot_v5/' + path.name
                archive.write(path, name)
                contents[name] = file_hash(path)
        name = 'data/processed/prompts/pilot_v5/wildguardmix_train_overlap_groups_v5.txt'
        text = '\n'.join(overlap) + '\n'
        archive.writestr(name, text)
        contents[name] = hashlib.sha256(text.encode('utf-8')).hexdigest()
    atomic_write(dataset / 'payload.json', {'archive_sha256': file_hash(payload), 'files': contents,
                                           'prompt_manifest_id': bundle['prompt_manifests'][0]['manifest_id']})
    atomic_write(dataset / 'source_terms.json', bundle['source_manifests'][0])
    atomic_write(dataset / 'dataset-metadata.json', {
        'title': 'LLM Risk Topology pilot v5', 'id': owner + '/llm-risk-topology-pilot-v5',
        'licenses': [{'name': 'other'}],
        'description': 'Pinned prompt-only research pilot. Source revisions, attribution URLs and '
                       'individual licenses are in source_terms.json. Keep this dataset private.'})
    runner = (ROOT / 'scripts/kaggle/runner.py').read_text(encoding='utf-8')
    runner = runner.replace("CODE_REVISION = 'SET_BY_PREPARE'", f"CODE_REVISION = '{revision}'")
    (kernel / 'runner.py').write_text(runner, encoding='utf-8', newline='\n')
    atomic_write(kernel / 'kernel-metadata.json', {
        'id': owner + '/llm-risk-topology-two-model-pilot', 'title': 'LLM Risk Topology two-model pilot',
        'code_file': 'runner.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [owner + '/llm-risk-topology-pilot-v5'],
        'competition_sources': [], 'kernel_sources': []})
    print('Staged dataset and kernel:', output)
    print('Review, then upload the private dataset:')
    print(f'kaggle datasets create -p "{dataset}"')
    print('Save the kernel without running it:')
    print(f'kaggle kernels push -p "{kernel}" --accelerator NvidiaTeslaT4 --no-run')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--owner', required=True, help='Kaggle username that will own the private uploads')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/responses/kaggle_package')
    args = parser.parse_args()
    prepare(args.output, args.code_revision, args.owner)


if __name__ == '__main__':
    main()
