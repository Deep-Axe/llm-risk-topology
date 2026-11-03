"""Stage the private six-model expansion and eight-model WildGuard notebook without execution."""
import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def prepare(output, revision, owner, models=None, splits=None):
    if re.fullmatch('[0-9a-f]{40}', revision) is None:
        raise ValueError('Pass the full committed labeling code SHA')
    subprocess.run(['git', 'cat-file', '-e', revision + ':scripts/kaggle/wildguard_runner.py'],
                   cwd=ROOT, check=True)
    registered = [m['model_id'] for m in json.loads((ROOT / 'manifests/models/scale6.bundle.json').read_text())
                  ['model_manifests'][0]['models']]
    config = {'models': registered if models is None else models,
              'splits': ['validation'] if splits is None else splits,
              'model_manifest': 'manifests/models/scale6.bundle.json'}
    if (config['splits'] != ['validation'] or len(config['models']) != len(set(config['models'])) or
            not set(config['models']).issubset(registered)):
        raise ValueError('Use registered models and the validation split')
    output.mkdir(parents=True, exist_ok=False)
    runner = (ROOT / 'scripts/kaggle/wildguard_runner.py').read_text(encoding='utf-8').replace(
        "CODE_REVISION = 'SET_BY_PREPARE'", f"CODE_REVISION = '{revision}'", 1)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [
            {'cell_type': 'markdown', 'id': 'launch-instructions', 'metadata': {}, 'source': [
                'Run this notebook after version 7 of the two-model pilot finishes generation.\n',
                'In Add-ons → Secrets, enable the existing HF_TOKEN (HF_Token is also accepted).\n',
                'Select GPU T4 ×2 and Internet on, then Save Version → Save & Run All.\n',
                'Notebook input: deepamahuja/llm-risk-topology-two-model-pilot (v7 outputs).\n',
                'Refresh the Notebook input to version 7 after its outputs are published.\n',
                'Reuses Mistral and Zephyr; generates six validation-only assistants, then labels all eight.\n',
                'Greedy decoding, seed 42, fp16, batch 16, context 4096, output limit 512 are fixed.\n',
                'A checkpoint archive and per-model reports are saved after every completed model.\n']},
            {'cell_type': 'code', 'id': 'scale-parameters', 'metadata': {'tags': ['parameters']},
             'execution_count': None, 'outputs': [], 'source': [
                 '# Default: all six extra models, validation only. No test responses.\n',
                 'RUN_CONFIG = ' + repr(config) + '\n']},
            {'cell_type': 'code', 'id': 'generate-and-label', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': runner.splitlines(keepends=True)}]}
    metadata = {'id': owner + '/llm-risk-topology-wildguard-v7',
                'title': 'LLM Risk Topology WildGuard v7', 'code_file': 'wildguard.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [owner + '/llm-risk-topology-pilot-v5'],
                'kernel_sources': [owner + '/llm-risk-topology-two-model-pilot'],
                'competition_sources': []}
    # kernel_sources uses the documented owner/slug form. The runtime additionally
    # rejects an attached output with a different generation revision or prompt set.
    for name, value in [('wildguard.ipynb', notebook), ('kernel-metadata.json', metadata)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Prepared scale-up and WildGuard notebook:', output)
    print('Upload with kaggle kernels push -p "' + str(output.resolve()) + '" --no-run')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--splits', nargs='+', choices=['validation'], default=['validation'])
    args = parser.parse_args()
    prepare(args.output, args.code_revision, args.owner, args.models, args.splits)


if __name__ == '__main__':
    main()
