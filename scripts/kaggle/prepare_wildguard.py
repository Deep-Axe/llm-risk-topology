"""Stage a private WildGuard-only notebook; upload with --no-run, then launch in the web editor."""
import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def prepare(output, revision, owner):
    if re.fullmatch('[0-9a-f]{40}', revision) is None:
        raise ValueError('Pass the full committed labeling code SHA')
    subprocess.run(['git', 'cat-file', '-e', revision + ':scripts/kaggle/wildguard_runner.py'],
                   cwd=ROOT, check=True)
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
                'This notebook only labels saved responses and exports the audit sample.\n']},
            {'cell_type': 'code', 'id': 'wildguard-only', 'metadata': {}, 'execution_count': None,
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
    print('Prepared WildGuard-only notebook:', output)
    print('Upload with kaggle kernels push -p "' + str(output.resolve()) + '" --no-run')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.code_revision, args.owner)


if __name__ == '__main__':
    main()
