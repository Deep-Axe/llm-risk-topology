"""Apply refusal rules and publish the completed laptop run's numbers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from report_model import report_model

ROOT = Path(__file__).resolve().parents[2]


def finish(model, manifest, report):
    manifest_id = json.loads(manifest.read_text())['model_manifests'][0]['manifest_id']
    matches = []
    for path in (ROOT / 'data/processed/responses').glob('*/*/run.bundle.json'):
        data = json.loads(path.read_text())
        run = data['run_manifests'][0]
        if (run['runtime']['configuration']['model_id'] == model and
                run['model_manifest_id'] == manifest_id and
                run['runtime']['configuration']['selection']['splits'] == ['validation'] and
                run['generation']['max_new_tokens'] == 256):
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('Expected one matching laptop run')
    folder = matches[0]
    if not json.loads((folder / 'stats.json').read_text())['completed']:
        raise ValueError('Laptop generation is incomplete')
    subprocess.run([sys.executable, '-u', 'python/evaluate_responses.py', 'label', '--run', str(folder),
                    '--labeler', 'refusal_rules'], cwd=ROOT, check=True)
    summary = report_model(folder, report)
    note = report.parent.parent / 'project_notes/agent_rounds/round3/A_report.md'
    with note.open('a', encoding='utf-8') as stream:
        stream.write(f"\nLaptop model {model} completed: {summary['responses']} validation responses, "
                     f"{summary['failures']} failures, {summary['output_truncated']} output truncations, "
                     f"{summary['output_tokens_per_second']:.2f} output tokens/s. "
                     f"Rule labels and per-source/intent/variant rates are in report/pilot; "
                     f"validation rule-refusal flips: {summary['validation_refusal_flips']}. "
                     "This small-model stratum has no WildGuard harm labels yet.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args();finish(args.model, args.manifest, args.report)
