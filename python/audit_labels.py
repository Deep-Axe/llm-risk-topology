"""Join agent labels onto an audit export and flag rows for human review.

The labeling step sees only response_id, prompt, and response. WildGuard and
rule columns are copied back after the labels are attached.

Rows with no agent label are marked declined and sent to the human.

Usage:
    python python/audit_labels.py --input audit.csv --output audit_agent.csv
    python python/audit_labels.py --input audit.csv --labels agent.jsonl --output audit_agent.csv
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HIDDEN = (
    'wildguard_refusal', 'wildguard_response_harmful', 'wildguard_prompt_harmful',
    'wildguard_error', 'refusal_rules_refusal', 'refusal_disagreement',
)
VISIBLE = ('response_id', 'prompt', 'response')
RANDOM_AGREE = 20
AGENT_FIELDS = ('agent_refusal', 'agent_harmful', 'agent_note', 'needs_human')


def as_bool(value):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    return None


def blind_row(row):
    return {key: row.get(key, '') for key in VISIBLE}


def load_agent_labels(path):
    path = Path(path)
    labels = {}
    if path.suffix == '.jsonl':
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row['response_id']] = row
        return labels
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            labels[row['response_id']] = row
    return labels


def attach_labels(rows, labels_by_id):
    """Attach labels that were made from blind rows. Missing ids are declined."""
    for row in rows:
        visible = blind_row(row)
        label = labels_by_id.get(visible['response_id'])
        if not label:
            row['agent_refusal'] = ''
            row['agent_harmful'] = ''
            row['agent_note'] = 'declined'
            continue
        refusal = as_bool(label.get('agent_refusal'))
        harmful = as_bool(label.get('agent_harmful'))
        note = (label.get('agent_note') or '').strip()
        if refusal is None or harmful is None:
            row['agent_refusal'] = ''
            row['agent_harmful'] = ''
            row['agent_note'] = note or 'declined'
            continue
        row['agent_refusal'] = refusal
        row['agent_harmful'] = harmful
        row['agent_note'] = note
    return rows


def _refusal_conflict(agent, wildguard, rules):
    values = [value for value in (agent, wildguard, rules) if value is not None]
    if len(values) < 3:
        return True
    return len(set(values)) > 1


def flag_rows(rows, seed=20261104):
    """needs_human on a refusal or harm disagreement, plus 20 seeded agreements."""
    agree = []
    flags = []
    for index, row in enumerate(rows):
        if row.get('agent_note') == 'declined' or as_bool(row.get('agent_refusal')) is None:
            flags.append(True)
            continue
        refusal_conflict = _refusal_conflict(
            as_bool(row.get('agent_refusal')),
            as_bool(row.get('wildguard_refusal')),
            as_bool(row.get('refusal_rules_refusal')),
        )
        agent_harm = as_bool(row.get('agent_harmful'))
        wildguard_harm = as_bool(row.get('wildguard_response_harmful'))
        harm_conflict = (
            agent_harm is not None and wildguard_harm is not None and agent_harm != wildguard_harm
        )
        if refusal_conflict or harm_conflict:
            flags.append(True)
        else:
            flags.append(False)
            agree.append(index)
    rng = np.random.default_rng(seed)
    sample_size = min(RANDOM_AGREE, len(agree))
    chosen = set(rng.choice(agree, size=sample_size, replace=False).tolist()) if sample_size else set()
    for index in chosen:
        flags[index] = True
    for row, flag in zip(rows, flags):
        row['needs_human'] = flag
    return {
        'rows': len(rows),
        'needs_human': int(sum(flags)),
        'declined': int(sum(row.get('agent_note') == 'declined' for row in rows)),
        'agree_rows': len(agree),
        'agree_sample': sample_size,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else list(VISIBLE) + list(AGENT_FIELDS)
    for name in AGENT_FIELDS:
        if name not in fieldnames:
            fieldnames.append(name)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def label_audit(rows, labels_by_id, seed=20261104):
    attach_labels(rows, labels_by_id)
    return flag_rows(rows, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, help='CSV or JSONL of agent labels, keyed by response_id')
    parser.add_argument('--blind', type=Path,
                        help='Write prompt and response only, with label columns removed')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    with args.input.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.blind:
        write_csv(args.blind, [blind_row(row) for row in rows])
    labels = load_agent_labels(args.labels) if args.labels else {}
    stats = label_audit(rows, labels, args.seed)
    write_csv(args.output, rows)
    summary_path = args.summary or args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
