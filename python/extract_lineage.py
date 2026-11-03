"""Dependency-free Pig streaming parser for captured card JSON (stdin TSV).

Output: status, child, parent, relation, evidence_field, evidence_value, reason.
Unspecified relations remain other. This extracts declarations, not inferred edges.
"""
import json
import re
import sys

HUB_ID = re.compile(r'^[^/\s]+/[^/\s]+$')
RELATIONS = {'finetune': 'fine_tune', 'fine-tune': 'fine_tune', 'fine_tune': 'fine_tune',
             'adapter': 'adapter', 'merge': 'merge', 'quantized': 'quantization', 'quantization': 'quantization'}


def extract(child, card):
    base = card.get('base_model')
    relation = RELATIONS.get(str(card.get('base_model_relation', '')).lower(), 'other')
    if base is None or base == [] or base == '':
        return [('no_lineage', child, '-', 'other', 'base_model', '-', 'missing_base_model')]
    parents = base if isinstance(base, list) else [base]
    evidence = json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    result = []
    for parent in parents:
        # Do not guess parents from arbitrary nested dictionaries or URLs.
        valid = isinstance(parent, str) and HUB_ID.fullmatch(parent) and parent != child
        result.append(('valid' if valid else 'rejected', child, parent if isinstance(parent, str) else '-',
                       relation, 'base_model', evidence, '-' if valid else 'invalid_or_self_parent'))
    return result


def main():
    for line in sys.stdin:
        child, revision, payload = line.rstrip('\n').split('\t', 2)
        card = json.loads(payload)
        for record in extract(child, card):
            print('\t'.join(str(v).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ') for v in record))


if __name__ == '__main__':
    main()
