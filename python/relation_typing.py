"""Infer lineage relation types from frozen cards, Hub tags and captured file lists.

Run: python python/relation_typing.py --snapshot data/raw/metadata/dev50k_v1 --output NEW_FOLDER
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from data_contracts import VERSION, canonical, digest, edge_id, validate_bundle
from extract_lineage import extract


LEVELS = {'low': 1, 'medium': 2, 'high': 3}
TAG_TYPES = {'finetune': 'fine_tune', 'quantized': 'quantization', 'adapter': 'adapter', 'merge': 'merge'}
QUANT_TAGS = {'gguf', 'ggml', 'quantized', 'quantization', 'awq', 'gptq', 'exl2', 'exllamav2', 'hqq'}
ADAPTER_TAGS = {'lora', 'qlora', 'peft', 'adapter', 'adapters'}
MERGE_TAGS = {'merge', 'mergekit', 'merged'}


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8', newline='\n')


def signals(child, parent, card, metadata):
    evidence = []

    def add(kind, rule, confidence, field, value):
        evidence.append({'relation': kind, 'rule': rule, 'confidence': confidence,
                         'field': field, 'value': value})

    tags = metadata.get('tags') or []
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            parts = tag.split(':', 2)
            if len(parts) == 3 and parts[0] == 'base_model' and parts[2] == parent:
                kind = TAG_TYPES.get(parts[1])
                if kind:
                    confidence = 'medium' if kind == 'fine_tune' else 'high'
                    add(kind, 'api_parent_' + parts[1], confidence, 'api.tags', tag)

    for field, values in [('card.tags', card.get('tags')), ('api.tags', metadata.get('tags'))]:
        if not isinstance(values, list):
            continue
        for tag in sorted({t.lower() for t in values if isinstance(t, str)}):
            if tag in QUANT_TAGS:
                add('quantization', 'quantization_tag', 'medium', field, tag)
            elif tag in ADAPTER_TAGS:
                add('adapter', 'adapter_tag', 'low', field, tag)
            elif tag in MERGE_TAGS:
                add('merge', 'merge_tag', 'medium', field, tag)
    for field, library in [('card.library_name', card.get('library_name')),
                           ('api.library_name', metadata.get('library_name'))]:
        if not isinstance(library, str):
            continue
        if library.lower() in {'gguf', 'ggml', 'autoawq', 'auto-gptq', 'exllamav2'}:
            add('quantization', 'quantization_library', 'high', field, library)
        elif library.lower() == 'peft':
            add('adapter', 'peft_library', 'medium', field, library)
        elif library.lower() == 'mergekit':
            add('merge', 'mergekit_library', 'high', field, library)
    files = [s.get('rfilename', '') for s in (metadata.get('siblings') or []) if isinstance(s, dict)]
    for name in sorted(files):
        if name.lower().endswith(('.gguf', '.ggml', '.exl2')):
            add('quantization', 'quantized_artifact', 'high', 'api.siblings.rfilename', name)
        elif name.rsplit('/', 1)[-1] == 'adapter_config.json':
            add('adapter', 'adapter_config_file', 'high', 'api.siblings.rfilename', name)
        elif name.rsplit('/', 1)[-1] in {'mergekit_config.yml', 'mergekit_config.yaml'}:
            add('merge', 'mergekit_config_file', 'high', 'api.siblings.rfilename', name)
    for field in ['quantization_config', 'quantization_method', 'quantization_format', 'quantized_by']:
        if card.get(field) not in [None, '', False, {}, []]:
            add('quantization', 'card_quantization_field', 'medium', 'card.' + field, card[field])
    if card.get('adapter_config'):
        add('adapter', 'card_adapter_config', 'high', 'card.adapter_config', card['adapter_config'])
    name = child.split('/', 1)[-1]
    patterns = [
        ('quantization', r'(?:^|[-_.])(?:gguf|ggml|awq|gptq|exl2|hqq|int[248]|[248]bit)(?:$|[-_.])'),
        ('quantization', r'(?:^|[-_.])q[2-8]_(?:k|0|1)(?:$|[-_.])'),
        ('adapter', r'(?:^|[-_.])(?:lora|qlora|adapter|peft)(?:$|[-_.])'),
        ('merge', r'(?:^|[-_.])(?:merge|merged|mergekit)(?:$|[-_.])'),
    ]
    for kind, pattern in patterns:
        if re.search(pattern, name, re.IGNORECASE):
            add(kind, 'repository_name', 'low', 'repository_name', name)
    return sorted(evidence, key=lambda row: (-LEVELS[row['confidence']], row['rule'],
                                            row['field'], canonical(row['value'])))


def infer(child, parent, card, metadata):
    candidates = signals(child, parent, card, metadata)
    specialist = [row for row in candidates if row['relation'] != 'fine_tune']
    strong = {row['relation'] for row in specialist if LEVELS[row['confidence']] >= 2}
    if len(strong) > 1:
        return None, 'conflicting_transformations', 'unresolved', candidates
    if strong:
        chosen = next(row for row in specialist if row['relation'] in strong)
    elif specialist:
        kinds = {row['relation'] for row in specialist}
        if len(kinds) > 1:
            return None, 'conflicting_weak_signals', 'unresolved', candidates
        chosen = specialist[0]
    elif candidates:
        chosen = candidates[0]
    else:
        return None, 'no_evidence', 'unresolved', []
    return chosen['relation'], chosen['rule'], chosen['confidence'], candidates


def type_snapshot(snapshot, output, audit_size=50, seed=42):
    manifest = snapshot / 'metadata_snapshot.json'
    frozen = json.loads(manifest.read_text(encoding='utf-8'))
    if frozen['status'] != 'frozen':
        raise ValueError('Relation typing requires a frozen snapshot')
    for artifact in frozen['artifacts']:
        if checksum(snapshot / artifact['path']) != artifact['sha256']:
            raise ValueError('Frozen snapshot checksum changed')
    snapshot_id = checksum(manifest)
    metadata = {}
    with gzip.open(snapshot / 'models.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            metadata[row['id']] = {key: row.get(key) for key in ['sha', 'tags', 'library_name', 'siblings']}
    annotations, inferred_edges = [], []
    seen = set()
    csv.field_size_limit(2_000_000)
    with (snapshot / 'models.tsv').open(encoding='utf-8', newline='') as stream:
        for child, revision, payload in csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
            card = json.loads(payload)
            raw = metadata[child]
            if (raw['sha'] or '-') != revision:
                raise ValueError('Card and API revisions differ')
            for status, _, parent, declared, _, _, _ in extract(child, card):
                if status != 'valid':
                    continue
                declared_id = edge_id(snapshot_id, child, parent, declared)
                if declared_id in seen:
                    continue
                seen.add(declared_id)
                inferred, rule, confidence, evidence = infer(child, parent, card, raw) \
                    if declared == 'other' else (None, 'declared_preserved', 'declared', [])
                inferred_id = edge_id(snapshot_id, child, parent, inferred) if inferred else None
                row = {'declared_edge_id': declared_id, 'snapshot_id': snapshot_id,
                       'child_model_id': child, 'parent_model_id': parent,
                       'child_revision': revision, 'declared_relation': declared,
                       'inferred_relation': inferred, 'inferred_edge_id': inferred_id,
                       'evidence_rule': rule, 'confidence': confidence, 'evidence': evidence}
                annotations.append(row)
                if inferred:
                    selected = next(e for e in evidence if e['relation'] == inferred and e['rule'] == rule)
                    inferred_edges.append({'schema_version': VERSION, 'edge_id': inferred_id,
                        'snapshot_id': snapshot_id, 'child_model_id': child, 'parent_model_id': parent,
                        'relation': inferred, 'lineage_status': 'inferred',
                        'evidence': [{'url': f'https://huggingface.co/{child}/tree/{revision}',
                                      'revision': None if revision == '-' else revision,
                                      'field': selected['field'], 'value': canonical(selected['value'])}]})
    validate_bundle({'metadata_edges': inferred_edges})
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('annotations', annotations), ('inferred_edges', inferred_edges)]:
        with (output / (name + '.jsonl')).open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
    candidates = [row for row in annotations if row['inferred_relation']]
    sample = sorted(candidates, key=lambda row: digest([seed, 'relation-audit', row['declared_edge_id']]))
    write_json(output / 'audit_sample.json', sample[:audit_size])
    counts = Counter((row['inferred_relation'], row['evidence_rule'], row['confidence']) for row in candidates)
    summary = {'snapshot_id': snapshot_id, 'declared_edges': len(annotations),
               'declared_other_edges': sum(row['declared_relation'] == 'other' for row in annotations),
               'inferred_edges': len(inferred_edges),
               'unresolved_other': sum(row['declared_relation'] == 'other' and not row['inferred_relation']
                                       for row in annotations),
               'by_rule': [{'relation': key[0], 'rule': key[1], 'confidence': key[2], 'edges': count}
                           for key, count in sorted(counts.items())],
               'unresolved_rules': dict(Counter(row['evidence_rule'] for row in annotations
                                                if row['declared_relation'] == 'other'
                                                and not row['inferred_relation'])),
               'audit_seed': seed, 'audit_sample_size': min(audit_size, len(candidates)),
               'script_sha256': checksum(Path(__file__)),
               'inputs': {item['path']: item['sha256'] for item in frozen['artifacts']},
               'artifacts': {path.name: checksum(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.audit_size < 1 or args.seed < 0:
        parser.error('Use a positive audit size and nonnegative seed')
    type_snapshot(args.snapshot, args.output, args.audit_size, args.seed)


if __name__ == '__main__':
    main()
