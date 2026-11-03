"""Prepare a bounded, pinned real prompt pilot; no inference or synthetic prompts.

Default: SALAD linked attacks/defenses, WildJailbreak train prefix, XSTest held out.
Run: python python/prepare_prompts.py --output data/processed/prompts/pilot_v1
Use --inputs PATH for an offline replay of the saved input bundle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import csv

import requests
from huggingface_hub import get_token

from data_contracts import ROOT, VERSION, canonical, digest, normalize_messages, prompt_hash, prompt_id, assign_groups, manifest_identity, validate_bundle, verify_artifacts


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')


def write_lines(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(canonical(row)+'\n')


class BoundedReader(io.RawIOBase):
    def __init__(self, raw, maximum=32_000_000):
        self.raw, self.maximum, self.total = raw, maximum, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.raw.read(min(len(buffer), self.maximum-self.total+1))
        self.total += len(data)
        if self.total > self.maximum:
            raise ValueError('Source preview exceeded the 32 MB per-file read limit')
        buffer[:len(data)] = data
        return len(data)


def json_records(stream):
    """Incremental JSON array or JSONL/concatenated object decoder."""
    decoder = json.JSONDecoder()
    pending, started, array, eof = '', False, False, False
    while True:
        if not eof:
            chunk = stream.read(16384)
            eof = not chunk
            pending += chunk
        pending = pending.lstrip()
        if not started and pending:
            array = pending.startswith('[')
            pending = pending[1:] if array else pending
            started = True
        while pending.strip():
            pending = pending.lstrip(', \r\n\t') if array else pending.lstrip()
            if array and pending.startswith(']'):
                return
            if not pending:
                break
            try:
                value, end = decoder.raw_decode(pending)
            except json.JSONDecodeError:
                break
            if not isinstance(value, dict):
                raise ValueError('Source rows must be JSON objects')
            yield value
            pending = pending[end:]
        if eof:
            if pending.strip():
                raise ValueError('Truncated source JSON')
            return


def fetch_rows(url, count, fmt, authenticated=False):
    headers = {'User-Agent': 'llm-risk-topology-prompt-pilot/1'}
    if authenticated and get_token():
        headers['Authorization'] = 'Bearer '+get_token()
    with requests.get(url, headers=headers, stream=True, timeout=(10, 45)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        bounded = BoundedReader(response.raw)
        text = io.TextIOWrapper(io.BufferedReader(bounded), encoding='utf-8-sig', newline='')
        csv.field_size_limit(2_000_000)
        iterator = json_records(text) if fmt == 'json' else csv.DictReader(text, delimiter='\t' if fmt == 'tsv' else ',')
        rows = []
        for index, row in enumerate(iterator):
            rows.append({'row_index': index, 'row': row})
            if len(rows) >= count:
                break
        return rows, bounded.total


def collect_inputs(attack_rows=600, defense_rows=100, wild_rows=300):
    inventory = json.loads((ROOT/'manifests/dataset_schema_inventory.json').read_text(encoding='utf-8'))
    catalog = {s['source']: s for s in inventory['sources']}
    specs = [('OpenSafetyLab/Salad-Data', 'attack_enhanced_set', 'train', 'attack_enhanced_set.json', attack_rows, 'json'),
             ('OpenSafetyLab/Salad-Data', 'defense_enhanced_set', 'train', 'defense_enhanced_set.json', defense_rows, 'json'),
             ('allenai/wildjailbreak', 'train', 'train', 'train/train.tsv', wild_rows, 'tsv'),
             ('paul-rottger/xstest', 'default', 'unspecified', 'xstest_prompts.csv', 450, 'csv')]
    result = []
    for source, config, split, path, count, fmt in specs:
        if count < 1:
            continue
        meta = catalog[source]
        revision = meta['revision']
        hf = meta['provider'] == 'huggingface'
        url = f'https://huggingface.co/datasets/{source}/resolve/{revision}/{path}' if hf else f'https://raw.githubusercontent.com/{source}/{revision}/{path}'
        rows, size = fetch_rows(url, count, fmt, hf)
        print(f'{source}/{config}: captured {len(rows)} source rows ({size} streamed bytes)', flush=True)
        license_value = meta.get('license')
        result.append({'source_id': source, 'revision': revision, 'provider': meta['provider'], 'config': config,
            'split': split, 'path': path, 'url': url, 'license': license_value if isinstance(license_value, str) else None,
            'rows': rows, 'streamed_bytes': size, 'requested_rows': count})
    return result


def make_record(source, record_id, text, variant, anchors, labels, parent=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty or non-string prompt')
    messages = normalize_messages([{'role': 'user', 'content': text}])
    content_hash = prompt_hash(messages)
    return {'schema_version': VERSION,
        'prompt_id': prompt_id(source['source_id'], source['revision'], source['config'], source['split'], record_id, variant, messages),
        'source_id': source['source_id'], 'source_revision': source['revision'], 'source_config': source['config'],
        'source_split': source['split'], 'source_record_id': record_id, 'messages': messages, 'task_type': 'assistant',
        'variant_type': variant, 'parent_prompt_id': parent, 'group_anchors': sorted(set(anchors+['messages:'+content_hash])),
        'prompt_sha256': content_hash, 'source_labels': labels}


def adapt(inputs):
    records, pairs, rejected = [], [], []
    for source in inputs:
        for captured in source['rows']:
            row, index = captured['row'], captured['row_index']
            try:
                if source['source_id'] == 'OpenSafetyLab/Salad-Data':
                    defense = source['config'] == 'defense_enhanced_set'
                    variant_field = 'daugq' if defense else 'augq'
                    variant = 'defense' if defense else 'adversarial'
                    # Keep declared qid families together conservatively, even when base wording differs.
                    base = row['baseq']
                    anchor = 'original_messages:'+prompt_hash(normalize_messages([{'role': 'user', 'content': base}]))
                    anchors = [anchor]
                    if row.get('qid') is not None:
                        anchors.append(source['source_id']+':'+source['revision']+':qid:'+str(row['qid']))
                    labels = {k: v for k, v in row.items() if k in ['qid', 'aid', 'did', 'method', 'dmethod', '1-category', '2-category', '3-category']}
                    original = make_record(source, f'row:{index}:original', base, 'original', anchors, labels)
                    changed = make_record(source, f'row:{index}:variant', row[variant_field], variant, anchors, labels, original['prompt_id'])
                    records.extend([original, changed])
                    pairs.append({'source_id': source['source_id'], 'source_config': source['config'], 'source_row_index': index,
                                  'original_prompt_id': original['prompt_id'], 'variant_prompt_id': changed['prompt_id'], 'variant_type': variant})
                elif source['source_id'] == 'allenai/wildjailbreak':
                    base = row['vanilla']
                    adversarial = row.get('adversarial') or ''
                    if not isinstance(adversarial, str) or row['data_type'] not in ['vanilla_harmful', 'vanilla_benign', 'adversarial_harmful', 'adversarial_benign']:
                        raise ValueError('Invalid WildJailbreak prompt fields or data type')
                    anchor = 'original_messages:'+prompt_hash(normalize_messages([{'role': 'user', 'content': base}]))
                    labels = {'data_type': row['data_type'], 'prompt_intent': 'unharmful' if row['data_type'].endswith('benign') else 'harmful'}
                    original = make_record(source, f'row:{index}:original', base, 'original', [anchor], labels)
                    records.append(original)
                    if adversarial.strip():
                        changed = make_record(source, f'row:{index}:variant', adversarial, 'adversarial', [anchor], labels, original['prompt_id'])
                        records.append(changed)
                        pairs.append({'source_id': source['source_id'], 'source_config': source['config'], 'source_row_index': index,
                                      'original_prompt_id': original['prompt_id'], 'variant_prompt_id': changed['prompt_id'], 'variant_type': 'adversarial'})
                elif source['source_id'] == 'paul-rottger/xstest':
                    records.append(make_record(source, str(row['id']), row['prompt'], 'unpaired', [],
                                               {k: row[k] for k in ['label', 'type', 'focus', 'note'] if k in row}))
                else:
                    raise ValueError('No adapter for source')
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append({'source_id': source['source_id'], 'config': source['config'], 'row_index': index, 'reason': str(exc)})
    return records, pairs, rejected


def prepare(inputs, seed=42, maximum=2500):
    records, source_pairs, rejected = adapt(inputs)
    # Group BEFORE collapsing duplicate inputs so every original/variant link survives.
    raw_groups = assign_groups({p['prompt_id']: p['group_anchors'] for p in records})
    by_hash = defaultdict(list)
    for p in records:
        by_hash[p['prompt_sha256']].append(p)
    redirects, canonical_records, aliases = {}, [], []
    priority = {'original': 0, 'unpaired': 1, 'adversarial': 2, 'defense': 2}
    for same_input in by_hash.values():
        chosen = min(same_input, key=lambda p: (priority[p['variant_type']], p['prompt_id']))
        copy = {**chosen, 'group_anchors': sorted({a for p in same_input for a in p['group_anchors']}),
                'prompt_group_id': raw_groups[chosen['prompt_id']]}
        canonical_records.append(copy)
        for p in same_input:
            redirects[p['prompt_id']] = chosen['prompt_id']
            aliases.append({'occurrence_prompt_id': p['prompt_id'], 'canonical_prompt_id': chosen['prompt_id'],
                'source_id': p['source_id'], 'source_revision': p['source_revision'], 'source_config': p['source_config'],
                'source_split': p['source_split'], 'source_record_id': p['source_record_id'], 'variant_type': p['variant_type'],
                'source_labels': p['source_labels']})
    for p in canonical_records:
        if p['parent_prompt_id']:
            parent = redirects[p['parent_prompt_id']]
            p['parent_prompt_id'] = None if parent == p['prompt_id'] else parent
    grouped = defaultdict(list)
    for p in canonical_records:
        grouped[p['prompt_group_id']].append(p)
    # XSTest is a reserved diagnostic, never method-development data; promote overlaps too.
    reserved = {raw_groups[p['prompt_id']] for p in records if p['source_id'] == 'paul-rottger/xstest'}
    selected, skipped = [], []
    for group in sorted(grouped, key=lambda g: (g not in reserved, digest([seed, 'selection', g]))):
        if len(selected)+len(grouped[group]) <= maximum:
            selected.extend(grouped[group])
        else:
            skipped.append(group)
    ids = {p['prompt_id'] for p in selected}
    selected_groups = {p['prompt_group_id'] for p in selected}
    split_map = {}
    for group in selected_groups:
        fraction = int(digest([seed, 'split', group])[:16], 16)/2**64
        split_map[group] = 'test' if group in reserved else ('development' if fraction < .7 else 'validation' if fraction < .85 else 'test')
    pairs, unchanged = [], []
    for pair in source_pairs:
        rewritten = {**pair, 'original_prompt_id': redirects[pair['original_prompt_id']], 'variant_prompt_id': redirects[pair['variant_prompt_id']]}
        if rewritten['original_prompt_id'] not in ids or rewritten['variant_prompt_id'] not in ids:
            continue
        if rewritten['original_prompt_id'] == rewritten['variant_prompt_id']:
            unchanged.append(rewritten)
        else:
            pairs.append(rewritten)
    aliases = [a for a in aliases if a['canonical_prompt_id'] in ids]
    selected.sort(key=lambda p: p['prompt_id'])
    pairs.sort(key=canonical)
    aliases.sort(key=lambda a: a['occurrence_prompt_id'])
    validate_bundle({'prompts': selected})
    # Independent assertions over the selected pair table, not only parent pointers.
    lookup = {p['prompt_id']: p for p in selected}
    for pair in pairs:
        a, b = [lookup[pair[key]] for key in ['original_prompt_id', 'variant_prompt_id']]
        assert a['prompt_group_id'] == b['prompt_group_id']
    assert len({p['prompt_sha256'] for p in selected}) == len(selected)
    quality = {'input_source_rows': sum(len(s['rows']) for s in inputs), 'candidate_occurrences': len(records),
        'unique_candidate_prompts': len(canonical_records), 'duplicate_occurrences_removed': len(records)-len(canonical_records),
        'selected_unique_prompts': len(selected), 'selected_groups': len(selected_groups), 'oversize_or_over_budget_groups_skipped': len(skipped),
        'linked_pair_occurrences': len(pairs), 'unique_linked_pairs': len({(p['original_prompt_id'],p['variant_prompt_id'],p['variant_type']) for p in pairs}),
        'unchanged_pair_occurrences': len(unchanged), 'rejected_rows': rejected,
        'split_prompt_counts': dict(Counter(split_map[p['prompt_group_id']] for p in selected)),
        'split_group_counts': dict(Counter(split_map.values())), 'variant_counts': dict(Counter(p['variant_type'] for p in selected)),
        'canonical_source_counts': dict(Counter(p['source_id'] for p in selected)),
        'source_occurrence_counts': dict(Counter(a['source_id'] for a in aliases)),
        'cross_source_duplicate_inputs': sum(len({p['source_id'] for p in rows}) > 1 for rows in by_hash.values()),
        'reserved_xstest_groups': len(selected_groups & reserved),
        'pair_split_counts': dict(Counter(split_map[lookup[p['original_prompt_id']]['prompt_group_id']] for p in pairs)),
        'checks': {'unique_inputs': True, 'parent_and_pair_groups_match': True, 'one_split_per_group': True, 'contract_validation': True}}
    quality['source_field_audit'] = []
    qid_texts = defaultdict(set)
    for source in inputs:
        rows = [r['row'] for r in source['rows']]
        if source['source_id'] == 'OpenSafetyLab/Salad-Data':
            for row in rows:
                if row.get('qid') is not None and isinstance(row.get('baseq'), str):
                    qid_texts[str(row['qid'])].add(prompt_hash(normalize_messages([{'role': 'user', 'content': row['baseq']}])) )
        fields = sorted({field for row in rows for field in row})
        quality['source_field_audit'].append({'source_id': source['source_id'], 'config': source['config'], 'observed_rows': len(rows),
            'fields': {field: {'missing': sum(field not in row for row in rows), 'null': sum(field in row and row[field] is None for row in rows),
                'empty_string': sum(isinstance(row.get(field), str) and not row[field].strip() for row in rows)} for field in fields}})
    quality['salad_qids_with_multiple_base_texts'] = sum(len(texts) > 1 for texts in qid_texts.values())
    return selected, pairs, aliases, split_map, quality, unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, help='Offline replay from a previously saved inputs.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/prompts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--max-prompts', type=int, default=2500)
    parser.add_argument('--min-prompts', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attack-rows', type=int, default=600)
    parser.add_argument('--defense-rows', type=int, default=100)
    parser.add_argument('--wild-rows', type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.min_prompts <= args.max_prompts <= 5000 or args.seed < 0:
        parser.error('Require 1 <= min-prompts <= max-prompts <= 5000 and nonnegative seed')
    if any(not 0 <= n <= 5000 for n in [args.attack_rows, args.defense_rows, args.wild_rows]):
        parser.error('Source row limits must be 0..5000')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data/processed/prompts'):
        parser.error('Output must be a new directory under data/processed/prompts')
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(args.inputs.read_text(encoding='utf-8')) if args.inputs else collect_inputs(args.attack_rows, args.defense_rows, args.wild_rows)
    # Saved rows are a bounded extraction from immutable pinned source files, not whole files.
    write_json(output/'inputs.json', inputs)
    records, pairs, aliases, splits, quality, unchanged = prepare(inputs, args.seed, args.max_prompts)
    write_json(output/'quality.json', quality)
    if len(records) < args.min_prompts:
        raise ValueError('Not enough unique prompts; quality evidence retained, no frozen manifest created')
    for name, rows in [('prompts',records), ('pairs',pairs), ('occurrences',aliases), ('unchanged_pairs',unchanged)]:
        write_lines(output/(name+'.jsonl'), rows)
    source_entries = {}
    for source in inputs:
        source_entries[source['source_id']] = {'source_id': source['source_id'], 'provider': source['provider'],
            'repository': source['source_id'], 'revision': source['revision'],
            'url': ('https://huggingface.co/datasets/' if source['provider'] == 'huggingface' else 'https://github.com/')+source['source_id'], 'license': source['license'],
            'access_status': 'verified', 'stretch': False}
    source_manifest = {'schema_version': VERSION, 'status': 'frozen', 'sources': sorted(source_entries.values(),key=lambda s:s['source_id'])}
    source_manifest['manifest_id'] = manifest_identity('source_manifests',source_manifest)
    artifact = output/'prompts.jsonl'
    manifest = {'schema_version': VERSION, 'status': 'frozen', 'source_manifest_id': source_manifest['manifest_id'],
        'grouping_version': VERSION, 'selection': {'seed': args.seed, 'description': 'Bounded pinned source prefixes; exact message deduplication and transitive original/variant grouping. Whole-group hash-ranked selection; 70/15/15 hash splits outside reserved XSTest groups, which are all test. See checksummed preparation provenance for limits and pair/occurrence artifacts.'},
        'artifacts': [{'path': artifact.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'row_count': len(records)}],
        'prompt_ids': [p['prompt_id'] for p in records], 'group_splits': dict(sorted(splits.items()))}
    manifest['manifest_id'] = manifest_identity('prompt_manifests',manifest)
    bundle = {'source_manifests': [source_manifest], 'prompt_manifests': [manifest], 'prompts': records}
    validate_bundle(bundle)
    verify_artifacts(bundle)
    write_json(output/'experiment.bundle.json', bundle)
    provenance = {'schema_version': VERSION, 'prompt_manifest_id': manifest['manifest_id'], 'created_at': datetime.now(timezone.utc).isoformat(),
        'command_settings': {'seed': args.seed, 'maximum': args.max_prompts, 'minimum': args.min_prompts},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts': [{'path': (output/name).relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest()}
                      for name in ['inputs.json','prompts.jsonl','pairs.jsonl','occurrences.jsonl','unchanged_pairs.jsonl','quality.json','experiment.bundle.json']],
        'limitations': ['First source rows are not a random or representative sample.', 'Only exact normalized-message duplicates are collapsed; near duplicates remain unaudited.',
            'WildJailbreak prefix may contain only vanilla harmful prompts; paired attacks in this pilot primarily come from SALAD.',
            'Benign prompts may be concentrated in reserved XSTest: audit intent/source balance before treating this engineering pilot as a balanced behavioral study.',
            'Source labels describe prompt provenance/intent and historical annotations, not fresh output behavior.', 'XSTest is held out from method development; public benchmark exposure during model training remains unknown.']}
    write_json(output/'preparation.json', provenance)
    run_dir = ROOT/'manifests/prompt_runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    # Shared record deliberately omits prompt text, prompt IDs, and source-row labels.
    write_json(run_dir/(manifest['manifest_id'][8:24]+'.json'), {'prompt_manifest_id': manifest['manifest_id'],
        'local_output': output.relative_to(ROOT).as_posix(), 'preparation_sha256': hashlib.sha256((output/'preparation.json').read_bytes()).hexdigest(),
        'sources': source_manifest['sources'], 'quality': quality, 'limitations': provenance['limitations']})
    print(json.dumps(quality,indent=2),flush=True)
    print('Frozen prompt bundle:', output/'experiment.bundle.json', flush=True)


if __name__ == '__main__':
    main()
