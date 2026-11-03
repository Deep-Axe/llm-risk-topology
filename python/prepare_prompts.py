"""Prepare frozen prompt pilots from pinned existing datasets.

Default: SALAD linked attacks/defenses, WildJailbreak train prefix, XSTest held out.
Run: python python/prepare_prompts.py --output data/processed/prompts/pilot_v1
Use --inputs PATH for an offline replay of the saved input bundle.
Use --extend-v4 PATH --wild-file PATH --max-prompts 6000 to build pilot_v5.
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
import heapq
import re
import unicodedata

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


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def overlap_normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text)).strip()


def word_grams(text):
    words = overlap_normalize(text).split()
    return {' '.join(words[i:i + 8]) for i in range(max(1, len(words) - 7))} if words else set()


def containment(left, right):
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def sample_wild_file(path, source, per_intent=950, seed=42):
    """Keep the lowest row-index hashes in each adversarial intent stratum."""
    heaps = {'adversarial_harmful': [], 'adversarial_benign': []}
    counts = Counter()
    invalid = Counter()
    csv.field_size_limit(2_000_000)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter='\t')):
            kind = row.get('data_type')
            counts[kind] += 1
            if kind not in heaps:
                continue
            if not row.get('vanilla', '').strip() or not row.get('adversarial', '').strip():
                invalid[kind] += 1
                continue
            rank = int(digest([seed, source['revision'], 'wild-row', index]), 16)
            entry = (-rank, index, {k: row[k] for k in ['vanilla', 'adversarial', 'data_type']})
            heap = heaps[kind]
            if len(heap) < per_intent:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if any(len(heap) != per_intent for heap in heaps.values()):
        raise ValueError('Insufficient adversarial rows for the requested intent strata')
    sampled = [{"row_index": index, "row": row} for heap in heaps.values() for _, index, row in heap]
    sampled.sort(key=lambda item: item['row_index'])
    audit = {'method': 'lowest SHA256(seed, revision, wild-row, zero-based row index) per intent',
             'seed': seed, 'per_intent': per_intent, 'file_sha256': file_sha256(path),
             'file_bytes': Path(path).stat().st_size, 'file_row_counts': dict(counts),
             'invalid_adversarial_rows': dict(invalid),
             'selected_row_indices': [row['row_index'] for row in sampled]}
    return {**source, 'rows': sampled, 'sampling': audit}, audit


def near_matches(records, pairs, threshold=.8):
    """Find shorter-text containment matches; identify shared variant wrappers separately."""
    lookup = {p['prompt_id']: p for p in records}
    cores = defaultdict(set)
    for pair in pairs:
        cores[pair['variant_prompt_id']].add(pair['original_prompt_id'])
    grams = {p['prompt_id']: word_grams(p['messages'][-1]['content']) for p in records}
    index = defaultdict(list)
    hits = []
    for row in sorted(records, key=lambda p: p['prompt_id']):
        pid = row['prompt_id']
        current = grams[pid]
        counts = Counter(other for gram in current for other in index[gram])
        for other, common in counts.items():
            denominator = min(len(current), len(grams[other]))
            score = common / denominator if denominator else 0.0
            if score < threshold:
                continue
            left_cores = cores.get(pid, {pid})
            right_cores = cores.get(other, {other})
            # Source-provided originals distinguish shared wrappers from repeated questions.
            both_variants = row['variant_type'] in {'adversarial', 'defense'}
            both_variants &= lookup[other]['variant_type'] in {'adversarial', 'defense'}
            core_score = max(containment(grams[a], grams[b]) for a in left_cores for b in right_cores)
            hits.append({'left': other, 'right': pid, 'containment': score,
                         'core_containment': core_score,
                         'template_only': bool(both_variants and core_score < threshold)})
        for gram in current:
            index[gram].append(pid)
    return hits


def extend_prepared(base, additions, seed=42, maximum=6000):
    """Extend a frozen bundle without changing its records, group IDs or splits."""
    old_records, old_pairs, old_aliases, old_splits, old_quality, old_unchanged = base
    records, pairs, rejected = adapt(additions)
    if rejected:
        raise ValueError('New source rows failed the adapter')
    frozen = {p['prompt_id']: p for p in old_records}
    all_records = {**frozen, **{p['prompt_id']: p for p in records}}
    all_pairs = old_pairs + pairs
    raw_groups = assign_groups({pid: p['group_anchors'] for pid, p in all_records.items()})
    parents = {group: group for group in set(raw_groups.values())}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)

    matches = near_matches(list(all_records.values()), all_pairs)
    existing_matches = []
    for hit in matches:
        if hit['template_only']:
            continue
        left, right = hit['left'], hit['right']
        if left in frozen and right in frozen:
            if frozen[left]['prompt_group_id'] != frozen[right]['prompt_group_id']:
                existing_matches.append(hit)
            continue
        union(raw_groups[left], raw_groups[right])

    components = defaultdict(list)
    for pid in all_records:
        components[find(raw_groups[pid])].append(pid)
    accepted = set(frozen)
    rejected_bridges = []
    merges = []
    for members in components.values():
        old_groups = {frozen[pid]['prompt_group_id'] for pid in members if pid in frozen}
        new_ids = set(members) - set(frozen)
        if len(old_groups) > 1:
            # A bridge cannot preserve both frozen group identities; keep its old endpoints only.
            rejected_bridges.append({'frozen_groups': sorted(old_groups),
                                     'excluded_occurrences': sorted(new_ids)})
            continue
        accepted.update(new_ids)
        if new_ids and (old_groups or len({raw_groups[pid] for pid in members}) > 1):
            merges.append({'frozen_groups': sorted(old_groups), 'new_occurrences': sorted(new_ids),
                           'input_groups': len({raw_groups[pid] for pid in members})})
        if old_groups:
            group = next(iter(old_groups))
            anchors = [a for pid in members if pid in frozen for a in frozen[pid]['group_anchors']]
            anchor = min(anchors)
        else:
            anchor = min(a for pid in members for a in all_records[pid]['group_anchors'])
        for pid in new_ids:
            all_records[pid] = {**all_records[pid], 'group_anchors': [anchor]}

    # Prefer the frozen canonical row even if a newly sampled occurrence sorts before it.
    by_hash = defaultdict(list)
    for pid in accepted:
        by_hash[all_records[pid]['prompt_sha256']].append(all_records[pid])
    chosen = []
    redirects = {}
    priority = {'original': 0, 'unpaired': 1, 'adversarial': 2, 'defense': 2}
    for duplicates in by_hash.values():
        row = min(duplicates, key=lambda p: (p['prompt_id'] not in frozen,
                  priority[p['variant_type']], p['prompt_id']))
        chosen.append(dict(row))
        for duplicate in duplicates:
            redirects[duplicate['prompt_id']] = row['prompt_id']
    groups = assign_groups({p['prompt_id']: p['group_anchors'] for p in chosen})
    for row in chosen:
        row['prompt_group_id'] = groups[row['prompt_id']]
        if row['parent_prompt_id']:
            row['parent_prompt_id'] = redirects[row['parent_prompt_id']]
            if row['parent_prompt_id'] == row['prompt_id']:
                row['parent_prompt_id'] = None
    chosen.sort(key=lambda row: row['prompt_id'])
    if len(chosen) > maximum:
        raise ValueError(f'{len(chosen)} prompts exceeds cap {maximum}; reduce --new-pairs-per-intent')
    splits = dict(old_splits)
    for group in set(groups.values()) - set(splits):
        fraction = int(digest([seed, 'split', group])[:16], 16) / 2**64
        splits[group] = 'development' if fraction < .7 else 'validation' if fraction < .85 else 'test'
    output_pairs, unchanged = list(old_pairs), list(old_unchanged)
    for pair in pairs:
        if not all(pair[key] in redirects for key in ['original_prompt_id', 'variant_prompt_id']):
            continue
        rewritten = {**pair, 'original_prompt_id': redirects[pair['original_prompt_id']],
                     'variant_prompt_id': redirects[pair['variant_prompt_id']]}
        same = rewritten['original_prompt_id'] == rewritten['variant_prompt_id']
        target = unchanged if same else output_pairs
        target.append(rewritten)
    aliases = list(old_aliases)
    for row in records:
        if row['prompt_id'] in redirects:
            aliases.append({'occurrence_prompt_id': row['prompt_id'],
                            'canonical_prompt_id': redirects[row['prompt_id']],
                            **{key: row[key] for key in ['source_id', 'source_revision', 'source_config',
                               'source_split', 'source_record_id', 'variant_type', 'source_labels']}})
    aliases = list({a['occurrence_prompt_id']: a for a in aliases}.values())
    aliases.sort(key=lambda row: row['occurrence_prompt_id'])
    output_pairs = list({canonical(pair): pair for pair in output_pairs}.values())
    output_pairs.sort(key=canonical)
    validate_bundle({'prompts': chosen})
    lookup = {p['prompt_id']: p for p in chosen}
    for pid, row in frozen.items():
        if lookup[pid] != row or splits[row['prompt_group_id']] != old_splits[row['prompt_group_id']]:
            raise ValueError('Frozen v4 record or split changed')
    for pair in output_pairs:
        left = lookup[pair['original_prompt_id']]['prompt_group_id']
        right = lookup[pair['variant_prompt_id']]['prompt_group_id']
        if left != right:
            raise ValueError('Pair crosses groups')
    final_matches = near_matches(chosen, output_pairs)
    unresolved = []
    for hit in final_matches:
        a, b = lookup[hit['left']], lookup[hit['right']]
        if hit['template_only'] or a['prompt_group_id'] == b['prompt_group_id']:
            continue
        unresolved.append(hit)
        if splits[a['prompt_group_id']] != splits[b['prompt_group_id']]:
            raise ValueError('Question near-duplicate crosses splits')
    audit = {'threshold': .8, 'normalization': 'NFKC, lowercase, punctuation removed, word 8-grams',
             'direction': 'shared grams / smaller gram-set size',
             'template_policy': 'Do not merge variant wrappers when source-provided original cores differ',
             'template_only_matches': sum(hit['template_only'] for hit in matches),
             'frozen_existing_matches': existing_matches,
             'merges': merges, 'excluded_bridge_components': rejected_bridges,
             'same_split_frozen_exceptions': unresolved,
             'final_cross_split_question_matches': 0}
    input_rows = old_quality['input_source_rows'] + sum(len(s['rows']) for s in additions)
    quality = {**old_quality, 'input_source_rows': input_rows,
               'selected_unique_prompts': len(chosen), 'selected_groups': len(splits),
               'linked_pair_occurrences': len(output_pairs),
               'unique_linked_pairs': len({(p['original_prompt_id'], p['variant_prompt_id'],
                                           p['variant_type'])
                                          for p in output_pairs}),
               'split_prompt_counts': dict(Counter(splits[p['prompt_group_id']] for p in chosen)),
               'split_group_counts': dict(Counter(splits.values())),
               'pair_split_counts': dict(Counter(splits[lookup[p['original_prompt_id']]['prompt_group_id']]
                                               for p in output_pairs)),
               'variant_counts': dict(Counter(p['variant_type'] for p in chosen)),
               'canonical_source_counts': dict(Counter(p['source_id'] for p in chosen)),
               'source_occurrence_counts': dict(Counter(p['source_id'] for p in aliases)),
               'base_prompt_count': len(frozen), 'base_records_unchanged': True,
               'near_duplicate_audit': audit, 'rejected_rows': rejected}
    return chosen, output_pairs, aliases, splits, quality, unchanged


def load_prepared(path):
    def rows(name):
        text = (path / (name + '.jsonl')).read_text(encoding='utf-8')
        return [json.loads(line) for line in text.splitlines()]
    bundle = json.loads((path / 'experiment.bundle.json').read_text(encoding='utf-8'))
    return (rows('prompts'), rows('pairs'), rows('occurrences'),
            bundle['prompt_manifests'][0]['group_splits'],
            json.loads((path / 'quality.json').read_text(encoding='utf-8')), rows('unchanged_pairs'))


def pair_table(records, pairs, aliases, splits):
    lookup = {p['prompt_id']: p for p in records}
    occurrence = {(a['source_id'], a['source_config'], a['source_record_id']): a for a in aliases}
    counts = Counter()
    groups = defaultdict(set)
    for pair in pairs:
        source = pair['source_id']
        group = lookup[pair['original_prompt_id']]['prompt_group_id']
        if source == 'OpenSafetyLab/Salad-Data':
            intent = 'harmful'
        else:
            row = occurrence[(source, pair['source_config'], f"row:{pair['source_row_index']}:original")]
            intent = 'benign' if row['source_labels']['prompt_intent'] == 'unharmful' else 'harmful'
        key = (splits[group], source, intent, pair['variant_type'])
        counts[key] += 1
        groups[key].add(group)
    return [{'split': key[0], 'source': key[1], 'intent': key[2], 'variant_type': key[3],
             'pairs': counts[key], 'pair_groups': len(groups[key])} for key in sorted(counts)]


def wildguard_overlap(records, splits):
    """Stream evaluator training prompts against an index of the pilot's grams."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    revision = 'd29c47f41c8b51348b5c8e8c81c039b3132b66d1'
    path = Path(hf_hub_download('allenai/wildguardmix', 'train/wildguard_train.parquet',
                               repo_type='dataset', revision=revision))
    query_index = defaultdict(list)
    exact_index = defaultdict(list)
    sizes = []
    for index, row in enumerate(records):
        text = overlap_normalize(row['messages'][-1]['content'])
        exact_index[text].append(index)
        grams = word_grams(text)
        sizes.append(len(grams))
        if len(grams) >= 3:
            for gram in grams:
                query_index[gram].append(index)
    best = [0.0] * len(records)
    exact = [False] * len(records)
    matched = [None] * len(records)
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['prompt']):
        for value in batch.column(0).to_pylist():
            text = overlap_normalize(str(value))
            for index in exact_index.get(text, ()):
                exact[index], best[index], matched[index] = True, 1.0, offset
            counts = Counter(index for gram in word_grams(text) for index in query_index.get(gram, ()))
            for index, common in counts.items():
                score = common / sizes[index]
                if score > best[index]:
                    best[index], matched[index] = score, offset
            offset += 1
    rows = [{'prompt_id': p['prompt_id'], 'prompt_group_id': p['prompt_group_id'],
             'source': p['source_id'], 'variant': p['variant_type'], 'split': splits[p['prompt_group_id']],
             'exact': exact[index], 'containment': best[index], 'overlap': best[index] >= .5,
             'wildguard_row_index': matched[index]} for index, p in enumerate(records)]
    groups = sorted({row['prompt_group_id'] for row in rows if row['overlap']})
    summary = {'revision': revision, 'file_sha256': file_sha256(path), 'training_rows': offset,
               'exact_prompts': sum(exact), 'overlap_prompts': sum(row['overlap'] for row in rows),
               'overlap_groups': len(groups), 'by_source_split': []}
    for source, split in sorted({(row['source'], row['split']) for row in rows}):
        subset = [row for row in rows if row['source'] == source and row['split'] == split]
        summary['by_source_split'].append({'source': source, 'split': split, 'prompts': len(subset),
            'exact': sum(row['exact'] for row in subset), 'overlap': sum(row['overlap'] for row in subset),
            'overlap_groups': len({row['prompt_group_id'] for row in subset if row['overlap']})})
    return rows, groups, summary


def estimate_tokens(records):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    model = 'mistralai/Mistral-7B-Instruct-v0.1'
    revision = HfApi().model_info(model).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    raw, chat = [], []
    for row in records:
        raw.append(len(tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False)))
        chat.append(len(tokenizer.apply_chat_template(row['messages'], tokenize=True,
                                                      add_generation_prompt=True)))
    return {'tokenizer': model, 'revision': revision, 'prompts': len(records),
            'raw_input_tokens_per_model': sum(raw), 'chat_input_tokens_per_model': sum(chat),
            'chat_input_tokens_two_models': 2 * sum(chat), 'chat_input_tokens_eight_models': 8 * sum(chat),
            'maximum_chat_tokens': max(chat), 'prompts_over_4096_chat_tokens': sum(n > 4096 for n in chat),
            'output_tokens_excluded': True, 'per_prompt_chat_tokens': dict(zip(
                [p['prompt_id'] for p in records], chat))}


def build_v5(args):
    output = Path(args.output.absolute())
    if output.exists():
        raise ValueError('Output already exists; frozen bundles cannot be overwritten')
    if not output.resolve().is_relative_to((ROOT / 'data/processed/prompts').resolve()):
        raise ValueError('Output must stay under data/processed/prompts')
    base_path = args.extend_v4
    base = load_prepared(base_path)
    base_inputs = json.loads((base_path / 'inputs.json').read_text(encoding='utf-8'))
    before = {path.name: file_sha256(path) for path in base_path.iterdir() if path.is_file()}
    if args.inputs:
        inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
        additions = [source for source in inputs if 'sampling' in source]
        if len(additions) != 1:
            raise ValueError('Extension replay needs one sampled WildJailbreak source')
        sampling = additions[0]['sampling']
        if file_sha256(args.wild_file) != sampling['file_sha256']:
            raise ValueError('WildJailbreak source file checksum changed')
    else:
        source = next(s for s in base_inputs if s['source_id'] == 'allenai/wildjailbreak')
        source = {key: value for key, value in source.items() if key not in
                  {'rows', 'streamed_bytes', 'requested_rows'}}
        sampled, sampling = sample_wild_file(args.wild_file, source, args.new_pairs_per_intent, args.seed)
        additions = [sampled]
        inputs = base_inputs + additions
    print('Full-file hash-ranked sample selected; grouping and near-duplicate audit...', flush=True)
    result = extend_prepared(base, additions, args.seed, args.max_prompts)
    records, pairs, aliases, splits, quality, unchanged = result
    table = pair_table(records, pairs, aliases, splits)
    lookup = {p['prompt_id']: p for p in records}
    test_groups = {lookup[pair['original_prompt_id']]['prompt_group_id'] for pair in pairs
                   if splits[lookup[pair['original_prompt_id']]['prompt_group_id']] == 'test'}
    benign_ids = {a['canonical_prompt_id'] for a in aliases
                  if a['source_labels'].get('prompt_intent') == 'unharmful'}
    benign_test = {lookup[pair['original_prompt_id']]['prompt_group_id'] for pair in pairs
                   if pair['original_prompt_id'] in benign_ids and
                   splits[lookup[pair['original_prompt_id']]['prompt_group_id']] == 'test'}
    if len(test_groups) < 300 or len(benign_test) < 100:
        raise ValueError(f'Insufficient test pair groups: {len(test_groups)} total, '
                         f'{len(benign_test)} benign')
    print(f'{len(records)} prompts, {len(test_groups)} test pair groups, '
          f'{len(benign_test)} benign; checking WildGuardMix overlap...', flush=True)
    overlap_rows, overlap_groups, overlap = wildguard_overlap(records, splits)
    overlap_set = set(overlap_groups)
    overlap['test_pair_groups'] = len(test_groups)
    overlap['clean_test_pair_groups'] = len(test_groups - overlap_set)
    overlap['benign_test_pair_groups'] = len(benign_test)
    overlap['clean_benign_test_pair_groups'] = len(benign_test - overlap_set)
    frozen_groups = set(base[3])
    overlap['new_test_pair_groups'] = len(test_groups - frozen_groups)
    overlap['clean_new_test_pair_groups'] = len(test_groups - frozen_groups - overlap_set)
    print('Overlap complete; estimating Mistral tokenizer input lengths...', flush=True)
    tokens = estimate_tokens(records)
    quality['pair_table'] = table
    quality['test_pair_groups'] = len(test_groups)
    quality['benign_test_pair_groups'] = len(benign_test)
    quality['wildguard_overlap'] = overlap
    quality['token_estimate'] = {
        key: value for key, value in tokens.items() if key != 'per_prompt_chat_tokens'}
    quality['checks']['v4_records_and_splits_unchanged'] = True
    quality['checks']['no_cross_split_question_near_duplicates'] = True
    quality['candidate_occurrences'] = base[4]['candidate_occurrences'] + 2 * len(additions[0]['rows'])
    quality['duplicate_occurrences_removed'] = quality['candidate_occurrences'] - len(records) - sum(
        len(component['excluded_occurrences'])
        for component in quality['near_duplicate_audit']['excluded_bridge_components'])
    quality['unique_candidate_prompts'] = len(records)
    quality['unchanged_pair_occurrences'] = len(unchanged)
    quality['source_field_audit'] = base[4]['source_field_audit'] + [{
        'source_id': additions[0]['source_id'], 'config': additions[0]['config'],
        'observed_rows': len(additions[0]['rows']), 'selection': 'full-file intent-stratified row hashes',
        'fields': {'vanilla': {'missing': 0, 'null': 0, 'empty_string': 0},
                   'adversarial': {'missing': 0, 'null': 0, 'empty_string': 0}}}]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', inputs)
    write_json(output / 'quality.json', quality)
    write_json(output / 'sampling.json', sampling)
    write_json(output / 'token_estimate.json', tokens)
    write_json(output / 'wildguard_overlap.json', overlap)
    write_lines(output / 'wildguard_overlap.jsonl', overlap_rows)
    for name, rows in [('prompts', records), ('pairs', pairs), ('occurrences', aliases),
                       ('unchanged_pairs', unchanged)]:
        write_lines(output / (name + '.jsonl'), rows)
    old_bundle = json.loads((base_path / 'experiment.bundle.json').read_text(encoding='utf-8'))
    source_manifest = old_bundle['source_manifests'][0]
    artifact = output / 'prompts.jsonl'
    manifest = {'schema_version': VERSION, 'status': 'frozen',
                'source_manifest_id': source_manifest['manifest_id'], 'grouping_version': VERSION,
                'selection': {'seed': args.seed, 'description':
                    'Immutable pilot_v4 plus full-file hash-ranked balanced WildJailbreak pairs. '
                    'Question near-duplicates share anchors; variant-only template matches stay separate. '
                    'Frozen groups keep records and splits; new groups use the existing 70/15/15 hash rule. '
                    'See checksummed quality and sampling artifacts for merge and overlap audits.'},
                'artifacts': [{'path': artifact.relative_to(ROOT).as_posix(),
                               'sha256': file_sha256(artifact), 'row_count': len(records)}],
                'prompt_ids': [p['prompt_id'] for p in records], 'group_splits': dict(sorted(splits.items()))}
    manifest['manifest_id'] = manifest_identity('prompt_manifests', manifest)
    bundle = {'source_manifests': [source_manifest], 'prompt_manifests': [manifest], 'prompts': records}
    validate_bundle(bundle)
    # Worktree data is a junction; the shared physical project owns the artifacts.
    verify_artifacts(bundle, root=(ROOT / 'data').resolve().parent)
    write_json(output / 'experiment.bundle.json', bundle)
    limitations = [
        'Inherited pilot_v4 rows are prefix samples; only new WildJailbreak rows are hash sampled.',
        'Template matches use source-original lexical containment; semantic paraphrases need review.',
        'Existing same-test-split XSTest near matches retain their frozen distinct group IDs.',
        'Public prompts may occur in assistant training data; evaluator overlap is separately recorded.',
        'Source intent is a provenance label, not a label on fresh output behavior.',
        'Token estimates exclude generated outputs and evaluator work.'
    ]
    provenance = {'schema_version': VERSION, 'prompt_manifest_id': manifest['manifest_id'],
                  'created_at': datetime.now(timezone.utc).isoformat(),
                  'command_settings': {'seed': args.seed, 'maximum': args.max_prompts,
                                       'new_pairs_per_intent': args.new_pairs_per_intent},
                  'script_sha256': file_sha256(Path(__file__)),
                  'base_manifest_id': old_bundle['prompt_manifests'][0]['manifest_id'],
                  'base_file_checksums': before, 'wild_file_sha256': sampling['file_sha256'],
                  'artifacts': [{'path': path.relative_to(ROOT).as_posix(), 'sha256': file_sha256(path)}
                                for path in sorted(output.iterdir()) if path.is_file()],
                  'limitations': limitations}
    after = {path.name: file_sha256(path) for path in base_path.iterdir() if path.is_file()}
    if before != after:
        raise ValueError('Frozen v4 files changed during preparation')
    write_json(output / 'preparation.json', provenance)
    report_dir = args.evaluator_report_dir
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / 'wildguardmix_train_overlap_groups_v5.txt').write_text(
            '\n'.join(overlap_groups) + '\n', encoding='utf-8', newline='\n')
        csv_path = report_dir / 'wildguardmix_train_overlap_v5.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(overlap_rows[0]))
            writer.writeheader()
            writer.writerows(overlap_rows)
    aggregate_quality = {key: value for key, value in quality.items() if key != 'near_duplicate_audit'}
    audit = quality['near_duplicate_audit']
    aggregate_quality['near_duplicate_audit'] = {
        'threshold': audit['threshold'], 'template_only_matches': audit['template_only_matches'],
        'merge_components': len(audit['merges']),
        'frozen_group_merge_components': sum(bool(row['frozen_groups']) for row in audit['merges']),
        'excluded_bridge_components': len(audit['excluded_bridge_components']),
        'same_split_frozen_exceptions': len(audit['same_split_frozen_exceptions']),
        'final_cross_split_question_matches': audit['final_cross_split_question_matches']}
    run_dir = ROOT / 'manifests/prompt_runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / (manifest['manifest_id'][8:24] + '.json'), {
        'prompt_manifest_id': manifest['manifest_id'], 'local_output': output.relative_to(ROOT).as_posix(),
        'preparation_sha256': file_sha256(output / 'preparation.json'), 'quality': aggregate_quality,
        'sources': source_manifest['sources'], 'limitations': limitations})
    print(json.dumps({'prompts': len(records), 'pair_table': table, 'overlap': overlap,
                      'tokens': quality['token_estimate'], 'manifest_id': manifest['manifest_id']}, indent=2))


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
    parser.add_argument('--extend-v4', type=Path,
                        help='Extend this frozen v4 folder with paired WildJailbreak rows')
    parser.add_argument('--wild-file', type=Path, help='Downloaded pinned WildJailbreak train TSV')
    parser.add_argument('--new-pairs-per-intent', type=int, default=950)
    parser.add_argument('--evaluator-report-dir', type=Path, help='Write v5 overlap group list and CSV here')
    args = parser.parse_args()
    cap = 6000 if args.extend_v4 else 5000
    if not 1 <= args.min_prompts <= args.max_prompts <= cap or args.seed < 0:
        parser.error(f'Require 1 <= min-prompts <= max-prompts <= {cap} and nonnegative seed')
    if args.extend_v4:
        if args.wild_file is None or not 1 <= args.new_pairs_per_intent <= 1500:
            parser.error('Extension requires --wild-file and 1..1500 new pairs per intent')
        build_v5(args)
        return
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
