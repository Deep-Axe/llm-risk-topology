"""Verify a completed metadata pipeline against captured bytes and source declarations."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from data_contracts import ROOT, validate_bundle
from extract_lineage import extract
from collect_metadata import file_hash


def verify_distributed(snapshot, report, manifest, snapshot_id):
    """Consume only aggregate Spark reports; never load the corpus or edge list."""
    def rows(name):
        return [json.loads(line) for line in (report/name).read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = json.loads((report/'summary.json').read_text(encoding='utf-8'))
    checks = rows('distributed_validation.jsonl')
    if len(checks) != 1 or not checks[0].get('passed'):
        raise ValueError('Distributed verification did not pass')
    check = checks[0]
    sample = rows('edges_sample.jsonl')
    if not 1 <= len(sample) <= 100: raise ValueError('Missing or oversized diagnostic sample')
    validate_bundle({'metadata_edges':sample})
    for field in ['missing_edges','unexpected_edges','invalid_contracts']:
        if check[field] != 0: raise ValueError(f'Distributed check failed: {field}')
    if summary['snapshot_id'] != snapshot_id or summary['sampled_models'] != manifest['record_count']:
        raise ValueError('Snapshot identity/count mismatch')
    for field in ['vertices','external_parent_vertices','weak_components','models_without_declared_lineage','rejected_parent_entries','models_with_valid_lineage']:
        if summary[field] != check[field]: raise ValueError(f'Distributed aggregate mismatch: {field}')
    if summary['typed_edges'] != check['expected_edges']: raise ValueError('Edge count mismatch')
    relation_rows, hubs = rows('reference_relations.jsonl'), rows('reference_hubs.jsonl')
    relations = {r['relation']:r['count'] for r in relation_rows}
    if sum(relations.values()) != summary['typed_edges']: raise ValueError('Relation total mismatch')
    hive = (report/'hive_queries.log').read_text(encoding='utf-8')
    clean = [m.group(0) for line in hive.splitlines() if (m := re.search(r'(?:models|edges|vertices|relation|hub)\t.*$',line))]
    (report/'hive_results.tsv').write_text('\n'.join(clean)+'\n',encoding='utf-8',newline='\n')
    for label,field in [('models','sampled_models'),('edges','typed_edges'),('vertices','vertices')]:
        if f'{label}\t{summary[field]}' not in hive: raise ValueError('Hive count mismatch: '+label)
    for relation,count in relations.items():
        if f'relation\t{relation}\t{count}' not in hive: raise ValueError('Hive relation mismatch')
    for hub in hubs:
        if f'hub\t{hub["parent_model_id"]}\t{hub["count"]}\t' not in hive: raise ValueError('Hive hub mismatch')
    result = {'status':'passed','verification_engine':'spark_distributed',
        'checks':['captured_file_checksums','edge_contracts','exact_source_edge_reference','snapshot_identity',
                  'counts_and_external_parents','weak_component_reference','hive_spark_count_agreement','hive_relation_counts_and_top_hubs'],
        'summary':summary,'relation_counts':relations,'distributed_checks':check}
    (report/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    record = {'schema_version':'1.0.0','metadata_snapshot':manifest,'snapshot_id':snapshot_id,
        'local_snapshot':snapshot.resolve().relative_to(ROOT).as_posix(),'local_evidence':report.resolve().relative_to(ROOT).as_posix(),'validation':result}
    environment = dict(line.split('=',1) for line in (report/'run.env').read_text().splitlines() if '=' in line)
    record['hdfs_root'] = environment['hdfs_root']
    if (report/'timings.tsv').exists(): record['stage_timings'] = list(csv.DictReader((report/'timings.tsv').read_text().splitlines(),delimiter='\t'))
    if (report/'resources.json').exists(): record['resources'] = json.loads((report/'resources.json').read_text())
    measured_baseline = ROOT/'manifests/metadata_runs/034633c5b7f6bab7_distributed.json'
    baseline_path = measured_baseline if summary['sampled_models'] != 5000 and measured_baseline.exists() else ROOT/'manifests/metadata_runs/034633c5b7f6bab7.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        baseline_seconds = {r['stage']:int(r['seconds']) for r in baseline.get('stage_timings',[])}
        current_seconds = {r['stage']:int(r['seconds']) for r in record.get('stage_timings',[])}
        record['comparison_to_5k'] = {'baseline_snapshot_id':baseline['snapshot_id'],
            'sampled_models':{'baseline':5000,'current':summary['sampled_models']},
            'lineage_coverage':{'baseline':baseline['validation']['summary']['models_with_valid_lineage']/5000,
                                'current':summary['models_with_valid_lineage']/summary['sampled_models']},
            'rejected_parent_entries':{'baseline':baseline['validation']['summary']['rejected_parent_entries'],'current':summary['rejected_parent_entries']},
            'stage_seconds':{stage:{'baseline':seconds,'current':current_seconds.get(stage)} for stage,seconds in baseline_seconds.items()},
            'memory_note':'No measured memory baseline exists for the original 5k run. Current resource samples include existing WSL services. No memory ratio is inferred.',
            'runtime_note':'Current Spark timings include distributed reference verification; original 5k Spark timings did not.'}
        if baseline.get('resources') and record.get('resources'):
            record['comparison_to_5k']['memory_observations'] = {
                field:{'baseline':baseline['resources'][field],'current':record['resources'][field]}
                for field in ['minimum_available_kib','maximum_used_kib','maximum_java_rss_kib','maximum_swap_used_kib']}
            record['comparison_to_5k']['memory_note'] = 'Both runs were sampled every two seconds with the same settings. Values cover the whole WSL system/all Java services; they are not isolated heap peaks.'
            record['comparison_to_5k']['runtime_note'] = 'Both measured runs include distributed source verification in the Spark phase. Initial 5k timings remain a separate historical baseline.'
    destination = ROOT/'manifests/metadata_runs'/(snapshot_id[:16]+'_distributed.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--mode', choices=['auto','python','distributed'], default='auto')
    args = parser.parse_args()
    manifest_path = args.snapshot/'metadata_snapshot.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    snapshot_id = file_hash(manifest_path)
    if manifest['status'] != 'frozen' or manifest['record_count'] != manifest['requested_records']:
        raise ValueError('Only completed, frozen snapshots can be verified')
    if manifest['record_count'] >= 10000 and args.mode == 'python':
        parser.error('Large snapshots require distributed verification; use auto or distributed')
    for artifact in manifest['artifacts']:
        if file_hash(args.snapshot/artifact['path']) != artifact['sha256']: raise ValueError('Captured artifact checksum mismatch')
    if args.mode == 'distributed' or (args.mode == 'auto' and manifest['record_count'] >= 10000):
        verify_distributed(args.snapshot,args.report,manifest,snapshot_id)
        return
    with gzip.open(args.snapshot/'models.jsonl.gz', 'rt', encoding='utf-8') as stream:
        models = [json.loads(line) for line in stream]
    expected = set()
    statuses = Counter()
    no_lineage = set()
    children = set()
    for model in models:
        for status, child, parent, relation, field, value, reason in extract(model['id'], model.get('cardData') or {}):
            statuses[status] += 1
            if status == 'valid':
                expected.add((child, parent, relation, field, value))
                children.add(child)
            if status == 'no_lineage':
                no_lineage.add(child)
    edges = [json.loads(line) for line in (args.report/'edges.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_bundle({'metadata_edges': edges})
    assert len(edges) == len(expected)
    actual = {(e['child_model_id'], e['parent_model_id'], e['relation'], e['evidence'][0]['field'], e['evidence'][0]['value']) for e in edges}
    assert actual == expected, 'Pig/Spark output differs from captured-card reference'
    summary = json.loads((args.report/'summary.json').read_text(encoding='utf-8'))
    assert summary['snapshot_id'] == snapshot_id
    assert all(e['snapshot_id'] == snapshot_id for e in edges)
    assert summary['sampled_models'] == len(models) == manifest['record_count']
    assert summary['typed_edges'] == len(edges)
    assert summary['models_with_valid_lineage'] == len(children)
    assert summary['models_without_declared_lineage'] == len(no_lineage)
    assert summary['rejected_parent_entries'] == statuses['rejected']
    ids = {m['id'] for m in models}
    external = {e['parent_model_id'] for e in edges} - ids
    assert summary['vertices'] == len(ids | external)
    assert summary['external_parent_vertices'] == len(external)
    simple = {(e['child_model_id'], e['parent_model_id']) for e in edges}
    assert summary['simple_edges'] == len(simple)
    parents = {name: name for name in ids | external}
    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name
    for child, parent in simple:
        parents[find(child)] = find(parent)
    assert summary['weak_components'] == len({find(name) for name in parents})
    hive = (args.report/'hive_queries.log').read_text(encoding='utf-8')
    clean_rows = [match.group(0) for line in hive.splitlines()
                  if (match := re.search(r'(?:models|edges|vertices|relation|hub)\t.*$', line))]
    (args.report/'hive_results.tsv').write_text('\n'.join(clean_rows)+'\n', encoding='utf-8', newline='\n')
    for label, count in [('models', len(models)), ('edges', len(edges)), ('vertices', summary['vertices'])]:
        assert f'{label}\t{count}' in hive, f'Hive count missing or inconsistent: {label}'
    relations = Counter(e['relation'] for e in edges)
    for relation, count in relations.items():
        assert f'relation\t{relation}\t{count}' in hive, 'Hive relation counts disagree'
    children_per_parent = Counter(parent for child, parent in simple)
    for parent, count in sorted(children_per_parent.items(), key=lambda item: (-item[1], item[0]))[:10]:
        assert f'hub\t{parent}\t{count}\t' in hive, 'Hive hub query disagrees with source graph'
    result = {'status': 'passed', 'checks': ['captured_file_checksums', 'edge_contracts', 'exact_source_edge_reference',
        'snapshot_identity', 'counts_and_external_parents', 'weak_component_reference', 'hive_spark_count_agreement',
        'hive_relation_counts_and_top_hubs'], 'summary': summary,
        'relation_counts': dict(Counter(e['relation'] for e in edges)), 'source_status_counts': dict(statuses)}
    (args.report/'validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    record = {'schema_version': '1.0.0', 'metadata_snapshot': manifest, 'snapshot_id': snapshot_id,
        'local_snapshot': args.snapshot.resolve().relative_to(ROOT).as_posix(), 'local_evidence': args.report.resolve().relative_to(ROOT).as_posix(),
        'validation': result}
    environment = dict(line.split('=', 1) for line in (args.report/'run.env').read_text(encoding='utf-8').splitlines() if '=' in line)
    record['hdfs_root'] = environment['hdfs_root']
    timing = args.report/'timings.tsv'
    if timing.exists():
        record['stage_timings'] = list(csv.DictReader(timing.read_text(encoding='utf-8').splitlines(), delimiter='\t'))
    destination = ROOT/'manifests/metadata_runs'/(snapshot_id[:16]+'.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
