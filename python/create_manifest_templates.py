"""Build draft project manifests and an explicitly synthetic validation fixture."""
from copy import deepcopy
import hashlib
import json

from data_contracts import ROOT, VERSION, digest, normalize_messages, prompt_hash, prompt_id, edge_id, response_id, assign_groups, manifest_identity, validate_bundle

PANEL = [
    ('mistralai/Mistral-7B-Instruct-v0.1', 'Mistral-7B-v0.1'),
    ('HuggingFaceH4/zephyr-7b-beta', 'Mistral-7B-v0.1'),
    ('teknium/OpenHermes-2.5-Mistral-7B', 'Mistral-7B-v0.1'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'Llama-3.1-8B'),
    ('NousResearch/Hermes-3-Llama-3.1-8B', 'Llama-3.1-8B'),
    ('dphn/dolphin-2.9.4-llama3.1-8b', 'Llama-3.1-8B'),
    ('allenai/Llama-3.1-Tulu-3-8B', 'Llama-3.1-8B'),
    ('Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B'),
]


def seal(kind, record):
    record['schema_version'] = VERSION
    record['run_id' if kind == 'run_manifests' else 'manifest_id'] = manifest_identity(kind, record)
    return record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def templates():
    inventory = json.loads((ROOT/'manifests/dataset_schema_inventory.json').read_text(encoding='utf-8'))
    sources = []
    for source in inventory['sources']:
        license_value = source.get('license')
        sources.append({'source_id': source['source'], 'provider': source['provider'], 'repository': source['source'],
            'revision': source['revision'], 'url': source['url'], 'license': license_value if isinstance(license_value, str) else None,
            'access_status': 'verified' if source['sampled_rows'] else 'card_only', 'stretch': source['stretch']})
    src = seal('source_manifests', {'status': 'draft', 'sources': sources})
    mod = seal('model_manifests', {'status': 'draft', 'stage': '8', 'models': [
        {'model_id': mid, 'revision': None, 'tokenizer_revision': None, 'chat_template_sha256': None,
         'family': family, 'role': 'primary_assistant', 'precision': None, 'system_policy': None} for mid, family in PANEL]})
    prompts = seal('prompt_manifests', {'status': 'draft', 'source_manifest_id': src['manifest_id'],
        'grouping_version': VERSION, 'selection': {'seed': 42, 'description': 'Pending source adapters, deduplication, grouping and audited pilot selection.'},
        'artifacts': [], 'prompt_ids': [], 'group_splits': {}})
    run = seal('run_manifests', {'status': 'draft', 'prompt_manifest_id': prompts['manifest_id'], 'model_manifest_id': mod['manifest_id'],
        'generation': {'seed': 42, 'do_sample': False, 'temperature': 0, 'top_p': 1, 'max_input_tokens': 2048, 'max_new_tokens': 256, 'repetitions': 1},
        'runtime': {'engine': None, 'engine_version': None, 'hardware': None}, 'output_path': 'data/raw/responses/pilot/'})
    return {'source_manifests': [src], 'model_manifests': [mod], 'prompt_manifests': [prompts], 'run_manifests': [run]}


def synthetic_fixture():
    # These identifiers/revisions represent invented records, never downloaded data.
    revision = 'a'*40
    src = seal('source_manifests', {'status': 'frozen', 'sources': [{'source_id': 'fixture/prompts', 'provider': 'github',
        'repository': 'fixture/prompts', 'revision': revision, 'url': 'https://example.org/synthetic-fixture',
        'license': 'CC0-1.0', 'access_status': 'verified', 'stretch': False}]})
    records = []
    for i, (text, variant) in enumerate([('Summarize the water cycle.', 'original'), ('Please explain the water cycle briefly.', 'paraphrase')]):
        messages = normalize_messages([{'role': 'user', 'content': text}])
        records.append({'schema_version': VERSION, 'prompt_id': prompt_id('fixture/prompts', revision, 'default', 'train', str(i), variant, messages),
            'source_id': 'fixture/prompts', 'source_revision': revision, 'source_config': 'default', 'source_split': 'train',
            'source_record_id': str(i), 'messages': messages, 'task_type': 'assistant', 'variant_type': variant,
            'parent_prompt_id': None, 'group_anchors': ['fixture:water-cycle', 'messages:'+prompt_hash(messages)],
            'prompt_sha256': prompt_hash(messages), 'source_labels': {}})
    groups = assign_groups({p['prompt_id']: p['group_anchors'] for p in records})
    for p in records:
        p['prompt_group_id'] = groups[p['prompt_id']]
    records[1]['parent_prompt_id'] = records[0]['prompt_id']
    artifact = ROOT/'fixtures/contracts/prompts.json'
    write(artifact, records)
    prm = seal('prompt_manifests', {'status': 'frozen', 'source_manifest_id': src['manifest_id'], 'grouping_version': VERSION,
        'selection': {'seed': 42, 'description': 'Synthetic contract fixture only; not experimental evidence.'},
        'artifacts': [{'path': 'fixtures/contracts/prompts.json', 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'row_count': 2}],
        'prompt_ids': [p['prompt_id'] for p in records], 'group_splits': {records[0]['prompt_group_id']: 'development'}})
    mod = seal('model_manifests', {'status': 'frozen', 'stage': 'pilot', 'models': [{'model_id': 'fixture/assistant',
        'revision': revision, 'tokenizer_revision': revision, 'chat_template_sha256': digest('fixture template'),
        'family': 'fixture', 'role': 'primary_assistant', 'precision': 'float32', 'system_policy': 'No system message.'}]})
    run = seal('run_manifests', {'status': 'frozen', 'prompt_manifest_id': prm['manifest_id'], 'model_manifest_id': mod['manifest_id'],
        'generation': {'seed': 42, 'do_sample': False, 'temperature': 0, 'top_p': 1, 'max_input_tokens': 128, 'max_new_tokens': 64, 'repetitions': 1},
        'runtime': {'engine': 'synthetic-fixture', 'engine_version': VERSION, 'hardware': 'none'}, 'output_path': 'fixtures/contracts/'})
    snapshot = digest('synthetic metadata snapshot')
    edge = {'schema_version': VERSION, 'edge_id': edge_id(snapshot, 'fixture/assistant', 'fixture/base', 'fine_tune'),
        'snapshot_id': snapshot, 'child_model_id': 'fixture/assistant', 'parent_model_id': 'fixture/base', 'relation': 'fine_tune',
        'lineage_status': 'declared', 'evidence': [{'url': 'https://example.org/synthetic-fixture', 'revision': revision, 'field': 'base_model', 'value': 'fixture/base'}]}
    response = {'schema_version': VERSION, 'response_id': response_id(run['run_id'], records[0]['prompt_id'], 'fixture/assistant', revision, 0),
        'run_id': run['run_id'], 'prompt_id': records[0]['prompt_id'], 'model_id': 'fixture/assistant', 'model_revision': revision,
        'repetition': 0, 'status': 'success', 'text': 'Water evaporates, condenses and falls as precipitation.', 'error': None,
        'input_tokens': 8, 'output_tokens': 12, 'input_truncated': False, 'output_truncated': False, 'finish_reason': 'stop'}
    return {'source_manifests': [src], 'model_manifests': [mod], 'prompt_manifests': [prm], 'run_manifests': [run],
            'metadata_edges': [edge], 'prompts': records, 'responses': [response]}


if __name__ == '__main__':
    for path, bundle in [(ROOT/'manifests/templates/experiment.bundle.json', templates()),
                         (ROOT/'fixtures/contracts/example.bundle.json', synthetic_fixture())]:
        validate_bundle(bundle)
        write(path, bundle)
        print('Created and validated', path.relative_to(ROOT))
