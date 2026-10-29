"""Version 1 data contracts, deterministic identities, and bundle validation.

Export schemas: python python/data_contracts.py export
Validate a JSON bundle: python python/data_contracts.py validate PATH
Bundles contain arrays named for CONTRACTS below. See project_notes/data_contracts.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"


def canonical(value):
    """Only explicit message normalization changes text; no stripping/case folding."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def normalize_messages(messages):
    return [{"role": m["role"], "content": unicodedata.normalize("NFC", m["content"].replace("\r\n", "\n").replace("\r", "\n"))} for m in messages]


def prompt_hash(messages):
    return digest(normalize_messages(messages))


def prompt_id(source, revision, config, split, record_id, variant_type, messages):
    return "p_" + digest([VERSION, source, revision, config, split, record_id, variant_type, prompt_hash(messages)])


def edge_id(snapshot_id, child, parent, relation):
    return "e_" + digest([VERSION, snapshot_id, child, parent, relation])


def response_id(run_id, prompt, model, revision, repetition):
    return "r_" + digest([VERSION, run_id, prompt, model, revision, repetition])


def assign_groups(prompt_anchors):
    """Union records sharing ANY explicit anchor, including cross-source duplicates.

    Input: {prompt_id: [nonempty canonical anchor strings]}. Caller supplies
    original/conversation IDs and duplicate-content anchors from audited adapters.
    Group ID hashes the smallest anchor of each connected component. Freeze these
    memberships in the prompt manifest; additions can merge existing components.
    """
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for anchors in prompt_anchors.values():
        if not anchors or any(not isinstance(a, str) or not a for a in anchors):
            raise ValueError("Each prompt requires nonempty string group anchors")
        first = find(anchors[0])
        for anchor in anchors[1:]:
            a, b = find(first), find(anchor)
            parents[max(a, b)] = min(a, b)
    return {p: "g_" + digest([VERSION, find(anchors[0])]) for p, anchors in prompt_anchors.items()}


def obj(properties, required=None, **extra):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False, **extra}


def array(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


S = {"type": "string", "minLength": 1}
N = {"type": "integer", "minimum": 0}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
REV = {"type": "string", "pattern": "^[0-9a-f]{40}$"}
MODEL = {"type": "string", "pattern": "^[^/\\s]+/[^/\\s]+$"}
NULL = {"type": "null"}
ANY_OBJECT = {"type": "object"}


def nullable(schema):
    return {"anyOf": [schema, NULL]}


def identity(prefix):
    return {"type": "string", "pattern": "^" + prefix + "_[0-9a-f]{64}$"}


MESSAGE = obj({"role": {"enum": ["system", "user", "assistant", "tool"]}, "content": {"type": "string"}})
SOURCE = obj({"source_id": S, "provider": {"enum": ["huggingface", "github"]}, "repository": MODEL,
              "revision": nullable(REV), "url": S, "license": nullable(S),
              "access_status": {"enum": ["verified", "card_only", "pending", "failed"]}, "stretch": {"type": "boolean"}})
ARTIFACT = obj({"path": S, "sha256": nullable(SHA), "row_count": N})
MODEL_ENTRY = obj({"model_id": MODEL, "revision": nullable(REV), "tokenizer_revision": nullable(REV),
                   "chat_template_sha256": nullable(SHA), "family": S,
                   "role": {"enum": ["primary_assistant", "base_diagnostic"]}, "precision": nullable(S),
                   "system_policy": nullable(S)})
COMMON = {"schema_version": {"const": VERSION}}
STATUS = {"enum": ["draft", "frozen"]}

CONTRACTS = {
    "metadata_edges": obj({**COMMON, "edge_id": identity("e"), "snapshot_id": SHA,
        "child_model_id": MODEL, "parent_model_id": MODEL,
        "relation": {"enum": ["fine_tune", "adapter", "merge", "quantization", "other"]},
        "lineage_status": {"enum": ["declared", "inferred", "audited"]},
        "evidence": array(obj({"url": S, "revision": nullable(REV), "field": S, "value": S}), 1)}),
    "prompts": obj({**COMMON, "prompt_id": identity("p"), "source_id": S, "source_revision": REV,
        "source_config": S, "source_split": S, "source_record_id": S,
        "messages": array(MESSAGE, 1), "task_type": {"enum": ["assistant", "summarization", "classification", "prompt_injection"]},
        "variant_type": {"enum": ["original", "adversarial", "defense", "paraphrase", "conversation_turn", "unpaired"]},
        "parent_prompt_id": nullable(identity("p")), "prompt_group_id": identity("g"),
        "group_anchors": array(S, 1), "prompt_sha256": SHA, "source_labels": ANY_OBJECT}),
    "responses": obj({**COMMON, "response_id": identity("r"), "run_id": identity("run"),
        "prompt_id": identity("p"), "model_id": MODEL, "model_revision": REV, "repetition": N,
        "status": {"enum": ["success", "failed"]}, "text": nullable({"type": "string"}),
        "error": nullable(S), "input_tokens": nullable(N), "output_tokens": nullable(N),
        "input_truncated": {"type": "boolean"}, "output_truncated": {"type": "boolean"},
        "finish_reason": nullable(S)}),
    "source_manifests": obj({**COMMON, "manifest_id": identity("sources"), "status": STATUS, "sources": array(SOURCE, 1)}),
    "model_manifests": obj({**COMMON, "manifest_id": identity("models"), "status": STATUS,
        "stage": {"enum": ["pilot", "8", "25", "100"]}, "models": array(MODEL_ENTRY, 1)}),
    "prompt_manifests": obj({**COMMON, "manifest_id": identity("prompts"), "status": STATUS,
        "source_manifest_id": identity("sources"), "grouping_version": {"const": VERSION},
        "selection": obj({"seed": N, "description": S}), "artifacts": array(ARTIFACT),
        "prompt_ids": {**array(identity("p")), "uniqueItems": True},
        "group_splits": {"type": "object", "propertyNames": identity("g"),
                         "additionalProperties": {"enum": ["development", "validation", "test"]}}}),
    "run_manifests": obj({**COMMON, "run_id": identity("run"), "status": STATUS,
        "prompt_manifest_id": identity("prompts"), "model_manifest_id": identity("models"),
        "generation": obj({"seed": N, "do_sample": {"type": "boolean"},
            "temperature": {"type": "number", "minimum": 0}, "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "max_input_tokens": {"type": "integer", "minimum": 1}, "max_new_tokens": {"type": "integer", "minimum": 1},
            "repetitions": {"type": "integer", "minimum": 1}}),
        "runtime": obj({"engine": nullable(S), "engine_version": nullable(S), "hardware": nullable(S)}),
        "output_path": S}),
}


def manifest_identity(kind, record):
    key, prefix = ("run_id", "run") if kind == "run_manifests" else ("manifest_id", {"source_manifests": "sources", "model_manifests": "models", "prompt_manifests": "prompts"}[kind])
    return prefix + "_" + digest({k: v for k, v in record.items() if k != key})


def validate_bundle(bundle):
    """Validate types, identities, links, immutable revisions, and leakage guards.

    This is an in-memory pilot validator. Full-corpus joins belong in Spark.
    A frozen manifest bundle must include its referenced manifests and prompts.
    """
    unknown = set(bundle) - set(CONTRACTS)
    if unknown:
        raise ValueError(f"Unknown bundle collections: {sorted(unknown)}")
    for kind, records in bundle.items():
        if not isinstance(records, list):
            raise ValueError(f"{kind} must be an array")
        validator = Draft202012Validator(CONTRACTS[kind])
        seen = set()
        key = "run_id" if kind == "run_manifests" else {"prompts": "prompt_id", "responses": "response_id", "metadata_edges": "edge_id"}.get(kind, "manifest_id")
        for record in records:
            validator.validate(record)
            if record[key] in seen:
                raise ValueError(f"Duplicate {key}")
            seen.add(record[key])
            if kind.endswith("manifests") and record[key] != manifest_identity(kind, record):
                raise ValueError(f"Invalid content identity: {kind}")

    sources = {m['manifest_id']: m for m in bundle.get('source_manifests', [])}
    models = {m['manifest_id']: m for m in bundle.get('model_manifests', [])}
    manifests = {m['manifest_id']: m for m in bundle.get('prompt_manifests', [])}
    runs = {m['run_id']: m for m in bundle.get('run_manifests', [])}
    prompts = {p['prompt_id']: p for p in bundle.get('prompts', [])}
    group_map = assign_groups({p['prompt_id']: p['group_anchors'] for p in prompts.values()})
    for e in bundle.get('metadata_edges', []):
        if e['child_model_id'] == e['parent_model_id'] or e['edge_id'] != edge_id(e['snapshot_id'], e['child_model_id'], e['parent_model_id'], e['relation']):
            raise ValueError('Invalid edge identity or self edge')
    for p in prompts.values():
        if p['messages'][-1]['role'] != 'user':
            raise ValueError('Prompt history must end at the user turn to regenerate')
        if p['messages'] != normalize_messages(p['messages']) or p['prompt_sha256'] != prompt_hash(p['messages']):
            raise ValueError('Prompt text/hash not canonical')
        if p['prompt_id'] != prompt_id(p['source_id'], p['source_revision'], p['source_config'], p['source_split'], p['source_record_id'], p['variant_type'], p['messages']):
            raise ValueError('Invalid prompt identity')
        if p['prompt_group_id'] != group_map[p['prompt_id']]:
            raise ValueError('Group does not match connected anchors')
        parent = p['parent_prompt_id']
        visited = {p['prompt_id']}
        while parent:
            if parent in visited or parent not in prompts:
                raise ValueError('Missing parent or parent cycle')
            visited.add(parent)
            if prompts[parent]['prompt_group_id'] != p['prompt_group_id']:
                raise ValueError('Parent and variant must share a group')
            parent = prompts[parent]['parent_prompt_id']
    hashes = {}
    for p in prompts.values():
        if p['prompt_sha256'] in hashes and hashes[p['prompt_sha256']] != p['prompt_group_id']:
            raise ValueError('Identical messages split across groups')
        hashes[p['prompt_sha256']] = p['prompt_group_id']
    for m in sources.values():
        ids = [s['source_id'] for s in m['sources']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate source ID')
        if m['status'] == 'frozen' and any(s['revision'] is None for s in m['sources']):
            raise ValueError('Frozen sources require revisions')
    for m in models.values():
        ids = [x['model_id'] for x in m['models']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate model ID')
        if m['status'] == 'frozen':
            if any(any(x[k] is None for k in ['revision', 'tokenizer_revision', 'chat_template_sha256', 'precision', 'system_policy']) for x in m['models']):
                raise ValueError('Frozen models require resolved revisions and inference policy')
            if m['stage'] != 'pilot' and sum(x['role'] == 'primary_assistant' for x in m['models']) != int(m['stage']):
                raise ValueError('Frozen model count does not match stage')
    for m in manifests.values():
        if m['status'] != 'frozen':
            continue
        source = sources.get(m['source_manifest_id'])
        if not source or source['status'] != 'frozen':
            raise ValueError('Frozen prompts require frozen source manifest')
        selected = [prompts.get(pid) for pid in m['prompt_ids']]
        if not selected or any(p is None for p in selected):
            raise ValueError('Frozen prompt selection is empty or references missing records')
        if not m['artifacts'] or any(a['sha256'] is None for a in m['artifacts']) or sum(a['row_count'] for a in m['artifacts']) != len(selected):
            raise ValueError('Frozen prompt artifacts require hashes and matching counts')
        available = {s['source_id']: s['revision'] for s in source['sources']}
        if any(available.get(p['source_id']) != p['source_revision'] for p in selected):
            raise ValueError('Prompt source revision mismatch')
        if set(m['group_splits']) != {p['prompt_group_id'] for p in selected}:
            raise ValueError('Every selected group needs exactly one split assignment')
        if any(p['parent_prompt_id'] and p['parent_prompt_id'] not in m['prompt_ids'] for p in selected):
            raise ValueError('Selected variant requires its parent in selection')
    for r in runs.values():
        if r['status'] == 'frozen':
            if any(r['runtime'][k] is None for k in r['runtime']):
                raise ValueError('Frozen runs require runtime details')
            if any(table.get(r[key], {}).get('status') != 'frozen' for table, key in [(models, 'model_manifest_id'), (manifests, 'prompt_manifest_id')]):
                raise ValueError('Frozen run requires frozen model and prompt manifests')
    for r in bundle.get('responses', []):
        run = runs.get(r['run_id'])
        if not run or run['status'] != 'frozen':
            raise ValueError('Response requires a frozen run')
        panel = models[run['model_manifest_id']]
        if not any(m['model_id'] == r['model_id'] and m['revision'] == r['model_revision'] for m in panel['models']):
            raise ValueError('Response checkpoint not in run panel')
        if r['prompt_id'] not in manifests[run['prompt_manifest_id']]['prompt_ids'] or r['repetition'] >= run['generation']['repetitions']:
            raise ValueError('Response prompt/repetition outside run')
        if r['response_id'] != response_id(r['run_id'], r['prompt_id'], r['model_id'], r['model_revision'], r['repetition']):
            raise ValueError('Invalid response identity')
        if (r['status'] == 'success' and (r['text'] is None or r['error'] is not None)) or (r['status'] == 'failed' and (r['text'] is not None or r['error'] is None)):
            raise ValueError('Inconsistent response status/text/error')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'validate'])
    parser.add_argument('path', nargs='?', type=Path)
    parser.add_argument('--verify-artifacts', action='store_true', help='Also verify selected prompt file bytes against manifest checksums')
    args = parser.parse_args()
    if args.command == 'export':
        directory = ROOT / 'schemas'
        directory.mkdir(exist_ok=True)
        for name, schema in CONTRACTS.items():
            Draft202012Validator.check_schema(schema)
            (directory / (name + '.schema.json')).write_text(json.dumps({'$schema': 'https://json-schema.org/draft/2020-12/schema', 'title': name, **schema}, indent=2), encoding='utf-8')
        print(f'Exported {len(CONTRACTS)} versioned contracts.')
    else:
        if args.path is None:
            parser.error('validate requires a bundle path')
        bundle = json.loads(args.path.read_text(encoding='utf-8'))
        validate_bundle(bundle)
        if args.verify_artifacts:
            verify_artifacts(bundle)
        print('Bundle validated.')


def verify_artifacts(bundle, root=ROOT):
    for manifest in bundle.get('prompt_manifests', []):
        for artifact in manifest['artifacts']:
            if artifact['sha256'] is None:
                raise ValueError('Cannot verify artifact without checksum')
            path = (root / artifact['path']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Local artifact must stay inside project root')
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    checksum.update(chunk)
            if checksum.hexdigest() != artifact['sha256']:
                raise ValueError('Artifact checksum mismatch')


if __name__ == '__main__':
    main()
