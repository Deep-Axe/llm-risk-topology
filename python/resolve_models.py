"""Pin model and tokenizer commits and inspect template defaults.

Run: python python/resolve_models.py --output manifests/models
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from data_contracts import VERSION, manifest_identity, validate_bundle


PINS = {
    'mistralai/Mistral-7B-Instruct-v0.1': 'ec5deb64f2c6',
    'HuggingFaceH4/zephyr-7b-beta': '892b3d7a7b1c',
    'teknium/OpenHermes-2.5-Mistral-7B': '24c0bea14d53',
    'meta-llama/Llama-3.1-8B-Instruct': '0e9e39f249a16976918f6564b8830bc894c89659',
    'NousResearch/Hermes-3-Llama-3.1-8B': '896ea440e5a9',
    'dphn/dolphin-2.9.4-llama3.1-8b': 'ca1f6d135846',
    'allenai/Llama-3.1-Tulu-3-8B': '666943798adb',
    'Qwen/Qwen2.5-7B-Instruct': 'a09a35458c70',
    'meta-llama/Llama-3.1-8B': 'd04e592bb4f6',
    'mistralai/Mistral-7B-v0.1': '27d67f1b5f57',
    'Qwen/Qwen2.5-7B': 'd14972939875',
}
SMOKE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SMOKE_PIN = '7ae557604adf67be50417f59c2c2f167def9a775'
SYSTEM_POLICY = 'user_only_template_defaults'
EXAMPLE = [{'role': 'user', 'content': 'How do I boil an egg?'}]
SLOW_TOKENIZERS = {'teknium/OpenHermes-2.5-Mistral-7B', 'allenai/wildguard'}


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def tokenizer_for(model):
    return AutoTokenizer.from_pretrained(model['model_id'], revision=model['tokenizer_revision'],
                                          use_fast=model['model_id'] not in SLOW_TOKENIZERS,
                                          trust_remote_code=False)


def template_text(tokenizer):
    return tokenizer.get_chat_template() if tokenizer.chat_template else ''


def resolve_entry(model_id, revision, role='primary_assistant', api=None):
    api = api or HfApi()
    resolved = api.model_info(model_id, revision=revision).sha
    if not resolved.startswith(revision):
        raise ValueError('Resolved revision differs from the requested commit')
    model = {'model_id': model_id, 'revision': resolved, 'tokenizer_revision': resolved,
             'chat_template_sha256': None, 'family': 'Llama-3.1' if 'llama' in model_id.lower() else
             ('Qwen2.5' if 'Qwen2.5' in model_id else 'Mistral-7B-v0.1'),
             'role': role, 'precision': 'float16', 'system_policy': SYSTEM_POLICY}
    tokenizer = tokenizer_for(model)
    template = template_text(tokenizer)
    if role == 'primary_assistant' and not template:
        raise ValueError(f'Assistant has no chat template: {model_id}')
    model['chat_template_sha256'] = sha256(template)
    if role == 'base_diagnostic':
        model['system_policy'] = 'base_completion_separate_protocol_required'
    rendered = tokenizer.apply_chat_template(EXAMPLE, tokenize=False, add_generation_prompt=True) \
        if template else EXAMPLE[0]['content']
    evidence = {'model_id': model_id, 'weight_repository': model_id, 'revision': resolved,
                'tokenizer_revision': resolved, 'tokenizer_class': type(tokenizer).__name__,
                'chat_template_sha256': model['chat_template_sha256'],
                'system_policy': model['system_policy'], 'example_messages': EXAMPLE,
                'rendered_example': rendered}
    return model, evidence


def model_bundle(models, stage, status):
    manifest = {'schema_version': VERSION, 'status': status, 'stage': stage, 'models': models}
    manifest['manifest_id'] = manifest_identity('model_manifests', manifest)
    bundle = {'model_manifests': [manifest]}
    validate_bundle(bundle)
    return bundle


def write_new(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Refusing to overwrite different manifest: {path}')
        return
    path.write_text(text, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('manifests/models'))
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models, examples = [], []
    if not args.smoke_only:
        for index, (name, pin) in enumerate(PINS.items()):
            role = 'primary_assistant' if index < 8 else 'base_diagnostic'
            model, example = resolve_entry(name, pin, role)
            models.append(model)
            examples.append(example)
            print(name, model['revision'], flush=True)
        write_new(args.output / 'pilot.bundle.json', model_bundle(models[:2], 'pilot', 'frozen'))
        write_new(args.output / 'panel8.draft.bundle.json', model_bundle(models[:8], '8', 'draft'))
        write_new(args.output / 'bases.draft.bundle.json', model_bundle(models[8:], 'pilot', 'draft'))
        write_new(args.output / 'template_examples.json', examples)
    smoke, example = resolve_entry(SMOKE_MODEL, SMOKE_PIN)
    write_new(args.output / 'smoke.bundle.json', model_bundle([smoke], 'pilot', 'frozen'))
    write_new(args.output / 'smoke_template_example.json', example)
    print('Validated model manifests:', args.output, flush=True)


if __name__ == '__main__':
    main()
