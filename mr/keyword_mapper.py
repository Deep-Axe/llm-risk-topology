"""Hadoop streaming mapper: model card metadata -> (keyword, model_id, category, field).

Input lines: model_id TAB revision TAB card_json (the collected snapshot's models.tsv).
Matches are metadata indicators declared by uploaders, not measured model behavior.
"""
import json
import re
import sys

CATEGORIES = {
    'safety_removed': ['uncensored', 'abliterated', 'abliteration', 'unaligned', 'unfiltered', 'decensored',
                       'nsfw', 'erp', 'jailbreak', 'jailbroken'],
    'safety_tuned': ['safety', 'guard', 'guardrail', 'harmless', 'rlhf', 'dpo', 'orpo', 'alignment',
                     'alignment-handbook'],
    'roleplay': ['roleplay', 'roleplaying', 'rp'],
    'quantized': ['gguf', 'gguf-my-repo', 'awq', 'gptq', 'exl2', 'quantized', 'bnb', '4-bit', '8-bit'],
    'adapter': ['lora', 'qlora', 'peft', 'adapter'],
    'merged': ['merge', 'mergekit', 'lazymergekit', 'automerger', 'merged'],
}
KEYWORDS = {word: category for category, words in CATEGORIES.items() for word in words}
# Repository names are split on separators so 'Qwen-7B-abliterated' matches but 'merger' does not.
NAME_SPLIT = re.compile(r'[^a-z0-9]+')


def matches(model_id, card):
    found = set()
    tags = card.get('tags') if isinstance(card.get('tags'), list) else []
    for tag in tags:
        if isinstance(tag, str) and tag.lower() in KEYWORDS:
            found.add((tag.lower(), 'tags'))
    datasets = card.get('datasets') if isinstance(card.get('datasets'), list) else []
    for name in datasets:
        if isinstance(name, str):
            for token in NAME_SPLIT.split(name.lower()):
                if token in KEYWORDS and KEYWORDS[token] in ('safety_removed', 'safety_tuned'):
                    found.add((token, 'datasets'))
    for token in NAME_SPLIT.split(model_id.split('/', 1)[-1].lower()):
        if token in KEYWORDS:
            found.add((token, 'name'))
    return found


def main():
    for line in sys.stdin:
        parts = line.rstrip('\n').split('\t', 2)
        if len(parts) != 3:
            continue
        model_id, _, payload = parts
        try:
            card = json.loads(payload)
        except ValueError:
            continue
        for keyword, field in sorted(matches(model_id, card)):
            print(f'{keyword}\t{model_id}\t{KEYWORDS[keyword]}\t{field}')


if __name__ == '__main__':
    main()
