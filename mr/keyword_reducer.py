"""Hadoop streaming reducer: sorted (keyword, model_id, category, field) -> one row per keyword and model.

Fields that matched for the same keyword and model are joined with '+', e.g. 'name+tags'.
"""
import sys


def flush(key, fields):
    if key:
        keyword, model_id, category = key
        print(f'{keyword}\t{model_id}\t{category}\t{"+".join(sorted(fields))}')


def main():
    current, fields = None, set()
    for line in sys.stdin:
        keyword, model_id, category, field = line.rstrip('\n').split('\t')
        key = (keyword, model_id, category)
        if key != current:
            flush(current, fields)
            current, fields = key, set()
        fields.add(field)
    flush(current, fields)


if __name__ == '__main__':
    main()
