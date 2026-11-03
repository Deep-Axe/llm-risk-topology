"""Embed texts with pinned MiniLM, or pass through vectors already computed.

Usage:
    python python/embed.py --texts-jsonl prompts.jsonl --text-field prompt_text --output out.parquet
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MINILM_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def unit_rows(vectors):
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def embed_texts(texts, batch_size=64, device=None):
    """Return unit-length 384-d MiniLM vectors in input order."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MINILM_ID, revision=MINILM_REVISION, device=device)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float64)


def write_embeddings(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['response_id', 'prompt_id', 'model_id', 'vector']
    frame[columns].to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--texts-jsonl', type=Path, required=True)
    parser.add_argument('--id-field', required=True)
    parser.add_argument('--text-field', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', default='prompt')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.texts_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
    vectors = embed_texts([row[args.text_field] for row in rows])
    frame = pd.DataFrame({
        'response_id': '',
        'prompt_id': [row[args.id_field] for row in rows],
        'model_id': args.model_id,
        'vector': list(vectors),
    })
    write_embeddings(args.output, frame)
    print('wrote %d vectors to %s' % (len(frame), args.output))


if __name__ == '__main__':
    main()
