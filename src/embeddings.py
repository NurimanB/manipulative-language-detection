"""Tokenization and averaged word-vector features (Experiments 3 and 4)."""

import re

import numpy as np


def tokenize_text(text):
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?",
        str(text).lower()
    )


def mean_embedding(text, keyed_vectors):
    tokens = tokenize_text(text)
    vectors = [
        keyed_vectors[word]
        for word in tokens
        if word in keyed_vectors
    ]

    if not vectors:
        return np.zeros(keyed_vectors.vector_size)

    return np.mean(vectors, axis=0)


def vectorize_texts(texts, keyed_vectors):
    return np.vstack([
        mean_embedding(text, keyed_vectors)
        for text in texts
    ])


def embedding_coverage(texts, keyed_vectors):
    total_tokens = 0
    known_tokens = 0

    for text in texts:
        tokens = tokenize_text(text)
        total_tokens += len(tokens)
        known_tokens += sum(word in keyed_vectors for word in tokens)

    coverage = known_tokens / total_tokens if total_tokens else 0
    return known_tokens, total_tokens, coverage
