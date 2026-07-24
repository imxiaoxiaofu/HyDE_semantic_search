from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding")

    for start in iterator:
        batch = texts[start : start + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    return np.asarray(embeddings)


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    np.save(path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(path)
