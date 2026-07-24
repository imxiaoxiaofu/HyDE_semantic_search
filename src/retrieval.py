from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchHit:
    rank: int
    item_id: Any
    score: float
    item_name: str
    combined_text: str


class FaissRetriever:
    def __init__(self, item_frame: pd.DataFrame, item_embeddings: np.ndarray):
        if len(item_frame) != len(item_embeddings):
            raise ValueError("Item frame length must match embedding count.")

        self.item_frame = item_frame.reset_index(drop=True)
        self.item_embeddings = item_embeddings.astype("float32").copy()
        faiss.normalize_L2(self.item_embeddings)

        dimension = self.item_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.item_embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchHit]:
        query = query_embedding.reshape(1, -1).astype("float32").copy()
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            row = self.item_frame.iloc[idx]
            hits.append(
                SearchHit(
                    rank=rank,
                    item_id=row["itemId"],
                    score=float(score),
                    item_name=row.get("item_name", ""),
                    combined_text=row.get("combined_text", ""),
                )
            )
        return hits


def baseline_search_dataframe(
    queries: pd.DataFrame,
    query_embeddings: np.ndarray,
    retriever: FaissRetriever,
    top_k: int = 10,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, row in queries.reset_index(drop=True).iterrows():
        hits = retriever.search(query_embeddings[idx], top_k=top_k)
        records.append(
            {
                "query": row["query"],
                "top10_item_ids": [hit.item_id for hit in hits],
                "top10_item_names": [hit.item_name for hit in hits],
                "top10_scores": [hit.score for hit in hits],
            }
        )
    return pd.DataFrame(records)


def load_hyde_embedding_bank(directory: str | Path, query_index: int) -> np.ndarray:
    return np.load(Path(directory) / f"{query_index}.npy")


def pooled_hyde_search_dataframe(
    queries: pd.DataFrame,
    hyde_embeddings_dir: str | Path,
    retriever: FaissRetriever,
    top_k: int = 10,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    hyde_dir = Path(hyde_embeddings_dir)

    for idx, row in queries.reset_index(drop=True).iterrows():
        candidate_hits: list[SearchHit] = []
        embeddings = load_hyde_embedding_bank(hyde_dir, idx)
        for embedding in embeddings:
            candidate_hits.extend(retriever.search(embedding, top_k=top_k))

        ranked = sorted(candidate_hits, key=lambda hit: hit.score, reverse=True)
        top_hits = ranked[:top_k]
        records.append(
            {
                "query": row["query"],
                "top10_item_ids": [hit.item_id for hit in top_hits],
                "top10_item_names": [hit.item_name for hit in top_hits],
                "top10_scores": [hit.score for hit in top_hits],
            }
        )

    return pd.DataFrame(records)
