import numpy as np
import pandas as pd

from retrieval import FaissRetriever, pooled_hyde_search_dataframe


def test_pooled_hyde_search_deduplicates_items(tmp_path) -> None:
    items = pd.DataFrame(
        {
            "itemId": ["item-a", "item-b", "item-c"],
            "item_name": ["A", "B", "C"],
            "combined_text": ["alpha", "beta", "gamma"],
        }
    )
    item_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype="float32",
    )
    hyde_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
        ],
        dtype="float32",
    )
    np.save(tmp_path / "0.npy", hyde_embeddings)

    retriever = FaissRetriever(items, item_embeddings)
    queries = pd.DataFrame({"query": ["alpha query"]})
    results = pooled_hyde_search_dataframe(queries, tmp_path, retriever, top_k=3)

    item_ids = results.loc[0, "top10_item_ids"]
    assert item_ids == ["item-a", "item-b", "item-c"]
