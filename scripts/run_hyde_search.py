from __future__ import annotations

import argparse

from config import ProjectPaths
from data import load_items, load_queries
from embeddings import load_embeddings
from retrieval import FaissRetriever, pooled_hyde_search_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pooled HyDE retrieval.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    items = load_items(paths.items_path)
    queries = load_queries(paths.queries_path)
    item_embeddings = load_embeddings(paths.item_embeddings_path)

    retriever = FaissRetriever(items, item_embeddings)
    hyde_results = pooled_hyde_search_dataframe(
        queries,
        paths.embeddings_dir,
        retriever,
        top_k=args.top_k,
    )
    hyde_results.to_csv(paths.hyde_results_path, index=False)
    print(f"Saved HyDE search results to {paths.hyde_results_path}")


if __name__ == "__main__":
    main()
