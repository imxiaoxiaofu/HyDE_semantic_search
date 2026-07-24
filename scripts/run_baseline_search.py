from __future__ import annotations

import argparse

from semantic_search_assignment.config import ProjectPaths
from semantic_search_assignment.data import load_items, load_queries
from semantic_search_assignment.embeddings import (
    encode_texts,
    load_embedding_model,
    save_embeddings,
)
from semantic_search_assignment.retrieval import FaissRetriever, baseline_search_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline semantic search.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    paths.ensure_output_dirs()

    items = load_items(paths.items_path)
    queries = load_queries(paths.queries_path)
    model = load_embedding_model(args.model_name) if args.model_name else load_embedding_model()

    item_embeddings = encode_texts(items["combined_text"].tolist(), model, batch_size=args.batch_size)
    query_embeddings = encode_texts(queries["query"].tolist(), model, batch_size=args.batch_size)

    save_embeddings(paths.item_embeddings_path, item_embeddings)
    save_embeddings(paths.query_embeddings_path, query_embeddings)

    retriever = FaissRetriever(items, item_embeddings)
    results = baseline_search_dataframe(queries, query_embeddings, retriever, top_k=args.top_k)
    results.to_csv(paths.baseline_results_path, index=False)
    print(f"Saved baseline search results to {paths.baseline_results_path}")


if __name__ == "__main__":
    main()
