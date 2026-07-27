from __future__ import annotations

import argparse

import pandas as pd

from config import ProjectPaths
from embeddings import load_embedding_model
from hyde import (
    OpenAIHyDEGenerator,
    generate_hyde_dataframe,
    write_hyde_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HyDE documents and embeddings.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--few-shot-path",
        required=True,
        help="CSV or TXT file containing example descriptions used in the prompt.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    paths.ensure_output_dirs()

    query_frame = pd.read_csv(paths.queries_path)
    query_column = "search_term_pt" if "search_term_pt" in query_frame.columns else "query"
    queries = query_frame[query_column].tolist()

    few_shot_path = paths.project_root / args.few_shot_path
    if few_shot_path.suffix.lower() == ".csv":
        descriptions = pd.read_csv(few_shot_path).iloc[:, 0].dropna().astype(str).tolist()
    else:
        descriptions = [line.strip() for line in few_shot_path.read_text().splitlines() if line.strip()]

    generator = OpenAIHyDEGenerator(model_name=args.model_name or "gpt-4.1-mini")
    hyde_df = generate_hyde_dataframe(queries, generator, descriptions)
    hyde_df.to_csv(paths.hyde_generated_docs_path, index=False)

    embedding_model = load_embedding_model()
    write_hyde_embeddings(hyde_df, paths.embeddings_dir, embedding_model, batch_size=args.batch_size)
    print(f"Saved HyDE docs to {paths.hyde_generated_docs_path}")
    print(f"Saved HyDE embeddings to {paths.embeddings_dir}")


if __name__ == "__main__":
    main()
