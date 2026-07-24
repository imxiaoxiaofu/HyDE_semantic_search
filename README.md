# Semantic Search with HyDE

This repository compares two retrieval strategies for Portuguese food-item search:

- Baseline semantic search using direct query-to-item embeddings
- HyDE search using LLM-generated hypothetical item descriptions before retrieval

The project is now organized as a reusable Python package with scripts for reproducible runs instead of notebook-first workflow.

## What The Project Does

The pipeline takes a query dataset and a curated catalog of food items, then:

1. Normalizes and flattens item metadata into one searchable text field
2. Embeds both items and user queries with a multilingual sentence-transformer
3. Runs FAISS similarity search for a baseline retriever
4. Generates HyDE documents with OpenAI for each query
5. Embeds those generated documents and pools their retrieval results
6. Evaluates baseline vs. HyDE outputs with an LLM-based satisfaction check

## Project Structure

```text
src/semantic_search_assignment/
  config.py        # project paths and artifact locations
  data.py          # dataset loading and item text construction
  embeddings.py    # sentence-transformer loading and batch encoding
  retrieval.py     # FAISS retrieval for baseline and HyDE pooling
  hyde.py          # prompt building, HyDE generation, HyDE embeddings
  evaluation.py    # LLM-based search result evaluation
  utils.py         # shared parsing and text cleanup helpers

scripts/
  build_item_corpus.py
  run_baseline_search.py
  run_hyde_generation.py
  run_hyde_search.py
  run_evaluation.py
```

## Data Expectations

The code expects the original assignment data at:

```text
prosusai_assignment_data/
  queries.csv
  5k_items_curated.csv
  output/
```

Dataset files and generated outputs are intentionally not tracked in Git in this refactored version. Place the original input CSVs under `prosusai_assignment_data/` before running the pipeline locally.

## Setup With uv

```bash
uv sync --extra dev
```

Set your OpenAI key before running HyDE generation or evaluation:

```bash
export OPENAI_API_KEY=your_key_here
```

If you prefer a one-off command inside the project environment:

```bash
uv run python --version
```

## Run The Pipeline

You can run the scripts directly through `uv`:

Build the normalized item corpus:

```bash
uv run python scripts/build_item_corpus.py --project-root .
```

Run baseline retrieval:

```bash
uv run python scripts/run_baseline_search.py --project-root .
```

Generate HyDE documents and their embeddings:

```bash
uv run python scripts/run_hyde_generation.py --project-root . --few-shot-path path/to/few_shot_examples.txt
```

Run pooled HyDE retrieval:

```bash
uv run python scripts/run_hyde_search.py --project-root .
```

Evaluate baseline and HyDE outputs:

```bash
uv run python scripts/run_evaluation.py --project-root .
```

Run tests with:

```bash
uv run pytest
```

## Next Cleanup Steps

- Add a reproducible few-shot curation step for HyDE prompt examples
- Add unit tests for retrieval and evaluation helpers
- Add one polished demo surface, likely Streamlit
- Document final quantitative results and key example queries
