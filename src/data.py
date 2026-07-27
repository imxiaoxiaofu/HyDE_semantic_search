from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import build_text_for_item, normalize_text, parse_json_like


def load_queries(path: str | Path) -> pd.DataFrame:
    queries = pd.read_csv(path)
    if "search_term_pt" not in queries.columns and "query" not in queries.columns:
        raise ValueError("Expected query data to contain a 'search_term_pt' or 'query' column.")
    queries = queries.copy()
    source_column = "search_term_pt" if "search_term_pt" in queries.columns else "query"
    queries["query"] = queries[source_column].apply(normalize_text)
    return queries


def load_items(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        items = pd.read_excel(path)
    else:
        items = pd.read_csv(path)
    required_columns = {"itemId", "itemMetadata", "itemProfile"}
    missing = required_columns - set(items.columns)
    if missing:
        raise ValueError(f"items dataset is missing required columns: {sorted(missing)}")

    items = items.copy()
    items["itemMetadata_dict"] = items["itemMetadata"].apply(parse_json_like)
    items["itemProfile_dict"] = items["itemProfile"].apply(parse_json_like)
    items["item_name"] = items["itemMetadata_dict"].apply(
        lambda metadata: metadata.get("name", "") if isinstance(metadata, dict) else ""
    )
    items["combined_text"] = items["itemMetadata_dict"].apply(build_text_for_item)
    return items


def export_combined_text(items: pd.DataFrame, output_path: str | Path) -> None:
    output = items.loc[:, ["itemId", "item_name", "combined_text"]]
    output.to_csv(output_path, index=False)


def unique_top_k(values: Iterable[str], k: int) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
        if len(unique_values) == k:
            break
    return unique_values
