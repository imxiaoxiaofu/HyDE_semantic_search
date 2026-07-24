from __future__ import annotations

import ast
import json
import re
from typing import Any


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected a list-like value, received {type(value)!r}")


def flatten_taxonomy(taxonomy: dict[str, Any] | Any) -> str:
    if not isinstance(taxonomy, dict):
        return ""

    try:
        sorted_keys = sorted(taxonomy.keys(), key=lambda key: int(key.lstrip("l")))
    except ValueError:
        sorted_keys = sorted(taxonomy.keys())

    levels: list[str] = []
    for key in sorted_keys:
        normalized = normalize_text(taxonomy.get(key, ""))
        if normalized:
            levels.append(normalized)

    return "/".join(levels)


def build_text_for_item(metadata: dict[str, Any] | Any) -> str:
    if not isinstance(metadata, dict):
        return ""

    name = normalize_text(metadata.get("name", ""))
    category = normalize_text(metadata.get("category_name", ""))
    taxonomy_text = flatten_taxonomy(metadata.get("taxonomy", {}))
    description = normalize_text(metadata.get("description", ""))

    tag_parts: list[str] = []
    for tag in metadata.get("tags", []):
        if not isinstance(tag, dict):
            continue
        key = normalize_text(tag.get("key", ""))
        values = tag.get("value", [])
        if isinstance(values, list):
            normalized_values = [normalize_text(item) for item in values]
            value_text = "|".join(item for item in normalized_values if item)
        else:
            value_text = normalize_text(str(values))
        if key and value_text:
            tag_parts.append(f"{key}={value_text}")

    flags = [
        "Orgânico" if metadata.get("organic", False) else "Não orgânico",
        "Vegano" if metadata.get("vegan", False) else "Não vegano",
        "Sem lactose" if metadata.get("lacFree", False) else "Contém lactose",
    ]

    segments = [
        f"Nome: {name}" if name else "",
        f"Categoria: {category}" if category else "",
        f"Taxonomia: {taxonomy_text}" if taxonomy_text else "",
        f"Descrição: {description}" if description else "",
        f"Tags: {', '.join(tag_parts)}" if tag_parts else "",
        "; ".join(flags),
    ]
    return ". ".join(segment for segment in segments if segment)
