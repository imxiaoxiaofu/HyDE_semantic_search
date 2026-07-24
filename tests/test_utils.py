from utils import (
    build_text_for_item,
    flatten_taxonomy,
    normalize_text,
    parse_literal_list,
)


def test_normalize_text_removes_html_and_extra_space() -> None:
    assert normalize_text("<b>  oi </b>\n mundo") == "oi mundo"


def test_flatten_taxonomy_orders_levels() -> None:
    taxonomy = {"l2": "FINAL", "l0": "ROOT", "l1": "MID"}
    assert flatten_taxonomy(taxonomy) == "ROOT/MID/FINAL"


def test_build_text_for_item_contains_core_fields() -> None:
    metadata = {
        "name": "Batata Frita",
        "category_name": "Porções",
        "taxonomy": {"l0": "PRATOS"},
        "description": "Porção grande",
        "tags": [{"key": "SIZE", "value": ["LARGE"]}],
        "organic": False,
        "vegan": False,
        "lacFree": True,
    }
    text = build_text_for_item(metadata)
    assert "Nome: Batata Frita" in text
    assert "Categoria: Porções" in text
    assert "Taxonomia: PRATOS" in text
    assert "Tags: SIZE=LARGE" in text
    assert "Sem lactose" in text


def test_parse_literal_list_parses_serialized_lists() -> None:
    assert parse_literal_list("['a', 'b']") == ["a", "b"]
