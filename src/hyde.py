from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from embeddings import encode_texts


DEFAULT_HYDE_MODEL = "gpt-4.1-mini"

PROMPT_TEMPLATE = """
Please write five food items that fit the description of the following query.
Example food item descriptions: {descriptions}
Query: {query}
Food Items:
""".strip()


class HyDEOutput(BaseModel):
    food_items: list[str] = Field(
        ...,
        description="A list of food descriptions that can fulfill the query.",
    )


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> list[str]:
        ...


@dataclass
class OpenAIHyDEGenerator:
    model_name: str = DEFAULT_HYDE_MODEL
    api_key: str | None = None
    temperature: float = 0.5

    def __post_init__(self) -> None:
        from openai import OpenAI

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required to generate HyDE documents.")
        self._client = OpenAI(api_key=key)

    def generate(self, prompt: str) -> list[str]:
        response = self._client.responses.parse(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
            text_format=HyDEOutput,
            temperature=self.temperature,
        )
        return response.output_parsed.food_items


def build_prompt(query: str, descriptions: list[str]) -> str:
    return PROMPT_TEMPLATE.format(query=query, descriptions=descriptions)


def generate_hyde_dataframe(
    queries: list[str],
    generator: TextGenerator,
    descriptions: list[str],
) -> pd.DataFrame:
    generated_docs: list[list[str]] = []
    for query in tqdm(queries, desc="Generating HyDE docs"):
        generated_docs.append(generator.generate(build_prompt(query, descriptions)))
    return pd.DataFrame({"query": queries, "generated_docs": generated_docs})


def write_hyde_embeddings(
    hyde_docs: pd.DataFrame,
    output_dir: str | Path,
    embedding_model,
    batch_size: int = 32,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, row in hyde_docs.reset_index(drop=True).iterrows():
        embeddings = encode_texts(
            list(row["generated_docs"]),
            model=embedding_model,
            batch_size=batch_size,
            show_progress=False,
        )
        path = output_path / f"{idx}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            import numpy as np

            np.save(handle, embeddings)
