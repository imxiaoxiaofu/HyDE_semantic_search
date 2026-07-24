from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from data import unique_top_k
from utils import parse_literal_list


class SatisfactionJudge(Protocol):
    def judge(self, query_text: str, top_candidates: list[str]) -> str:
        ...


@dataclass
class OpenAISatisfactionJudge:
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for evaluation.")
        self._client = OpenAI(api_key=key)

    def judge(self, query_text: str, top_candidates: list[str]) -> str:
        prompt = (
            f"Query: {query_text}\n"
            "Top 3 candidate items:\n"
            f"1. {top_candidates[0]}\n"
            f"2. {top_candidates[1]}\n"
            f"3. {top_candidates[2]}\n\n"
            "If any item largely matches the main query intent, reply with exactly Yes or No."
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You only reply with Yes or No when evaluating search satisfaction.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().split()[0]
        return answer.title()


def prepare_top3_columns(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["top10_item_names"] = output["top10_item_names"].apply(parse_literal_list)
    output["top3_item_names"] = output["top10_item_names"].apply(
        lambda values: unique_top_k(values, k=3)
    )
    return output


def evaluate_results(results: pd.DataFrame, judge: SatisfactionJudge) -> pd.DataFrame:
    output = prepare_top3_columns(results)
    labels: list[str] = []
    for _, row in output.iterrows():
        candidates = list(row["top3_item_names"])
        if len(candidates) < 3:
            candidates.extend([""] * (3 - len(candidates)))
        labels.append(judge.judge(row["query"], candidates))
    output["gpt_satisfaction"] = labels
    return output
