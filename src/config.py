from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized project paths used across the retrieval pipeline."""

    project_root: Path
    data_dir: Path
    output_dir: Path
    embeddings_dir: Path

    @classmethod
    def from_root(cls, project_root: str | Path) -> "ProjectPaths":
        root = Path(project_root).resolve()
        output_dir = root / "output"
        data_dir = output_dir
        embeddings_dir = output_dir / "hyde_embeddings"
        return cls(
            project_root=root,
            data_dir=data_dir,
            output_dir=output_dir,
            embeddings_dir=embeddings_dir,
        )

    @property
    def queries_path(self) -> Path:
        candidates = [
            self.data_dir / "queries.csv",
            self.baseline_results_path,
            self.hyde_generated_docs_path,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    @property
    def items_path(self) -> Path:
        candidates = [
            self.data_dir / "5k_items_curated.xlsx",
            self.data_dir / "5k_items_curated.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    @property
    def combined_text_path(self) -> Path:
        return self.output_dir / "combined_text.csv"

    @property
    def item_embeddings_path(self) -> Path:
        return self.output_dir / "item_embeddings_multilingual.npy"

    @property
    def query_embeddings_path(self) -> Path:
        return self.output_dir / "query_embeddings_multilingual.npy"

    @property
    def baseline_results_path(self) -> Path:
        return self.output_dir / "queries_with_top10_search.csv"

    @property
    def hyde_generated_docs_path(self) -> Path:
        return self.output_dir / "hyde_generated_docs.csv"

    @property
    def hyde_results_path(self) -> Path:
        return self.output_dir / "queries_with_top10_hyde_pooled_search.csv"

    @property
    def traditional_eval_path(self) -> Path:
        return self.output_dir / "traditional_search_satisfaction.csv"

    @property
    def hyde_eval_path(self) -> Path:
        return self.output_dir / "hyde_search_satisfaction.csv"

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
