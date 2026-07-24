"""Semantic search assignment package.

This package extracts the reusable project logic from the exploratory
notebooks so the retrieval pipeline can be run as code instead of only
through notebook cells.
"""

from .config import ProjectPaths

__all__ = ["ProjectPaths"]
