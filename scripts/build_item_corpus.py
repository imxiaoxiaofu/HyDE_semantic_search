from __future__ import annotations

import argparse

from config import ProjectPaths
from data import export_combined_text, load_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized item text corpus.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    paths.ensure_output_dirs()

    items = load_items(paths.items_path)
    export_combined_text(items, paths.combined_text_path)
    print(f"Saved combined text corpus to {paths.combined_text_path}")


if __name__ == "__main__":
    main()
