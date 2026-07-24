from __future__ import annotations

import argparse

import pandas as pd

from config import ProjectPaths
from evaluation import OpenAISatisfactionJudge, evaluate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and HyDE search results.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.project_root)
    judge = OpenAISatisfactionJudge()

    baseline = pd.read_csv(paths.baseline_results_path)
    hyde = pd.read_csv(paths.hyde_results_path)

    baseline_eval = evaluate_results(baseline, judge)
    hyde_eval = evaluate_results(hyde, judge)

    baseline_eval.to_csv(paths.traditional_eval_path, index=False)
    hyde_eval.to_csv(paths.hyde_eval_path, index=False)

    print(f"Saved baseline evaluation to {paths.traditional_eval_path}")
    print(f"Saved HyDE evaluation to {paths.hyde_eval_path}")


if __name__ == "__main__":
    main()
