#!/usr/bin/env python3
"""
Supplemental experiments for the course project.

Adds:
- early-round recall analysis
- batch redundancy comparison
- initial seed-size ablation
- second-dataset generalization check
"""

from __future__ import annotations

import csv
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ligand_vs_baselines import (
    build_feature_matrix,
    ensure_dir,
    run_condition,
)

DATASET_URLS = {
    "bace": [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    ],
    "bbbp": [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bbbp.csv",
    ],
}


def download_dataset(data_dir: Path, dataset_name: str) -> Path:
    ensure_dir(data_dir)
    target = data_dir / f"{dataset_name}.csv"
    if target.exists():
        return target

    last_error = None
    for url in DATASET_URLS[dataset_name]:
        try:
            print(f"Downloading {dataset_name.upper()} from {url}")
            urllib.request.urlretrieve(url, target)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()

    raise RuntimeError(f"Failed to download dataset {dataset_name}: {last_error}")


def load_dataset(data_dir: Path, dataset_name: str):
    csv_path = download_dataset(data_dir, dataset_name)
    df = pd.read_csv(csv_path)
    x, y = build_feature_matrix(df)
    print(f"Loaded {dataset_name.upper()} with {len(y)} molecules; prevalence={y.mean():.3f}")
    return x, y


def metric_at_budget(result: Dict[str, object], target_budget: int) -> float:
    counts = result["evaluated_counts"]
    values = result["mean_recall"]
    idx = min(range(len(counts)), key=lambda i: abs(int(counts[i]) - target_budget))
    return float(values[idx])


def mean_query_redundancy(result: Dict[str, object]) -> float:
    values = result["mean_batch_redundancy"]
    if len(values) <= 1:
        return float(values[-1])
    return float(sum(values[1:]) / len(values[1:]))


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_supplemental_experiments(data_dir: Path, output_dir: Path, n_seeds: int = 3) -> None:
    ensure_dir(output_dir)

    bace_x, bace_y = load_dataset(data_dir, "bace")

    core_configs = {
        "Random sampling": {"model_name": "rf", "query_name": "random"},
        "Uncertainty sampling": {"model_name": "rf", "query_name": "uncertainty"},
        "Diversity-aware uncertainty": {"model_name": "rf", "query_name": "diversity"},
    }

    core_results = {}
    for label, cfg in core_configs.items():
        print(f"Running supplemental core config: {label}")
        core_results[label] = run_condition(
            x=bace_x,
            y=bace_y,
            model_name=cfg["model_name"],
            query_name=cfg["query_name"],
            init_name="random",
            n0=50,
            budget=500,
            batch_size=25,
            n_seeds=n_seeds,
        )

    early_rows = []
    for label, result in core_results.items():
        row = {"method": label}
        for target in [100, 200, 300]:
            row[f"recall_at_{target}"] = f"{metric_at_budget(result, target):.4f}"
        early_rows.append(row)
    write_csv(early_rows, output_dir / "early_round_analysis.csv")

    redundancy_rows = []
    for label, result in core_results.items():
        redundancy_rows.append(
            {
                "method": label,
                "mean_query_redundancy": f"{mean_query_redundancy(result):.4f}",
                "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                "mean_ausc": f"{float(result['mean_ausc']):.4f}",
            }
        )
    write_csv(redundancy_rows, output_dir / "redundancy_analysis.csv")

    seed_size_rows = []
    for n0 in [25, 50, 100]:
        for init_name in ["random", "maxmin"]:
            label = f"n0={n0}, init={init_name}"
            print(f"Running seed-size ablation: {label}")
            result = run_condition(
                x=bace_x,
                y=bace_y,
                model_name="rf",
                query_name="diversity",
                init_name=init_name,
                n0=n0,
                budget=500,
                batch_size=25,
                n_seeds=n_seeds,
            )
            seed_size_rows.append(
                {
                    "condition": label,
                    "mean_ausc": f"{float(result['mean_ausc']):.4f}",
                    "std_ausc": f"{float(result['std_ausc']):.4f}",
                    "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                    "final_pr_auc": f"{float(result['mean_pr_auc'][-1]):.4f}",
                }
            )
    write_csv(seed_size_rows, output_dir / "seed_size_ablation.csv")

    bbbp_x, bbbp_y = load_dataset(data_dir, "bbbp")
    generalization_rows = []
    for label, cfg in core_configs.items():
        print(f"Running generalization experiment on BBBP: {label}")
        result = run_condition(
            x=bbbp_x,
            y=bbbp_y,
            model_name=cfg["model_name"],
            query_name=cfg["query_name"],
            init_name="random",
            n0=50,
            budget=500,
            batch_size=25,
            n_seeds=n_seeds,
        )
        generalization_rows.append(
            {
                "dataset": "BBBP",
                "method": label,
                "mean_ausc": f"{float(result['mean_ausc']):.4f}",
                "std_ausc": f"{float(result['std_ausc']):.4f}",
                "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                "final_roc_auc": f"{float(result['mean_roc_auc'][-1]):.4f}",
                "final_pr_auc": f"{float(result['mean_pr_auc'][-1]):.4f}",
            }
        )
    write_csv(generalization_rows, output_dir / "second_dataset_generalization.csv")


def main():
    run_supplemental_experiments(
        data_dir=Path("data"),
        output_dir=Path("results_course_project"),
        n_seeds=3,
    )


if __name__ == "__main__":
    main()
