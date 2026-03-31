#!/usr/bin/env python3

from __future__ import annotations

import csv
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ligand_vs_baselines import build_feature_matrix, ensure_dir, run_condition

DATASET_URLS = {
    "bace": ["https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"],
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
            urllib.request.urlretrieve(url, target)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()
    raise RuntimeError(f"Failed to download dataset {dataset_name}: {last_error}")


def load_dataset(data_dir: Path, dataset_name: str):
    df = pd.read_csv(download_dataset(data_dir, dataset_name))
    x, y = build_feature_matrix(df)
    print(f"Loaded {dataset_name.upper()} with {len(y)} molecules; prevalence={y.mean():.3f}")
    return x, y


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_dir = Path("data")
    output_dir = Path("results_course_project/mlp_extension")
    ensure_dir(output_dir)

    configs = [
        ("RF diversity-hit-seeking (alpha=0.25)", "rf", "diversity_hit_seeking", {"alpha": 0.25}),
        ("MLP uncertainty", "mlp", "uncertainty", {}),
        ("MLP hit-seeking (alpha=0.25)", "mlp", "hit_seeking", {"alpha": 0.25}),
    ]

    rows = []
    for dataset_name in ["bace", "bbbp"]:
        x, y = load_dataset(data_dir, dataset_name)
        for label, model_name, query_name, extra in configs:
            print(f"Running {dataset_name.upper()} MLP comparison: {label}")
            result = run_condition(
                x=x,
                y=y,
                model_name=model_name,
                query_name=query_name,
                init_name="random",
                n0=50,
                budget=500,
                batch_size=25,
                n_seeds=1,
                **extra,
            )
            rows.append(
                {
                    "dataset": dataset_name.upper(),
                    "method": label,
                    "mean_ausc": f"{float(result['mean_ausc']):.4f}",
                    "std_ausc": f"{float(result['std_ausc']):.4f}",
                    "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                    "final_roc_auc": f"{float(result['mean_roc_auc'][-1]):.4f}",
                    "final_pr_auc": f"{float(result['mean_pr_auc'][-1]):.4f}",
                }
            )

    write_csv(rows, output_dir / "mlp_extension_results.csv")


if __name__ == "__main__":
    main()
