#!/usr/bin/env python3
"""
Deeper analysis for the active learning course project.

Focus areas:
- budget sensitivity on BACE
- uncertainty calibration and uncertainty-quality analysis
- BBBP failure analysis
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from ligand_vs_baselines import (
    build_feature_matrix,
    ensure_dir,
    make_model,
    predictive_uncertainty,
    run_condition,
)

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
            print(f"Downloading {dataset_name.upper()} from {url}")
            urllib.request.urlretrieve(url, target)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()

    raise RuntimeError(f"Failed to download dataset {dataset_name}: {last_error}")


def load_dataset(data_dir: Path, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(download_dataset(data_dir, dataset_name))
    x, y = build_feature_matrix(df)
    print(f"Loaded {dataset_name.upper()} with {len(y)} molecules; prevalence={y.mean():.3f}")
    return x, y


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, object]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if np.any(mask):
            rows.append(
                {
                    "bin": i,
                    "lower": f"{lo:.2f}",
                    "upper": f"{hi:.2f}",
                    "count": int(np.sum(mask)),
                    "mean_confidence": f"{np.mean(y_prob[mask]):.4f}",
                    "empirical_positive_rate": f"{np.mean(y_true[mask]):.4f}",
                }
            )
    return rows


def run_calibration_analysis(data_dir: Path, output_dir: Path) -> None:
    rows = []
    for dataset_name in ["bace", "bbbp"]:
        x, y = load_dataset(data_dir, dataset_name)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=0
        )
        for model_name in ["logreg", "rf"]:
            model = make_model(model_name, seed=0)
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

            rows.append(
                {
                    "dataset": dataset_name.upper(),
                    "model": model_name,
                    "roc_auc": f"{roc_auc_score(y_test, y_prob):.4f}",
                    "pr_auc": f"{average_precision_score(y_test, y_prob):.4f}",
                    "brier": f"{brier_score_loss(y_test, y_prob):.4f}",
                    "ece": f"{expected_calibration_error(y_test, y_prob):.4f}",
                }
            )

            bin_rows = calibration_bins(y_test, y_prob)
            for row in bin_rows:
                row["dataset"] = dataset_name.upper()
                row["model"] = model_name
            write_csv(bin_rows, output_dir / f"calibration_bins_{dataset_name}_{model_name}.csv")

    write_csv(rows, output_dir / "calibration_summary.csv")


def run_budget_sensitivity(data_dir: Path, output_dir: Path, n_seeds: int = 3) -> None:
    x, y = load_dataset(data_dir, "bace")
    rows = []
    configs = {
        "Random sampling": {"model_name": "rf", "query_name": "random"},
        "Uncertainty sampling": {"model_name": "rf", "query_name": "uncertainty"},
        "Diversity-aware uncertainty": {"model_name": "rf", "query_name": "diversity"},
    }
    for budget in [100, 200, 300, 500]:
        for label, cfg in configs.items():
            print(f"Running budget sensitivity: budget={budget}, method={label}")
            result = run_condition(
                x=x,
                y=y,
                model_name=cfg["model_name"],
                query_name=cfg["query_name"],
                init_name="random",
                n0=50,
                budget=budget,
                batch_size=25,
                n_seeds=n_seeds,
            )
            rows.append(
                {
                    "budget": budget,
                    "method": label,
                    "mean_ausc": f"{float(result['mean_ausc']):.4f}",
                    "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                    "final_roc_auc": f"{float(result['mean_roc_auc'][-1]):.4f}",
                    "final_pr_auc": f"{float(result['mean_pr_auc'][-1]):.4f}",
                }
            )
    write_csv(rows, output_dir / "budget_sensitivity.csv")


def uncertainty_quality_rows(
    dataset_name: str, x: np.ndarray, y: np.ndarray, model_name: str
) -> Dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=0
    )
    model = make_model(model_name, seed=0)
    model.fit(x_train, y_train)
    probs, uncertainty = predictive_uncertainty(model, x_test)
    abs_error = np.abs(y_test - probs)
    correctness = (y_test == (probs >= 0.5).astype(int)).astype(float)

    uncertainty_error_corr = float(np.corrcoef(uncertainty, abs_error)[0, 1])
    uncertainty_correctness_corr = float(np.corrcoef(uncertainty, correctness)[0, 1])

    positive_mask = y_test == 1
    negative_mask = y_test == 0

    top_k = min(100, len(y_test))
    top_uncertain = np.argsort(uncertainty)[-top_k:]
    top_hit_rate = float(np.mean(y_test[top_uncertain]))

    return {
        "dataset": dataset_name.upper(),
        "model": model_name,
        "mean_uncertainty_positive": f"{float(np.mean(uncertainty[positive_mask])):.4f}",
        "mean_uncertainty_negative": f"{float(np.mean(uncertainty[negative_mask])):.4f}",
        "uncertainty_error_corr": f"{uncertainty_error_corr:.4f}",
        "uncertainty_correctness_corr": f"{uncertainty_correctness_corr:.4f}",
        "top100_uncertain_hit_rate": f"{top_hit_rate:.4f}",
        "dataset_prevalence": f"{float(np.mean(y_test)):.4f}",
    }


def run_failure_analysis(data_dir: Path, output_dir: Path, n_seeds: int = 3) -> None:
    rows = []
    for dataset_name in ["bace", "bbbp"]:
        x, y = load_dataset(data_dir, dataset_name)
        for model_name in ["logreg", "rf"]:
            rows.append(uncertainty_quality_rows(dataset_name, x, y, model_name))
    write_csv(rows, output_dir / "uncertainty_quality.csv")

    bbbp_x, bbbp_y = load_dataset(data_dir, "bbbp")
    core_configs = {
        "Random sampling": {"model_name": "rf", "query_name": "random"},
        "Uncertainty sampling": {"model_name": "rf", "query_name": "uncertainty"},
        "Diversity-aware uncertainty": {"model_name": "rf", "query_name": "diversity"},
    }

    early_rows = []
    hit_rows = []
    for label, cfg in core_configs.items():
        print(f"Running BBBP failure analysis core config: {label}")
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
        counts = result["evaluated_counts"]
        recalls = result["mean_recall"]
        for target in [100, 200, 300]:
            idx = min(range(len(counts)), key=lambda i: abs(int(counts[i]) - target))
            early_rows.append(
                {
                    "method": label,
                    "budget": target,
                    "recall": f"{float(recalls[idx]):.4f}",
                }
            )

        hit_rows.append(
            {
                "method": label,
                "mean_batch_hit_rate": f"{float(np.mean(result['mean_hit_rate'][1:])):.4f}",
                "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
                "mean_ausc": f"{float(result['mean_ausc']):.4f}",
            }
        )

    write_csv(early_rows, output_dir / "bbbp_early_round_analysis.csv")
    write_csv(hit_rows, output_dir / "bbbp_hit_rate_analysis.csv")


def main():
    data_dir = Path("data")
    output_dir = Path("results_course_project/deep_dive")
    ensure_dir(output_dir)

    run_budget_sensitivity(data_dir, output_dir, n_seeds=3)
    run_calibration_analysis(data_dir, output_dir)
    run_failure_analysis(data_dir, output_dir, n_seeds=3)


if __name__ == "__main__":
    main()
