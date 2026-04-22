#!/usr/bin/env python3
"""
make_report_outputs.py

Generate report-ready tables and plots from the bandit-vs-fixed experiment outputs.

Expected input files:
    ./results_bandit_vs_fixed_scaffold/*_bandit_vs_fixed_results.csv
Optional input files for learning curves:
    ./results_bandit_vs_fixed_scaffold/*_bandit_vs_fixed_curves.csv

What this script outputs:
1. summary_table.csv
   - one row per dataset
   - alpha=0 baseline
   - best fixed alpha
   - bandit
2. alpha_sweep_normausc.png
   - NormAUSC vs alpha, one line per dataset
3. alpha_sweep_ef.png
   - EF vs alpha, one line per dataset
4. comparison_bar_normausc.png
   - alpha=0 vs best fixed vs bandit, grouped by dataset
5. comparison_bar_hits.png
   - final hits, grouped by dataset
6. comparison_bar_ef.png
   - EF, grouped by dataset
7. learning_curve_<dataset>.png
   - recall vs evaluated budget for:
       alpha=0, best fixed alpha, bandit
   - only produced if corresponding *_curves.csv exists

Usage:
    python make_report_outputs.py

Optional:
    python make_report_outputs.py --results-dir ./results_bandit_vs_fixed_scaffold
    python make_report_outputs.py --output-dir ./report_outputs
    python make_report_outputs.py --curve-datasets tox21,clintox
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_alpha_value(x):
    if isinstance(x, str) and x.strip().lower() == "bandit":
        return np.nan
    return safe_float(x)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_best_fixed_row(df_dataset: pd.DataFrame) -> pd.Series:
    fixed = df_dataset[df_dataset["method"].isin(["exploitation", "fixed_alpha"])].copy()
    fixed["alpha_float"] = fixed["alpha"].apply(parse_alpha_value)
    fixed["mean_norm_ausc_float"] = fixed["mean_norm_ausc"].astype(float)
    fixed = fixed.sort_values(["mean_norm_ausc_float", "alpha_float"], ascending=[False, True])
    return fixed.iloc[0]


def pick_alpha0_row(df_dataset: pd.DataFrame) -> pd.Series:
    alpha0 = df_dataset[df_dataset["alpha"].astype(str) == "0.00"].copy()
    if alpha0.empty:
        alpha0 = df_dataset[
            df_dataset["alpha"].apply(lambda x: np.isclose(parse_alpha_value(x), 0.0, equal_nan=False))
        ].copy()
    if alpha0.empty:
        raise ValueError(f"No alpha=0 row found for dataset {df_dataset['dataset'].iloc[0]}")
    return alpha0.iloc[0]


def pick_bandit_row(df_dataset: pd.DataFrame) -> Optional[pd.Series]:
    bandit = df_dataset[df_dataset["method"] == "bandit"].copy()
    if bandit.empty:
        return None
    return bandit.iloc[0]


def dataset_display_name(row: pd.Series) -> str:
    label = str(row.get("label_col", "")).strip()
    dataset = str(row.get("dataset", "")).strip()
    return f"{dataset} ({label})" if label and label != "nan" else dataset


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


# -----------------------------
# Summary table
# -----------------------------

def build_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset_name, group in results_df.groupby("dataset", sort=False):
        alpha0 = pick_alpha0_row(group)
        best_fixed = pick_best_fixed_row(group)
        bandit = pick_bandit_row(group)

        row = {
            "dataset": dataset_name,
            "label_col": alpha0.get("label_col", ""),
            "pool_prevalence": alpha0.get("pool_prevalence", ""),
            "alpha0_normausc": alpha0.get("mean_norm_ausc", ""),
            "alpha0_ef": alpha0.get("mean_ef", ""),
            "alpha0_hits": alpha0.get("mean_final_hits", ""),
            "best_fixed_alpha": best_fixed.get("alpha", ""),
            "best_fixed_normausc": best_fixed.get("mean_norm_ausc", ""),
            "best_fixed_ef": best_fixed.get("mean_ef", ""),
            "best_fixed_hits": best_fixed.get("mean_final_hits", ""),
            "best_fixed_method": best_fixed.get("method", ""),
        }

        if bandit is not None:
            row.update(
                {
                    "bandit_normausc": bandit.get("mean_norm_ausc", ""),
                    "bandit_ef": bandit.get("mean_ef", ""),
                    "bandit_hits": bandit.get("mean_final_hits", ""),
                    "bandit_final_alpha": bandit.get("mean_final_alpha", ""),
                }
            )
        else:
            row.update(
                {
                    "bandit_normausc": "",
                    "bandit_ef": "",
                    "bandit_hits": "",
                    "bandit_final_alpha": "",
                }
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    # nice ordering
    if not summary.empty:
        summary["best_fixed_normausc_float"] = summary["best_fixed_normausc"].astype(float)
        summary = summary.sort_values("best_fixed_normausc_float", ascending=False).drop(
            columns=["best_fixed_normausc_float"]
        )

    return summary


# -----------------------------
# Plotting
# -----------------------------

def plot_alpha_sweep_metric(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fixed = results_df[results_df["method"].isin(["exploitation", "fixed_alpha"])].copy()
    if fixed.empty:
        print(f"Skipping {output_path.name}: no fixed-alpha rows found.")
        return

    fixed["alpha_float"] = fixed["alpha"].apply(parse_alpha_value)
    fixed = fixed.dropna(subset=["alpha_float"]).copy()
    fixed[metric_col] = fixed[metric_col].astype(float)

    plt.figure(figsize=(8.5, 5.5))
    for dataset_name, group in fixed.groupby("dataset", sort=False):
        label = dataset_name
        if "label_col" in group.columns:
            first_label = str(group["label_col"].iloc[0])
            if first_label and first_label != "nan":
                label = f"{dataset_name} ({first_label})"
        group = group.sort_values("alpha_float")
        plt.plot(group["alpha_float"], group[metric_col], marker="o", label=label)

    plt.xlabel("Fixed exploration weight α")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_comparison_bars(
    summary_df: pd.DataFrame,
    alpha0_col: str,
    best_col: str,
    bandit_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if summary_df.empty:
        print(f"Skipping {output_path.name}: summary table is empty.")
        return

    labels = []
    alpha0_vals = []
    best_vals = []
    bandit_vals = []

    for _, row in summary_df.iterrows():
        label = str(row["dataset"])
        label_col = str(row.get("label_col", "")).strip()
        if label_col and label_col != "nan":
            label = f"{label}\n({label_col})"
        labels.append(label)

        alpha0_vals.append(safe_float(row[alpha0_col]))
        best_vals.append(safe_float(row[best_col]))
        bandit_vals.append(safe_float(row[bandit_col]))

    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(10.5, 5.5))
    plt.bar(x - width, alpha0_vals, width=width, label="α = 0")
    plt.bar(x, best_vals, width=width, label="Best fixed α")
    plt.bar(x + width, bandit_vals, width=width, label="Bandit")

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_learning_curve_for_dataset(
    curves_path: Path,
    results_df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    curves_df = pd.read_csv(curves_path)

    if curves_df.empty:
        print(f"Skipping {output_path.name}: empty curve file.")
        return

    dataset_results = results_df[results_df["dataset"] == dataset_name].copy()
    if dataset_results.empty:
        print(f"Skipping {output_path.name}: dataset {dataset_name} not found in results.")
        return

    best_fixed = pick_best_fixed_row(dataset_results)
    alpha0 = pick_alpha0_row(dataset_results)
    bandit = pick_bandit_row(dataset_results)

    best_alpha = str(best_fixed["alpha"])
    alpha0_alpha = str(alpha0["alpha"])

    plt.figure(figsize=(8.5, 5.5))

    # alpha=0
    curve_alpha0 = curves_df[
        (curves_df["method"] == alpha0["method"]) &
        (curves_df["alpha"].astype(str) == alpha0_alpha)
    ].copy()
    if not curve_alpha0.empty:
        curve_alpha0["evaluated"] = curve_alpha0["evaluated"].astype(int)
        curve_alpha0["mean_recall"] = curve_alpha0["mean_recall"].astype(float)
        plt.plot(
            curve_alpha0["evaluated"],
            curve_alpha0["mean_recall"],
            marker="o",
            label="α = 0",
        )

    # best fixed
    curve_best = curves_df[
        (curves_df["method"] == best_fixed["method"]) &
        (curves_df["alpha"].astype(str) == best_alpha)
    ].copy()
    if not curve_best.empty:
        curve_best["evaluated"] = curve_best["evaluated"].astype(int)
        curve_best["mean_recall"] = curve_best["mean_recall"].astype(float)
        plt.plot(
            curve_best["evaluated"],
            curve_best["mean_recall"],
            marker="o",
            label=f"Best fixed α = {best_alpha}",
        )

    # bandit
    if bandit is not None:
        curve_bandit = curves_df[curves_df["method"] == "bandit"].copy()
        if not curve_bandit.empty:
            curve_bandit["evaluated"] = curve_bandit["evaluated"].astype(int)
            curve_bandit["mean_recall"] = curve_bandit["mean_recall"].astype(float)
            bandit_alpha = safe_float(bandit["mean_final_alpha"])
            label = f"Bandit (final α ≈ {bandit_alpha:.2f})" if not np.isnan(bandit_alpha) else "Bandit"
            plt.plot(
                curve_bandit["evaluated"],
                curve_bandit["mean_recall"],
                marker="o",
                label=label,
            )

    label_col = str(dataset_results["label_col"].iloc[0]) if "label_col" in dataset_results.columns else ""
    title = f"Recall vs budget: {dataset_name}"
    if label_col and label_col != "nan":
        title += f" ({label_col})"

    plt.xlabel("Evaluated molecules")
    plt.ylabel("Mean recall")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready tables and plots.")
    parser.add_argument("--results-dir", type=str, default="./results_bandit_vs_fixed_scaffold")
    parser.add_argument("--output-dir", type=str, default="./report_outputs")
    parser.add_argument(
        "--curve-datasets",
        type=str,
        default="tox21,clintox",
        help="Comma-separated datasets for learning-curve plots, if curve CSVs exist.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    result_files = sorted(results_dir.glob("*_bandit_vs_fixed_results.csv"))
    if not result_files:
        raise FileNotFoundError(
            f"No '*_bandit_vs_fixed_results.csv' files found in {results_dir.resolve()}"
        )

    # Read and merge all result files
    all_results = []
    for f in result_files:
        df = pd.read_csv(f)
        if not df.empty:
            all_results.append(df)

    if not all_results:
        raise ValueError("All result files were empty.")

    results_df = pd.concat(all_results, ignore_index=True)

    # Save merged raw results too
    save_dataframe(results_df, output_dir / "all_results_merged.csv")

    # Build summary table
    summary_df = build_summary_table(results_df)
    save_dataframe(summary_df, output_dir / "summary_table.csv")

    # Plot 1: NormAUSC vs alpha
    plot_alpha_sweep_metric(
        results_df=results_df,
        metric_col="mean_norm_ausc",
        ylabel="Mean normalized AUSC",
        title="Normalized AUSC vs fixed exploration weight α",
        output_path=output_dir / "alpha_sweep_normausc.png",
    )

    # Plot 2: EF vs alpha
    plot_alpha_sweep_metric(
        results_df=results_df,
        metric_col="mean_ef",
        ylabel="Mean EF",
        title="EF vs fixed exploration weight α",
        output_path=output_dir / "alpha_sweep_ef.png",
    )

    # Plot 3: Bar plot for NormAUSC
    plot_comparison_bars(
        summary_df=summary_df,
        alpha0_col="alpha0_normausc",
        best_col="best_fixed_normausc",
        bandit_col="bandit_normausc",
        ylabel="Mean normalized AUSC",
        title="Baseline vs best fixed α vs bandit",
        output_path=output_dir / "comparison_bar_normausc.png",
    )

    # Plot 4: Bar plot for final hits
    plot_comparison_bars(
        summary_df=summary_df,
        alpha0_col="alpha0_hits",
        best_col="best_fixed_hits",
        bandit_col="bandit_hits",
        ylabel="Mean final hits",
        title="Final hits: baseline vs best fixed α vs bandit",
        output_path=output_dir / "comparison_bar_hits.png",
    )

    # Plot 5: Bar plot for EF
    plot_comparison_bars(
        summary_df=summary_df,
        alpha0_col="alpha0_ef",
        best_col="best_fixed_ef",
        bandit_col="bandit_ef",
        ylabel="Mean EF",
        title="EF: baseline vs best fixed α vs bandit",
        output_path=output_dir / "comparison_bar_ef.png",
    )

    # Plot 6+: learning curves, if curve files exist
    curve_datasets = [x.strip() for x in args.curve_datasets.split(",") if x.strip()]
    for dataset_name in curve_datasets:
        curve_candidates = sorted(results_dir.glob(f"{dataset_name}_*_bandit_vs_fixed_curves.csv"))
        if not curve_candidates:
            print(f"Skipping learning curve for {dataset_name}: no curve CSV found.")
            continue

        # If multiple files exist, just use the first one.
        curves_path = curve_candidates[0]
        plot_learning_curve_for_dataset(
            curves_path=curves_path,
            results_df=results_df,
            dataset_name=dataset_name,
            output_path=output_dir / f"learning_curve_{dataset_name}.png",
        )

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")
    print("Generated:")
    print("  - all_results_merged.csv")
    print("  - summary_table.csv")
    print("  - alpha_sweep_normausc.png")
    print("  - alpha_sweep_ef.png")
    print("  - comparison_bar_normausc.png")
    print("  - comparison_bar_hits.png")
    print("  - comparison_bar_ef.png")
    print("  - learning_curve_<dataset>.png (if curve files were found)")


if __name__ == "__main__":
    main()