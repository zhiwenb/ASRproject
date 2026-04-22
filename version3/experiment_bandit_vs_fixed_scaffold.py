#!/usr/bin/env python3
"""
experiment_bandit_vs_fixed_scaffold.py

Compare on scaffold split:
1. Pure exploitation baseline (alpha = 0)
2. Best fixed alpha from a sweep
3. Bandit adaptive alpha (UCB over candidate alpha arms)

Metrics:
- Normalized AUSC (main)
- EF at final budget
- Final hits / recall
- Final ROC-AUC / PR-AUC
- Mean final alpha chosen by bandit

Uses:
- local datasets in ./data
- scaffold split
- rank-based acquisition scores

Example:
    python experiment_bandit_vs_fixed_scaffold.py --dataset tox21 --label-col NR-AR
    python experiment_bandit_vs_fixed_scaffold.py --dataset muv --label-col MUV-466
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset_loader_scaffold import (
    build_features,
    initialize_pool_indices,
    load_dataset_dataframe,
    make_active_learning_pool_from_scaffold_split,
)


# -----------------------------
# Model / scoring helpers
# -----------------------------

def make_model(seed: int, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def predict_proba_and_entropy(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(x)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return probs, entropy


def rank_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / (n - 1)


def compute_scores(probs: np.ndarray, uncertainty: np.ndarray, alpha: float) -> np.ndarray:
    rank_p = rank_scale(probs)
    rank_u = rank_scale(uncertainty)
    return (1.0 - alpha) * rank_p + alpha * rank_u


def select_top_batch(scores: np.ndarray, remaining: np.ndarray, batch_size: int) -> np.ndarray:
    chosen_local = np.argsort(scores)[-batch_size:]
    return remaining[chosen_local]


def safe_auc_scores(model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_test)) < 2:
        return np.nan, np.nan
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs)), float(average_precision_score(y_test, probs))


# -----------------------------
# Metric helpers
# -----------------------------

def compute_ausc(recall_curve: np.ndarray, evaluated_counts: np.ndarray) -> float:
    if len(recall_curve) <= 1:
        return float(recall_curve[-1]) if len(recall_curve) else 0.0
    x = evaluated_counts.astype(float)
    x = x - x.min()
    if x.max() <= 0:
        return float(np.mean(recall_curve))
    return float(np.trapezoid(recall_curve, x / x.max()))


def random_recall_curve(evaluated_counts: np.ndarray, n_pool: int) -> np.ndarray:
    return evaluated_counts.astype(float) / float(n_pool)


def compute_normalized_ausc(recall_curve: np.ndarray, evaluated_counts: np.ndarray, n_pool: int) -> float:
    method_ausc = compute_ausc(recall_curve, evaluated_counts)
    rand_curve = random_recall_curve(evaluated_counts, n_pool)
    rand_ausc = compute_ausc(rand_curve, evaluated_counts)
    if rand_ausc <= 1e-12:
        return np.nan
    return method_ausc / rand_ausc


def compute_ef(final_hits: int, budget: int, total_actives: int, n_pool: int) -> float:
    if budget <= 0 or total_actives <= 0 or n_pool <= 0:
        return np.nan
    prevalence = total_actives / n_pool
    if prevalence <= 0:
        return np.nan
    return (final_hits / budget) / prevalence


def pad_curves(curves: Sequence[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    return np.asarray([np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curves])


# -----------------------------
# Bandit helpers
# -----------------------------

def parse_alphas(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def choose_bandit_alpha(
    arms: np.ndarray,
    arm_counts: np.ndarray,
    arm_rewards: np.ndarray,
    t: int,
    exploration: float,
) -> Tuple[int, float]:
    untried = np.flatnonzero(arm_counts == 0)
    if len(untried) > 0:
        arm_idx = int(untried[0])
        return arm_idx, float(arms[arm_idx])

    mean_rewards = arm_rewards / np.maximum(arm_counts, 1.0)
    bonus = exploration * np.sqrt(np.log(max(t, 2)) / arm_counts)
    arm_idx = int(np.argmax(mean_rewards + bonus))
    return arm_idx, float(arms[arm_idx])


# -----------------------------
# One-seed runs
# -----------------------------

def run_one_seed_fixed(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    seed: int,
    n0: int,
    budget: int,
    batch_size: int,
    rf_estimators: int,
) -> Dict[str, np.ndarray | float]:
    n_pool = len(y_pool)
    budget = min(budget, n_pool)
    if budget <= n0:
        raise ValueError(f"budget={budget} must be > n0={n0}")

    selected_idx = initialize_pool_indices(y_pool, n0=n0, seed=seed)
    unlabeled_mask = np.ones(n_pool, dtype=bool)
    unlabeled_mask[selected_idx] = False

    selected_labels = y_pool[selected_idx].copy()
    total_actives = int(np.sum(y_pool == 1))

    evaluated_counts: List[int] = [len(selected_idx)]
    recall_vals: List[float] = [float(np.sum(selected_labels) / total_actives) if total_actives > 0 else 0.0]
    batch_hit_rates: List[float] = [float(np.mean(selected_labels))]
    alpha_curve: List[float] = [alpha]
    roc_vals: List[float] = []
    pr_vals: List[float] = []

    model = make_model(seed=seed, n_estimators=rf_estimators)
    model.fit(x_pool[selected_idx], y_pool[selected_idx])
    roc, pr = safe_auc_scores(model, x_test, y_test)
    roc_vals.append(roc)
    pr_vals.append(pr)

    while len(selected_idx) < budget and np.any(unlabeled_mask):
        remaining = np.flatnonzero(unlabeled_mask)
        current_batch_size = min(batch_size, budget - len(selected_idx), len(remaining))

        probs, uncertainty = predict_proba_and_entropy(model, x_pool[remaining])
        scores = compute_scores(probs, uncertainty, alpha=alpha)
        chosen = select_top_batch(scores, remaining, current_batch_size)

        batch_labels = y_pool[chosen]
        selected_idx = np.concatenate([selected_idx, chosen])
        unlabeled_mask[chosen] = False
        selected_labels = np.concatenate([selected_labels, batch_labels])

        evaluated_counts.append(len(selected_idx))
        recall_vals.append(float(np.sum(selected_labels) / total_actives) if total_actives > 0 else 0.0)
        batch_hit_rates.append(float(np.mean(batch_labels)))
        alpha_curve.append(alpha)

        model = make_model(seed=seed, n_estimators=rf_estimators)
        model.fit(x_pool[selected_idx], y_pool[selected_idx])
        roc, pr = safe_auc_scores(model, x_test, y_test)
        roc_vals.append(roc)
        pr_vals.append(pr)

    evaluated_counts_arr = np.asarray(evaluated_counts, dtype=int)
    recall_curve_arr = np.asarray(recall_vals, dtype=float)
    final_hits = int(np.sum(selected_labels))
    final_budget = int(len(selected_labels))

    return {
        "evaluated_counts": evaluated_counts_arr,
        "recall_curve": recall_curve_arr,
        "batch_hit_rate_curve": np.asarray(batch_hit_rates, dtype=float),
        "alpha_curve": np.asarray(alpha_curve, dtype=float),
        "roc_auc_curve": np.asarray(roc_vals, dtype=float),
        "pr_auc_curve": np.asarray(pr_vals, dtype=float),
        "norm_ausc": float(compute_normalized_ausc(recall_curve_arr, evaluated_counts_arr, n_pool=n_pool)),
        "ef": float(compute_ef(final_hits, final_budget, total_actives, n_pool)),
        "final_hits": float(final_hits),
        "final_recall": float(recall_curve_arr[-1]),
        "final_roc_auc": float(roc_vals[-1]),
        "final_pr_auc": float(pr_vals[-1]),
        "final_alpha": float(alpha_curve[-1]),
    }


def run_one_seed_bandit(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    bandit_arms: Sequence[float],
    bandit_exploration: float,
    seed: int,
    n0: int,
    budget: int,
    batch_size: int,
    rf_estimators: int,
) -> Dict[str, np.ndarray | float]:
    n_pool = len(y_pool)
    budget = min(budget, n_pool)
    if budget <= n0:
        raise ValueError(f"budget={budget} must be > n0={n0}")

    arms = np.asarray(list(bandit_arms), dtype=float)
    if len(arms) == 0:
        raise ValueError("bandit_arms must contain at least one alpha.")

    selected_idx = initialize_pool_indices(y_pool, n0=n0, seed=seed)
    unlabeled_mask = np.ones(n_pool, dtype=bool)
    unlabeled_mask[selected_idx] = False

    selected_labels = y_pool[selected_idx].copy()
    total_actives = int(np.sum(y_pool == 1))

    arm_counts = np.zeros(len(arms), dtype=float)
    arm_rewards = np.zeros(len(arms), dtype=float)
    current_arm_idx, current_alpha = choose_bandit_alpha(
        arms=arms,
        arm_counts=arm_counts,
        arm_rewards=arm_rewards,
        t=1,
        exploration=bandit_exploration,
    )

    evaluated_counts: List[int] = [len(selected_idx)]
    recall_vals: List[float] = [float(np.sum(selected_labels) / total_actives) if total_actives > 0 else 0.0]
    batch_hit_rates: List[float] = [float(np.mean(selected_labels))]
    alpha_curve: List[float] = [current_alpha]
    roc_vals: List[float] = []
    pr_vals: List[float] = []

    model = make_model(seed=seed, n_estimators=rf_estimators)
    model.fit(x_pool[selected_idx], y_pool[selected_idx])
    roc, pr = safe_auc_scores(model, x_test, y_test)
    roc_vals.append(roc)
    pr_vals.append(pr)

    while len(selected_idx) < budget and np.any(unlabeled_mask):
        remaining = np.flatnonzero(unlabeled_mask)
        current_batch_size = min(batch_size, budget - len(selected_idx), len(remaining))

        probs, uncertainty = predict_proba_and_entropy(model, x_pool[remaining])
        scores = compute_scores(probs, uncertainty, alpha=current_alpha)
        chosen = select_top_batch(scores, remaining, current_batch_size)

        batch_labels = y_pool[chosen]
        batch_hit_rate = float(np.mean(batch_labels))

        selected_idx = np.concatenate([selected_idx, chosen])
        unlabeled_mask[chosen] = False
        selected_labels = np.concatenate([selected_labels, batch_labels])

        # Update reward for the arm just used
        arm_counts[current_arm_idx] += 1.0
        arm_rewards[current_arm_idx] += batch_hit_rate

        # Choose next alpha
        current_arm_idx, current_alpha = choose_bandit_alpha(
            arms=arms,
            arm_counts=arm_counts,
            arm_rewards=arm_rewards,
            t=int(np.sum(arm_counts) + 1),
            exploration=bandit_exploration,
        )

        evaluated_counts.append(len(selected_idx))
        recall_vals.append(float(np.sum(selected_labels) / total_actives) if total_actives > 0 else 0.0)
        batch_hit_rates.append(batch_hit_rate)
        alpha_curve.append(current_alpha)

        model = make_model(seed=seed, n_estimators=rf_estimators)
        model.fit(x_pool[selected_idx], y_pool[selected_idx])
        roc, pr = safe_auc_scores(model, x_test, y_test)
        roc_vals.append(roc)
        pr_vals.append(pr)

    evaluated_counts_arr = np.asarray(evaluated_counts, dtype=int)
    recall_curve_arr = np.asarray(recall_vals, dtype=float)
    final_hits = int(np.sum(selected_labels))
    final_budget = int(len(selected_labels))

    return {
        "evaluated_counts": evaluated_counts_arr,
        "recall_curve": recall_curve_arr,
        "batch_hit_rate_curve": np.asarray(batch_hit_rates, dtype=float),
        "alpha_curve": np.asarray(alpha_curve, dtype=float),
        "roc_auc_curve": np.asarray(roc_vals, dtype=float),
        "pr_auc_curve": np.asarray(pr_vals, dtype=float),
        "norm_ausc": float(compute_normalized_ausc(recall_curve_arr, evaluated_counts_arr, n_pool=n_pool)),
        "ef": float(compute_ef(final_hits, final_budget, total_actives, n_pool)),
        "final_hits": float(final_hits),
        "final_recall": float(recall_curve_arr[-1]),
        "final_roc_auc": float(roc_vals[-1]),
        "final_pr_auc": float(pr_vals[-1]),
        "final_alpha": float(alpha_curve[-1]),
    }


# -----------------------------
# Aggregation
# -----------------------------

def aggregate_seed_results(seed_results: List[Dict[str, np.ndarray | float]]) -> Dict[str, np.ndarray | float]:
    evaluated_counts = seed_results[0]["evaluated_counts"]
    recall = pad_curves([r["recall_curve"] for r in seed_results])  # type: ignore[arg-type]
    batch_hit = pad_curves([r["batch_hit_rate_curve"] for r in seed_results])  # type: ignore[arg-type]
    alpha_curve = pad_curves([r["alpha_curve"] for r in seed_results])  # type: ignore[arg-type]
    roc = pad_curves([r["roc_auc_curve"] for r in seed_results])  # type: ignore[arg-type]
    pr = pad_curves([r["pr_auc_curve"] for r in seed_results])  # type: ignore[arg-type]

    norm_ausc_values = np.asarray([r["norm_ausc"] for r in seed_results], dtype=float)
    ef_values = np.asarray([r["ef"] for r in seed_results], dtype=float)
    hits_values = np.asarray([r["final_hits"] for r in seed_results], dtype=float)
    recall_values = np.asarray([r["final_recall"] for r in seed_results], dtype=float)
    roc_values = np.asarray([r["final_roc_auc"] for r in seed_results], dtype=float)
    pr_values = np.asarray([r["final_pr_auc"] for r in seed_results], dtype=float)
    final_alpha_values = np.asarray([r["final_alpha"] for r in seed_results], dtype=float)

    return {
        "evaluated_counts": evaluated_counts,
        "mean_recall_curve": recall.mean(axis=0),
        "mean_batch_hit_rate_curve": batch_hit.mean(axis=0),
        "mean_alpha_curve": alpha_curve.mean(axis=0),
        "mean_roc_auc_curve": roc.mean(axis=0),
        "mean_pr_auc_curve": pr.mean(axis=0),
        "mean_norm_ausc": float(np.nanmean(norm_ausc_values)),
        "std_norm_ausc": float(np.nanstd(norm_ausc_values)),
        "mean_ef": float(np.nanmean(ef_values)),
        "std_ef": float(np.nanstd(ef_values)),
        "mean_final_hits": float(np.mean(hits_values)),
        "std_final_hits": float(np.std(hits_values)),
        "mean_final_recall": float(np.mean(recall_values)),
        "std_final_recall": float(np.std(recall_values)),
        "mean_final_roc_auc": float(np.nanmean(roc_values)),
        "mean_final_pr_auc": float(np.nanmean(pr_values)),
        "mean_final_alpha": float(np.nanmean(final_alpha_values)),
        "std_final_alpha": float(np.nanstd(final_alpha_values)),
    }


def run_fixed_condition(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    n_seeds: int,
    n0: int,
    budget: int,
    batch_size: int,
    rf_estimators: int,
) -> Dict[str, np.ndarray | float]:
    results = []
    for seed in range(n_seeds):
        results.append(
            run_one_seed_fixed(
                x_pool=x_pool,
                y_pool=y_pool,
                x_test=x_test,
                y_test=y_test,
                alpha=alpha,
                seed=seed,
                n0=n0,
                budget=budget,
                batch_size=batch_size,
                rf_estimators=rf_estimators,
            )
        )
    return aggregate_seed_results(results)


def run_bandit_condition(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    bandit_arms: Sequence[float],
    bandit_exploration: float,
    n_seeds: int,
    n0: int,
    budget: int,
    batch_size: int,
    rf_estimators: int,
) -> Dict[str, np.ndarray | float]:
    results = []
    for seed in range(n_seeds):
        results.append(
            run_one_seed_bandit(
                x_pool=x_pool,
                y_pool=y_pool,
                x_test=x_test,
                y_test=y_test,
                bandit_arms=bandit_arms,
                bandit_exploration=bandit_exploration,
                seed=seed,
                n0=n0,
                budget=budget,
                batch_size=batch_size,
                rf_estimators=rf_estimators,
            )
        )
    return aggregate_seed_results(results)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare alpha=0, best fixed alpha, and bandit adaptive alpha.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hiv", "bace", "bbbp", "clintox", "tox21", "muv"])
    parser.add_argument("--label-col", type=str, default=None,
                        help="Override default label column, useful for tox21/muv.")
    parser.add_argument("--output-dir", type=str, default="./results_bandit_vs_fixed_scaffold")
    parser.add_argument("--alphas", type=str, default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--bandit-exploration", type=float, default=0.35)
    parser.add_argument("--save-curves", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, task, chosen_label = load_dataset_dataframe(
        data_dir=Path(args.data_dir),
        dataset_key=args.dataset,
        label_col=args.label_col,
    )
    pool_df, test_df = make_active_learning_pool_from_scaffold_split(df, frac_pool=0.8, frac_test=0.2)

    x_pool, y_pool = build_features(pool_df)
    x_test, y_test = build_features(test_df)

    n_pool = len(y_pool)
    n_test = len(y_test)
    pool_actives = int(np.sum(y_pool))
    test_actives = int(np.sum(y_test))
    bandit_arms = sorted(set(parse_alphas(args.alphas)))

    print(f"\nDataset: {args.dataset}")
    print(f"Label column: {chosen_label}")
    print(f"Pool size: {n_pool}, pool actives: {pool_actives}, pool prevalence: {pool_actives / n_pool:.4f}")
    print(f"Test size: {n_test}, test actives: {test_actives}, test prevalence: {test_actives / n_test:.4f}")

    # 1) Fixed-alpha sweep
    sweep_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for alpha in bandit_arms:
        print(f"Running fixed alpha = {alpha:.2f}", flush=True)
        result = run_fixed_condition(
            x_pool=x_pool,
            y_pool=y_pool,
            x_test=x_test,
            y_test=y_test,
            alpha=alpha,
            n_seeds=args.n_seeds,
            n0=args.n0,
            budget=args.budget,
            batch_size=args.batch_size,
            rf_estimators=args.rf_estimators,
        )

        method_name = "exploitation" if np.isclose(alpha, 0.0) else "fixed_alpha"
        sweep_rows.append(
            {
                "dataset": args.dataset,
                "label_col": chosen_label,
                "split_type": "scaffold",
                "method": method_name,
                "alpha": f"{alpha:.2f}",
                "n_pool": n_pool,
                "pool_actives": pool_actives,
                "pool_prevalence": f"{pool_actives / n_pool:.6f}",
                "n_test": n_test,
                "test_actives": test_actives,
                "test_prevalence": f"{test_actives / n_test:.6f}",
                "mean_norm_ausc": f"{float(result['mean_norm_ausc']):.4f}",
                "std_norm_ausc": f"{float(result['std_norm_ausc']):.4f}",
                "mean_ef": f"{float(result['mean_ef']):.4f}",
                "std_ef": f"{float(result['std_ef']):.4f}",
                "mean_final_hits": f"{float(result['mean_final_hits']):.2f}",
                "std_final_hits": f"{float(result['std_final_hits']):.2f}",
                "mean_final_recall": f"{float(result['mean_final_recall']):.4f}",
                "std_final_recall": f"{float(result['std_final_recall']):.4f}",
                "mean_final_roc_auc": f"{float(result['mean_final_roc_auc']):.4f}",
                "mean_final_pr_auc": f"{float(result['mean_final_pr_auc']):.4f}",
                "mean_final_alpha": f"{float(result['mean_final_alpha']):.4f}",
                "std_final_alpha": f"{float(result['std_final_alpha']):.4f}",
            }
        )

        if args.save_curves:
            evaluated_counts = result["evaluated_counts"]  # type: ignore[assignment]
            mean_recall_curve = result["mean_recall_curve"]  # type: ignore[assignment]
            mean_batch_hit_rate_curve = result["mean_batch_hit_rate_curve"]  # type: ignore[assignment]
            mean_alpha_curve = result["mean_alpha_curve"]  # type: ignore[assignment]
            for i in range(len(evaluated_counts)):
                curve_rows.append(
                    {
                        "dataset": args.dataset,
                        "label_col": chosen_label,
                        "split_type": "scaffold",
                        "method": method_name,
                        "alpha": f"{alpha:.2f}",
                        "evaluated": int(evaluated_counts[i]),
                        "mean_recall": f"{float(mean_recall_curve[i]):.6f}",
                        "mean_batch_hit_rate": f"{float(mean_batch_hit_rate_curve[i]):.6f}",
                        "mean_alpha_used": f"{float(mean_alpha_curve[i]):.6f}",
                    }
                )

    sweep_df = pd.DataFrame(sweep_rows)

    # 2) Best fixed alpha
    sortable = sweep_df.copy()
    sortable["mean_norm_ausc_float"] = sortable["mean_norm_ausc"].astype(float)
    best_row = sortable.sort_values("mean_norm_ausc_float", ascending=False).iloc[0]
    best_fixed_alpha = float(best_row["alpha"])

    print(f"\nBest fixed alpha by normalized AUSC: {best_fixed_alpha:.2f}")

    # 3) Bandit
    print("Running bandit adaptive alpha", flush=True)
    bandit_result = run_bandit_condition(
        x_pool=x_pool,
        y_pool=y_pool,
        x_test=x_test,
        y_test=y_test,
        bandit_arms=bandit_arms,
        bandit_exploration=args.bandit_exploration,
        n_seeds=args.n_seeds,
        n0=args.n0,
        budget=args.budget,
        batch_size=args.batch_size,
        rf_estimators=args.rf_estimators,
    )

    comparison_rows = sweep_rows.copy()
    comparison_rows.append(
        {
            "dataset": args.dataset,
            "label_col": chosen_label,
            "split_type": "scaffold",
            "method": "bandit",
            "alpha": "bandit",
            "n_pool": n_pool,
            "pool_actives": pool_actives,
            "pool_prevalence": f"{pool_actives / n_pool:.6f}",
            "n_test": n_test,
            "test_actives": test_actives,
            "test_prevalence": f"{test_actives / n_test:.6f}",
            "mean_norm_ausc": f"{float(bandit_result['mean_norm_ausc']):.4f}",
            "std_norm_ausc": f"{float(bandit_result['std_norm_ausc']):.4f}",
            "mean_ef": f"{float(bandit_result['mean_ef']):.4f}",
            "std_ef": f"{float(bandit_result['std_ef']):.4f}",
            "mean_final_hits": f"{float(bandit_result['mean_final_hits']):.2f}",
            "std_final_hits": f"{float(bandit_result['std_final_hits']):.2f}",
            "mean_final_recall": f"{float(bandit_result['mean_final_recall']):.4f}",
            "std_final_recall": f"{float(bandit_result['std_final_recall']):.4f}",
            "mean_final_roc_auc": f"{float(bandit_result['mean_final_roc_auc']):.4f}",
            "mean_final_pr_auc": f"{float(bandit_result['mean_final_pr_auc']):.4f}",
            "mean_final_alpha": f"{float(bandit_result['mean_final_alpha']):.4f}",
            "std_final_alpha": f"{float(bandit_result['std_final_alpha']):.4f}",
        }
    )

    if args.save_curves:
        evaluated_counts = bandit_result["evaluated_counts"]  # type: ignore[assignment]
        mean_recall_curve = bandit_result["mean_recall_curve"]  # type: ignore[assignment]
        mean_batch_hit_rate_curve = bandit_result["mean_batch_hit_rate_curve"]  # type: ignore[assignment]
        mean_alpha_curve = bandit_result["mean_alpha_curve"]  # type: ignore[assignment]
        for i in range(len(evaluated_counts)):
            curve_rows.append(
                {
                    "dataset": args.dataset,
                    "label_col": chosen_label,
                    "split_type": "scaffold",
                    "method": "bandit",
                    "alpha": "bandit",
                    "evaluated": int(evaluated_counts[i]),
                    "mean_recall": f"{float(mean_recall_curve[i]):.6f}",
                    "mean_batch_hit_rate": f"{float(mean_batch_hit_rate_curve[i]):.6f}",
                    "mean_alpha_used": f"{float(mean_alpha_curve[i]):.6f}",
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    results_path = output_dir / f"{args.dataset}_{chosen_label}_bandit_vs_fixed_results.csv"
    comparison_df.to_csv(results_path, index=False)

    # Save a compact summary too
    summary_rows = []

    # alpha=0 row
    alpha0_row = sortable[np.isclose(sortable["alpha"].astype(float), 0.0)].iloc[0]
    summary_rows.append(alpha0_row.drop(labels=["mean_norm_ausc_float"]).to_dict())

    # best fixed row
    summary_rows.append(best_row.drop(labels=["mean_norm_ausc_float"]).to_dict())

    # bandit row
    summary_rows.append(comparison_rows[-1])

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.dataset}_{chosen_label}_bandit_vs_fixed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.save_curves:
        curves_df = pd.DataFrame(curve_rows)
        curves_path = output_dir / f"{args.dataset}_{chosen_label}_bandit_vs_fixed_curves.csv"
        curves_df.to_csv(curves_path, index=False)

    print("\nFinished.")
    print(f"Saved full results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    if args.save_curves:
        print(f"Saved curves to: {curves_path}")

    print("\nKey comparison:")
    print(summary_df[["method", "alpha", "mean_norm_ausc", "mean_ef", "mean_final_hits", "mean_final_alpha"]].to_string(index=False))


if __name__ == "__main__":
    main()