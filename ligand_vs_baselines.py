#!/usr/bin/env python3
"""
Course-project implementation of active learning for ligand-based virtual screening.

Scope aligned with `ligand_vs_project_proposal.md`:
- Primary dataset: BACE
- Representation: Morgan fingerprints
- Models: Logistic Regression, Random Forest
- Query strategies: random, uncertainty, diversity-aware uncertainty
- Ablations: batch size and initialization strategy

This script is intentionally narrower than the earlier notebook so the experiments
are feasible, consistent, and easy to explain in a course project report.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from ligand_vs_mlp_models import mlp_entropy_uncertainty, mlp_predict_proba, train_mlp

RDLogger.DisableLog("rdApp.*")

BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"


@dataclass
class ActiveLearningResult:
    evaluated_counts: np.ndarray
    screening_recall: np.ndarray
    hit_rate: np.ndarray
    roc_auc: np.ndarray
    pr_auc: np.ndarray
    batch_redundancy: np.ndarray
    ausc: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    urllib.request.urlretrieve(url, destination)


def load_bace_csv(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "bace.csv"
    if not csv_path.exists():
        print(f"Downloading BACE dataset to {csv_path} ...")
        download_file(BACE_URL, csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded BACE CSV is empty.")
    return df


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    smiles_candidates = ["mol", "smiles", "SMILES"]
    label_candidates = ["Class", "class", "activity", "label", "y", "p_np", "HIV_active"]

    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if smiles_col is None:
        raise ValueError(f"Could not find a SMILES column in columns={list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find a label column in columns={list(df.columns)}")
    return smiles_col, label_col


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(df: pd.DataFrame, radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    smiles_col, label_col = detect_columns(df)
    smiles = df[smiles_col].astype(str).tolist()
    labels = df[label_col].astype(int).to_numpy()
    features = np.asarray([smiles_to_fp(s, radius=radius, n_bits=n_bits) for s in smiles], dtype=np.uint8)
    return features, labels


def make_model(model_name: str, seed: int):
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "mlp":
        return {"model_type": "mlp", "seed": seed}
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(model_name: str, seed: int, x_train: np.ndarray, y_train: np.ndarray):
    if model_name == "mlp":
        return train_mlp(x_train, y_train, seed=seed)
    model = make_model(model_name, seed)
    model.fit(x_train, y_train)
    return model


def predictive_uncertainty(model, x_pool: np.ndarray, model_name: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "mlp":
        return mlp_entropy_uncertainty(model, x_pool)
    probs = model.predict_proba(x_pool)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return probs, entropy


def tanimoto_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(bool)
    intersection = x @ x.T
    row_sums = x.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    union = np.maximum(union, 1)
    return intersection / union


def tanimoto_similarity_to_set(x_pool: np.ndarray, selected: np.ndarray) -> np.ndarray:
    a = x_pool.astype(bool)
    b = selected.astype(bool)
    intersection = a @ b.T
    sum_a = a.sum(axis=1, keepdims=True)
    sum_b = b.sum(axis=1, keepdims=True).T
    union = sum_a + sum_b - intersection
    union = np.maximum(union, 1)
    similarities = intersection / union
    return similarities


def batch_redundancy_score(x_batch: np.ndarray) -> float:
    if len(x_batch) <= 1:
        return 0.0
    sims = tanimoto_similarity_matrix(x_batch)
    upper = sims[np.triu_indices(len(x_batch), k=1)]
    return float(np.mean(upper))


def init_random(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    n0: int,
    rng: np.random.Generator,
    min_positives: int = 1,
) -> np.ndarray:
    positive_idx = np.flatnonzero(y_pool == 1)
    if len(positive_idx) < min_positives:
        raise ValueError("Not enough positive molecules to satisfy initialization constraint.")

    picked_pos = rng.choice(positive_idx, size=min_positives, replace=False)
    remaining = n0 - min_positives
    remaining_candidates = np.setdiff1d(np.arange(len(y_pool)), picked_pos, assume_unique=False)
    picked_rest = rng.choice(remaining_candidates, size=remaining, replace=False)
    return np.concatenate([picked_pos, picked_rest])


def init_maxmin(x_pool: np.ndarray, y_pool: np.ndarray, n0: int, rng: np.random.Generator, min_positives: int = 1) -> np.ndarray:
    positive_idx = np.flatnonzero(y_pool == 1)
    if len(positive_idx) < min_positives:
        raise ValueError("Not enough positive molecules to satisfy initialization constraint.")

    selected = [int(rng.choice(positive_idx))]
    available = np.ones(len(x_pool), dtype=bool)
    available[selected[0]] = False

    while len(selected) < n0:
        candidate_idx = np.flatnonzero(available)
        candidate_x = x_pool[candidate_idx]
        selected_x = x_pool[np.array(selected)]
        similarities = tanimoto_similarity_to_set(candidate_x, selected_x)
        min_dist = 1.0 - similarities.max(axis=1)
        next_global = int(candidate_idx[np.argmax(min_dist)])
        selected.append(next_global)
        available[next_global] = False

    selected = np.asarray(selected, dtype=int)
    if np.sum(y_pool[selected]) < min_positives:
        raise RuntimeError("MaxMin initialization failed to include required positives.")
    return selected


def query_random(pool_indices: np.ndarray, batch_size: int, rng: np.random.Generator, **_) -> np.ndarray:
    return rng.choice(pool_indices, size=batch_size, replace=False)


def query_uncertainty(
    pool_indices: np.ndarray,
    uncertainty_scores: np.ndarray,
    batch_size: int,
    **_,
) -> np.ndarray:
    order = np.argsort(uncertainty_scores)[-batch_size:]
    return pool_indices[order]


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def query_hit_seeking_uncertainty(
    pool_indices: np.ndarray,
    uncertainty_scores: np.ndarray,
    predicted_probs: np.ndarray,
    batch_size: int,
    alpha: float = 0.5,
    **_,
) -> np.ndarray:
    """
    Hybrid acquisition for virtual screening.

    score = alpha * normalized_uncertainty + (1 - alpha) * normalized_predicted_positive_probability

    alpha near 1.0 behaves like pure uncertainty sampling.
    alpha near 0.0 behaves like greedy hit-seeking based on predicted positives.
    """
    scaled_uncertainty = _minmax_scale(uncertainty_scores)
    scaled_probs = _minmax_scale(predicted_probs)
    scores = alpha * scaled_uncertainty + (1.0 - alpha) * scaled_probs
    order = np.argsort(scores)[-batch_size:]
    return pool_indices[order]


def query_diversity(
    pool_indices: np.ndarray,
    uncertainty_scores: np.ndarray,
    x_pool: np.ndarray,
    batch_size: int,
    predicted_probs: np.ndarray | None = None,
    candidate_multiplier: int = 4,
    alpha: float = 1.0,
    **_,
) -> np.ndarray:
    top_k = min(len(pool_indices), batch_size * candidate_multiplier)
    if predicted_probs is None:
        acquisition_score = uncertainty_scores
    else:
        acquisition_score = alpha * _minmax_scale(uncertainty_scores) + (1.0 - alpha) * _minmax_scale(predicted_probs)
    candidate_local = np.argsort(acquisition_score)[-top_k:]
    candidate_global = pool_indices[candidate_local]
    candidate_x = x_pool[candidate_local]

    first_local = int(np.argmax(acquisition_score[candidate_local]))
    chosen_local = [first_local]

    while len(chosen_local) < batch_size:
        selected_x = candidate_x[np.array(chosen_local)]
        similarities = tanimoto_similarity_to_set(candidate_x, selected_x)
        diversity_score = 1.0 - similarities.max(axis=1)
        diversity_score[np.array(chosen_local)] = -1.0
        next_local = int(np.argmax(diversity_score))
        chosen_local.append(next_local)

    return candidate_global[np.array(chosen_local)]


def compute_ausc(recall_curve: np.ndarray, evaluated_counts: np.ndarray) -> float:
    if len(recall_curve) == 0:
        return 0.0
    if len(recall_curve) == 1:
        return float(recall_curve[0])
    x = evaluated_counts.astype(float)
    x = x - x.min()
    max_x = x.max()
    if max_x <= 0:
        return float(np.mean(recall_curve))
    return float(np.trapz(recall_curve, x / max_x))


def screening_recall(found_positive_mask: np.ndarray, y_pool: np.ndarray) -> float:
    total_positives = int(np.sum(y_pool == 1))
    if total_positives == 0:
        return 0.0
    found_positives = int(np.sum(found_positive_mask))
    return found_positives / total_positives


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)[:, 1]
    else:
        probs = mlp_predict_proba(model, x_test)
    if len(np.unique(y_test)) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(y_test, probs)), float(average_precision_score(y_test, probs))


def run_active_learning(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    query_name: str,
    init_name: str,
    n0: int,
    budget: int,
    batch_size: int,
    seed: int,
    **query_kwargs,
) -> ActiveLearningResult:
    if budget <= n0:
        raise ValueError("budget must be larger than n0.")

    rng = np.random.default_rng(seed)
    init_fn = {"random": init_random, "maxmin": init_maxmin}[init_name]
    query_fn = {
        "random": query_random,
        "uncertainty": query_uncertainty,
        "hit_seeking": query_hit_seeking_uncertainty,
        "diversity": query_diversity,
        "diversity_hit_seeking": query_diversity,
    }[query_name]

    labeled_idx = np.sort(init_fn(x_pool, y_pool, n0, rng))
    unlabeled_mask = np.ones(len(x_pool), dtype=bool)
    unlabeled_mask[labeled_idx] = False

    evaluated_counts: List[int] = [len(labeled_idx)]
    found_positive_mask = (y_pool[labeled_idx] == 1).copy()
    screening_curve: List[float] = [screening_recall(found_positive_mask, y_pool)]
    hit_rate_curve: List[float] = [float(np.mean(y_pool[labeled_idx]))]
    redundancy_curve: List[float] = [batch_redundancy_score(x_pool[labeled_idx])]

    model = fit_model(model_name, seed, x_pool[labeled_idx], y_pool[labeled_idx])
    roc, pr = evaluate_model(model, x_test, y_test)
    roc_curve: List[float] = [roc]
    pr_curve: List[float] = [pr]

    while len(labeled_idx) < budget and np.any(unlabeled_mask):
        remaining = np.flatnonzero(unlabeled_mask)
        current_batch_size = min(batch_size, budget - len(labeled_idx), len(remaining))
        pool_probs, pool_uncertainty = predictive_uncertainty(model, x_pool[remaining], model_name=model_name)

        chosen = query_fn(
            pool_indices=remaining,
            uncertainty_scores=pool_uncertainty,
            predicted_probs=pool_probs,
            x_pool=x_pool[remaining],
            batch_size=current_batch_size,
            rng=rng,
            **query_kwargs,
        )

        labeled_idx = np.concatenate([labeled_idx, chosen])
        unlabeled_mask[chosen] = False

        batch_labels = y_pool[chosen]
        found_positive_mask = np.concatenate([found_positive_mask, batch_labels == 1])
        evaluated_counts.append(len(labeled_idx))
        screening_curve.append(screening_recall(found_positive_mask, y_pool))
        hit_rate_curve.append(float(np.mean(batch_labels)))
        redundancy_curve.append(batch_redundancy_score(x_pool[chosen]))

        model = fit_model(model_name, seed, x_pool[labeled_idx], y_pool[labeled_idx])
        roc, pr = evaluate_model(model, x_test, y_test)
        roc_curve.append(roc)
        pr_curve.append(pr)

    evaluated_counts_arr = np.asarray(evaluated_counts, dtype=int)
    screening_curve_arr = np.asarray(screening_curve, dtype=float)

    return ActiveLearningResult(
        evaluated_counts=evaluated_counts_arr,
        screening_recall=screening_curve_arr,
        hit_rate=np.asarray(hit_rate_curve, dtype=float),
        roc_auc=np.asarray(roc_curve, dtype=float),
        pr_auc=np.asarray(pr_curve, dtype=float),
        batch_redundancy=np.asarray(redundancy_curve, dtype=float),
        ausc=compute_ausc(screening_curve_arr, evaluated_counts_arr),
    )


def pad_curves(curves: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    return np.asarray([np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curves], dtype=float)


def aggregate_results(results: List[ActiveLearningResult]) -> Dict[str, np.ndarray | float]:
    recall = pad_curves([r.screening_recall for r in results])
    hit_rate = pad_curves([r.hit_rate for r in results])
    roc_auc = pad_curves([r.roc_auc for r in results])
    pr_auc = pad_curves([r.pr_auc for r in results])
    redundancy = pad_curves([r.batch_redundancy for r in results])

    ausc_values = np.asarray([r.ausc for r in results], dtype=float)
    n_seeds = len(results)

    return {
        "evaluated_counts": results[0].evaluated_counts,
        "mean_recall": recall.mean(axis=0),
        "ci95_recall": 1.96 * recall.std(axis=0, ddof=0) / math.sqrt(n_seeds),
        "mean_hit_rate": hit_rate.mean(axis=0),
        "mean_roc_auc": roc_auc.mean(axis=0),
        "mean_pr_auc": pr_auc.mean(axis=0),
        "mean_batch_redundancy": redundancy.mean(axis=0),
        "mean_ausc": float(np.mean(ausc_values)),
        "std_ausc": float(np.std(ausc_values)),
    }


def run_condition(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    query_name: str,
    init_name: str,
    n0: int,
    budget: int,
    batch_size: int,
    n_seeds: int,
    **query_kwargs,
) -> Dict[str, np.ndarray | float]:
    seed_results: List[ActiveLearningResult] = []

    for seed in range(n_seeds):
        x_pool, x_test, y_pool, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed,
        )
        result = run_active_learning(
            x_pool=x_pool,
            y_pool=y_pool,
            x_test=x_test,
            y_test=y_test,
            model_name=model_name,
            query_name=query_name,
            init_name=init_name,
            n0=n0,
            budget=budget,
            batch_size=batch_size,
            seed=seed,
            **query_kwargs,
        )
        seed_results.append(result)

    return aggregate_results(seed_results)


def plot_screening_curves(results: Dict[str, Dict[str, np.ndarray | float]], title: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot generation for {output_path.name}: matplotlib unavailable ({exc}).")
        return

    plt.figure(figsize=(8, 5))
    for label, result in results.items():
        x = result["evaluated_counts"]
        y = result["mean_recall"]
        ci = result["ci95_recall"]
        plt.plot(x, y, label=f"{label} (AUSC={result['mean_ausc']:.3f})")
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)
    plt.xlabel("Molecules Evaluated")
    plt.ylabel("Screening Recall")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_core_experiment(x: np.ndarray, y: np.ndarray, args, output_dir: Path) -> List[Dict[str, object]]:
    settings = {
        "Random sampling": {"model_name": "rf", "query_name": "random"},
        "Uncertainty sampling": {"model_name": "rf", "query_name": "uncertainty"},
        "Diversity-aware uncertainty": {"model_name": "rf", "query_name": "diversity"},
        "Logistic regression uncertainty": {"model_name": "logreg", "query_name": "uncertainty"},
    }

    results = {}
    summary_rows = []
    for label, cfg in settings.items():
        print(f"Running core condition: {label}")
        results[label] = run_condition(
            x=x,
            y=y,
            model_name=cfg["model_name"],
            query_name=cfg["query_name"],
            init_name="random",
            n0=args.n0,
            budget=args.budget,
            batch_size=args.batch_size,
            n_seeds=args.n_seeds,
        )
        res = results[label]
        summary_rows.append(
            {
                "experiment": "core",
                "condition": label,
                "mean_ausc": f"{res['mean_ausc']:.4f}",
                "std_ausc": f"{res['std_ausc']:.4f}",
                "final_recall": f"{res['mean_recall'][-1]:.4f}",
                "final_roc_auc": f"{res['mean_roc_auc'][-1]:.4f}",
                "final_pr_auc": f"{res['mean_pr_auc'][-1]:.4f}",
            }
        )

    plot_screening_curves(
        results,
        title="Core Experiment: Active Learning on BACE",
        output_path=output_dir / "core_screening_curves.png",
    )
    return summary_rows


def run_batch_ablation(x: np.ndarray, y: np.ndarray, args, output_dir: Path) -> List[Dict[str, object]]:
    results = {}
    summary_rows = []
    for batch_size in [10, 25, 50]:
        label = f"batch={batch_size}"
        print(f"Running batch ablation: {label}")
        results[label] = run_condition(
            x=x,
            y=y,
            model_name="rf",
            query_name="diversity",
            init_name="random",
            n0=args.n0,
            budget=args.budget,
            batch_size=batch_size,
            n_seeds=args.n_seeds,
        )
        res = results[label]
        summary_rows.append(
            {
                "experiment": "batch",
                "condition": label,
                "mean_ausc": f"{res['mean_ausc']:.4f}",
                "std_ausc": f"{res['std_ausc']:.4f}",
                "final_recall": f"{res['mean_recall'][-1]:.4f}",
                "final_roc_auc": f"{res['mean_roc_auc'][-1]:.4f}",
                "final_pr_auc": f"{res['mean_pr_auc'][-1]:.4f}",
            }
        )

    plot_screening_curves(
        results,
        title="Batch Size Ablation",
        output_path=output_dir / "batch_ablation.png",
    )
    return summary_rows


def run_init_ablation(x: np.ndarray, y: np.ndarray, args, output_dir: Path) -> List[Dict[str, object]]:
    results = {}
    summary_rows = []
    for init_name in ["random", "maxmin"]:
        label = f"init={init_name}"
        print(f"Running initialization ablation: {label}")
        results[label] = run_condition(
            x=x,
            y=y,
            model_name="rf",
            query_name="diversity",
            init_name=init_name,
            n0=args.n0,
            budget=args.budget,
            batch_size=args.batch_size,
            n_seeds=args.n_seeds,
        )
        res = results[label]
        summary_rows.append(
            {
                "experiment": "initialization",
                "condition": label,
                "mean_ausc": f"{res['mean_ausc']:.4f}",
                "std_ausc": f"{res['std_ausc']:.4f}",
                "final_recall": f"{res['mean_recall'][-1]:.4f}",
                "final_roc_auc": f"{res['mean_roc_auc'][-1]:.4f}",
                "final_pr_auc": f"{res['mean_pr_auc'][-1]:.4f}",
            }
        )

    plot_screening_curves(
        results,
        title="Initialization Ablation",
        output_path=output_dir / "initialization_ablation.png",
    )
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Course project active learning pipeline for BACE.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_course_project"))
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--experiment",
        choices=["core", "batch", "init", "all"],
        default="all",
        help="Which experiment set to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.data_dir)

    df = load_bace_csv(args.data_dir)
    x, y = build_feature_matrix(df)

    print(f"Loaded BACE with {len(y)} molecules.")
    print(f"Positive prevalence: {np.mean(y):.3f}")

    summary_rows: List[Dict[str, object]] = []

    if args.experiment in {"core", "all"}:
        summary_rows.extend(run_core_experiment(x, y, args, args.output_dir))
    if args.experiment in {"batch", "all"}:
        summary_rows.extend(run_batch_ablation(x, y, args, args.output_dir))
    if args.experiment in {"init", "all"}:
        summary_rows.extend(run_init_ablation(x, y, args, args.output_dir))

    save_summary_csv(summary_rows, args.output_dir / "summary.csv")
    print(f"Saved summary to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
