#!/usr/bin/env python3
"""
Dataset-dependent acquisition strategies for ligand-based virtual screening.

Main framing:
- The objective is to recover active molecules under a fixed evaluation budget.
- The baseline is pure exploitation: select molecules with high P(active).
- The main experiment sweeps alpha in an exploration-exploitation acquisition:

      score(x) = (1 - alpha) * P(active | x) + alpha * U(x)

  alpha = 0 is pure exploitation.
  alpha = 1 is pure uncertainty.

- The proposed method adapts alpha online for each dataset based on recent
  screening hit rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class DatasetTask:
    key: str
    display_name: str
    filename: str
    urls: Tuple[str, ...]
    smiles_col: str
    label_col: str
    expected_hit_rate: str


DATASETS: Dict[str, DatasetTask] = {
    "hiv": DatasetTask(
        key="hiv",
        display_name="HIV",
        filename="hiv.csv",
        urls=("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",),
        smiles_col="smiles",
        label_col="HIV_active",
        expected_hit_rate="very low",
    ),
    "clintox": DatasetTask(
        key="clintox",
        display_name="ClinTox CT_TOX",
        filename="clintox.csv.gz",
        urls=(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv",
        ),
        smiles_col="smiles",
        label_col="CT_TOX",
        expected_hit_rate="low",
    ),
    "tox21": DatasetTask(
        key="tox21",
        display_name="Tox21 NR-AR",
        filename="tox21.csv.gz",
        urls=(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv",
        ),
        smiles_col="smiles",
        label_col="NR-AR",
        expected_hit_rate="low-medium",
    ),
    "bace": DatasetTask(
        key="bace",
        display_name="BACE",
        filename="bace.csv",
        urls=("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",),
        smiles_col="mol",
        label_col="Class",
        expected_hit_rate="medium",
    ),
    "bbbp": DatasetTask(
        key="bbbp",
        display_name="BBBP",
        filename="bbbp.csv",
        urls=(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bbbp.csv",
        ),
        smiles_col="smiles",
        label_col="p_np",
        expected_hit_rate="high",
    ),
}


@dataclass
class ScreeningResult:
    method: str
    dataset: str
    alpha: float | None
    evaluated_counts: np.ndarray
    recall_curve: np.ndarray
    hit_rate_curve: np.ndarray
    alpha_curve: np.ndarray
    roc_auc_curve: np.ndarray
    pr_auc_curve: np.ndarray
    ausc: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def download_dataset(task: DatasetTask, data_dir: Path) -> Path:
    ensure_dir(data_dir)
    target = data_dir / task.filename
    if target.exists():
        return target

    last_error: Exception | None = None
    for url in task.urls:
        try:
            print(f"Downloading {task.display_name} from {url}")
            urllib.request.urlretrieve(url, target)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()

    raise RuntimeError(
        f"Could not download {task.display_name}. Last error: {last_error}. "
        "If network is unavailable, download the CSV manually into data/."
    )


def load_task_dataframe(task: DatasetTask, data_dir: Path) -> pd.DataFrame:
    csv_path = download_dataset(task, data_dir)
    df = pd.read_csv(csv_path)
    missing = [c for c in [task.smiles_col, task.label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{task.display_name} is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[task.smiles_col, task.label_col]].dropna().copy()
    df[task.label_col] = pd.to_numeric(df[task.label_col], errors="coerce")
    df = df.dropna(subset=[task.label_col])
    df = df[df[task.label_col].isin([0, 1])]
    df[task.label_col] = df[task.label_col].astype(int)
    if df.empty:
        raise ValueError(f"No valid binary labels found for {task.display_name}.")
    return df


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_features(
    df: pd.DataFrame,
    task: DatasetTask,
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    smiles = df[task.smiles_col].astype(str).tolist()
    y = df[task.label_col].astype(int).to_numpy()
    x = np.asarray([smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in smiles], dtype=np.uint8)
    return x, y


def make_model(model_name: str, seed: int, rf_estimators: int = 300):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=rf_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown model_name={model_name}")


def predict_proba_and_uncertainty(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(x)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return probs, entropy


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def compute_ausc(recall_curve: np.ndarray, evaluated_counts: np.ndarray) -> float:
    if len(recall_curve) <= 1:
        return float(recall_curve[-1]) if len(recall_curve) else 0.0
    x = evaluated_counts.astype(float)
    x = x - x.min()
    if float(x.max()) <= 0:
        return float(np.mean(recall_curve))
    return float(np.trapz(recall_curve, x / x.max()))


def screening_recall(selected_labels: np.ndarray, y_pool: np.ndarray) -> float:
    total_hits = int(np.sum(y_pool == 1))
    if total_hits == 0:
        return 0.0
    return float(np.sum(selected_labels == 1) / total_hits)


def safe_auc_scores(model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_test)) < 2:
        return math.nan, math.nan
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs)), float(average_precision_score(y_test, probs))


def initialize_pool(y_pool: np.ndarray, n0: int, rng: np.random.Generator) -> np.ndarray:
    if n0 >= len(y_pool):
        raise ValueError("n0 must be smaller than the pool size.")
    if n0 < 2:
        raise ValueError("n0 must be at least 2 so both classes can be initialized.")

    positive_idx = np.flatnonzero(y_pool == 1)
    negative_idx = np.flatnonzero(y_pool == 0)
    if len(positive_idx) == 0:
        raise ValueError("The training pool contains no positive molecules.")
    if len(negative_idx) == 0:
        raise ValueError("The training pool contains no negative molecules.")

    first_positive = int(rng.choice(positive_idx))
    first_negative = int(rng.choice(negative_idx))
    remaining_candidates = np.setdiff1d(np.arange(len(y_pool)), [first_positive, first_negative])
    rest = rng.choice(remaining_candidates, size=n0 - 2, replace=False)
    return np.sort(np.concatenate([[first_positive, first_negative], rest]))


def compute_cluster_labels(x_pool: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_clusters = min(max(int(n_clusters), 1), len(x_pool))
    if n_clusters <= 1:
        return np.zeros(len(x_pool), dtype=int)

    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(1024, len(x_pool)),
        n_init=3,
        max_iter=100,
    )
    return clusterer.fit_predict(x_pool.astype(np.float32, copy=False))


def select_batch(
    method: str,
    remaining: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    alpha: float,
    cluster_labels: np.ndarray | None = None,
    cluster_penalty: float = 0.10,
) -> np.ndarray:
    if method == "random":
        return rng.choice(remaining, size=batch_size, replace=False)

    scaled_probs = minmax_scale(probs)
    scaled_uncertainty = minmax_scale(uncertainty)

    if method == "exploitation":
        scores = scaled_probs
    elif method in {"weighted", "adaptive", "bandit", "cluster_bandit"}:
        scores = (1.0 - alpha) * scaled_probs + alpha * scaled_uncertainty
    else:
        raise ValueError(f"Unknown acquisition method: {method}")

    if method == "cluster_bandit":
        if cluster_labels is None:
            raise ValueError("cluster_bandit requires cluster labels for the remaining pool.")
        chosen_local: List[int] = []
        available = np.ones(len(remaining), dtype=bool)
        cluster_counts: Dict[int, int] = {}
        for _ in range(batch_size):
            penalties = np.asarray(
                [cluster_counts.get(int(cluster_labels[i]), 0) for i in range(len(remaining))],
                dtype=float,
            )
            adjusted_scores = scores - cluster_penalty * penalties
            adjusted_scores[~available] = -np.inf
            next_local = int(np.argmax(adjusted_scores))
            chosen_local.append(next_local)
            available[next_local] = False
            cluster_id = int(cluster_labels[next_local])
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        return remaining[np.asarray(chosen_local, dtype=int)]

    chosen_local = np.argsort(scores)[-batch_size:]
    return remaining[chosen_local]


def update_alpha(
    alpha: float,
    batch_hit_rate: float,
    reference_hit_rate: float,
    eta: float,
    alpha_min: float,
    alpha_max: float,
) -> float:
    # If recent exploration is producing more hits than the current baseline,
    # allow more exploration; otherwise shift toward exploitation.
    new_alpha = alpha + eta * (batch_hit_rate - reference_hit_rate)
    return float(np.clip(new_alpha, alpha_min, alpha_max))


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

    mean_rewards = arm_rewards / np.maximum(arm_counts, 1)
    bonus = exploration * np.sqrt(np.log(max(t, 2)) / arm_counts)
    arm_idx = int(np.argmax(mean_rewards + bonus))
    return arm_idx, float(arms[arm_idx])


def run_one_seed(
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    method: str,
    alpha: float | None,
    model_name: str,
    n0: int,
    budget: int,
    batch_size: int,
    seed: int,
    adaptive_start: float,
    adaptive_eta: float,
    adaptive_min: float,
    adaptive_max: float,
    rf_estimators: int,
    bandit_arms: Sequence[float],
    bandit_exploration: float,
    n_clusters: int,
    cluster_penalty: float,
) -> ScreeningResult:
    x_pool, x_test, y_pool, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    budget = min(budget, len(y_pool))
    if budget <= n0:
        raise ValueError(f"Budget {budget} must be larger than n0 {n0} for {dataset_name}.")
    pool_cluster_labels = None
    if method == "cluster_bandit":
        pool_cluster_labels = compute_cluster_labels(x_pool, n_clusters=n_clusters, seed=seed)

    rng = np.random.default_rng(seed)
    selected_idx = initialize_pool(y_pool, n0=n0, rng=rng)
    unlabeled_mask = np.ones(len(y_pool), dtype=bool)
    unlabeled_mask[selected_idx] = False

    selected_labels = y_pool[selected_idx].copy()
    evaluated_counts = [len(selected_idx)]
    recall_curve = [screening_recall(selected_labels, y_pool)]
    hit_rate_curve = [float(np.mean(selected_labels))]

    arms = np.asarray(list(bandit_arms), dtype=float)
    if len(arms) == 0:
        raise ValueError("bandit_arms must contain at least one alpha value.")
    arm_counts = np.zeros(len(arms), dtype=float)
    arm_rewards = np.zeros(len(arms), dtype=float)
    current_arm_idx = -1

    if method == "adaptive":
        current_alpha = adaptive_start
    elif method in {"bandit", "cluster_bandit"}:
        current_arm_idx, current_alpha = choose_bandit_alpha(
            arms=arms,
            arm_counts=arm_counts,
            arm_rewards=arm_rewards,
            t=1,
            exploration=bandit_exploration,
        )
    else:
        current_alpha = float(alpha if alpha is not None else 0.0)
    alpha_curve = [current_alpha]

    model = make_model(model_name, seed, rf_estimators=rf_estimators)
    model.fit(x_pool[selected_idx], y_pool[selected_idx])
    roc, pr = safe_auc_scores(model, x_test, y_test)
    roc_auc_curve = [roc]
    pr_auc_curve = [pr]

    while len(selected_idx) < budget and np.any(unlabeled_mask):
        remaining = np.flatnonzero(unlabeled_mask)
        current_batch_size = min(batch_size, budget - len(selected_idx), len(remaining))

        probs, uncertainty = predict_proba_and_uncertainty(model, x_pool[remaining])
        chosen = select_batch(
            method=method,
            remaining=remaining,
            probs=probs,
            uncertainty=uncertainty,
            batch_size=current_batch_size,
            rng=rng,
            alpha=current_alpha,
            cluster_labels=None if pool_cluster_labels is None else pool_cluster_labels[remaining],
            cluster_penalty=cluster_penalty,
        )

        batch_labels = y_pool[chosen]
        reference_hit_rate = float(np.mean(y_pool[selected_idx]))

        selected_idx = np.concatenate([selected_idx, chosen])
        unlabeled_mask[chosen] = False
        selected_labels = np.concatenate([selected_labels, batch_labels])

        batch_hit_rate = float(np.mean(batch_labels))
        if method == "adaptive":
            current_alpha = update_alpha(
                alpha=current_alpha,
                batch_hit_rate=batch_hit_rate,
                reference_hit_rate=reference_hit_rate,
                eta=adaptive_eta,
                alpha_min=adaptive_min,
                alpha_max=adaptive_max,
            )
        elif method in {"bandit", "cluster_bandit"}:
            arm_counts[current_arm_idx] += 1.0
            arm_rewards[current_arm_idx] += batch_hit_rate
            current_arm_idx, current_alpha = choose_bandit_alpha(
                arms=arms,
                arm_counts=arm_counts,
                arm_rewards=arm_rewards,
                t=int(np.sum(arm_counts) + 1),
                exploration=bandit_exploration,
            )

        evaluated_counts.append(len(selected_idx))
        recall_curve.append(screening_recall(selected_labels, y_pool))
        hit_rate_curve.append(batch_hit_rate)
        alpha_curve.append(current_alpha)

        model = make_model(model_name, seed, rf_estimators=rf_estimators)
        model.fit(x_pool[selected_idx], y_pool[selected_idx])
        roc, pr = safe_auc_scores(model, x_test, y_test)
        roc_auc_curve.append(roc)
        pr_auc_curve.append(pr)

    evaluated = np.asarray(evaluated_counts, dtype=int)
    recall = np.asarray(recall_curve, dtype=float)
    return ScreeningResult(
        method=method,
        dataset=dataset_name,
        alpha=alpha,
        evaluated_counts=evaluated,
        recall_curve=recall,
        hit_rate_curve=np.asarray(hit_rate_curve, dtype=float),
        alpha_curve=np.asarray(alpha_curve, dtype=float),
        roc_auc_curve=np.asarray(roc_auc_curve, dtype=float),
        pr_auc_curve=np.asarray(pr_auc_curve, dtype=float),
        ausc=compute_ausc(recall, evaluated),
    )


def pad_curves(curves: Sequence[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    return np.asarray([np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in curves])


def aggregate_seed_results(results: List[ScreeningResult]) -> Dict[str, object]:
    recall = pad_curves([r.recall_curve for r in results])
    hit_rate = pad_curves([r.hit_rate_curve for r in results])
    alpha_curve = pad_curves([r.alpha_curve for r in results])
    roc = pad_curves([r.roc_auc_curve for r in results])
    pr = pad_curves([r.pr_auc_curve for r in results])
    ausc_values = np.asarray([r.ausc for r in results], dtype=float)

    return {
        "dataset": results[0].dataset,
        "method": results[0].method,
        "alpha": results[0].alpha,
        "evaluated_counts": results[0].evaluated_counts,
        "mean_recall": recall.mean(axis=0),
        "mean_hit_rate": hit_rate.mean(axis=0),
        "mean_alpha": alpha_curve.mean(axis=0),
        "mean_roc_auc": roc.mean(axis=0),
        "mean_pr_auc": pr.mean(axis=0),
        "mean_ausc": float(np.mean(ausc_values)),
        "std_ausc": float(np.std(ausc_values)),
    }


def run_condition(
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    method: str,
    alpha: float | None,
    args: argparse.Namespace,
) -> Dict[str, object]:
    seed_results = []
    for seed in range(args.n_seeds):
        seed_results.append(
            run_one_seed(
                x=x,
                y=y,
                dataset_name=dataset_name,
                method=method,
                alpha=alpha,
                model_name=args.model,
                n0=args.n0,
                budget=args.budget,
                batch_size=args.batch_size,
                seed=seed,
                adaptive_start=args.adaptive_start,
                adaptive_eta=args.adaptive_eta,
                adaptive_min=args.adaptive_min,
                adaptive_max=args.adaptive_max,
                rf_estimators=args.rf_estimators,
                bandit_arms=parse_alphas(args.bandit_alphas),
                bandit_exploration=args.bandit_exploration,
                n_clusters=args.n_clusters,
                cluster_penalty=args.cluster_penalty,
            )
        )
    return aggregate_seed_results(seed_results)


def result_to_row(result: Dict[str, object], task: DatasetTask, prevalence: float) -> Dict[str, object]:
    alpha = result["alpha"]
    if result["method"] in {"adaptive", "bandit", "cluster_bandit"}:
        alpha_label = "adaptive"
        if result["method"] == "bandit":
            alpha_label = "bandit"
        if result["method"] == "cluster_bandit":
            alpha_label = "cluster_bandit"
    elif alpha is None:
        alpha_label = "NA"
    else:
        alpha_label = f"{float(alpha):.2f}"
    return {
        "dataset": task.display_name,
        "dataset_key": task.key,
        "expected_hit_rate_group": task.expected_hit_rate,
        "observed_prevalence": f"{prevalence:.4f}",
        "method": result["method"],
        "alpha": alpha_label,
        "mean_ausc": f"{float(result['mean_ausc']):.4f}",
        "std_ausc": f"{float(result['std_ausc']):.4f}",
        "final_recall": f"{float(result['mean_recall'][-1]):.4f}",
        "final_hit_rate": f"{float(result['mean_hit_rate'][-1]):.4f}",
        "final_alpha": f"{float(result['mean_alpha'][-1]):.4f}",
        "final_roc_auc": f"{float(result['mean_roc_auc'][-1]):.4f}",
        "final_pr_auc": f"{float(result['mean_pr_auc'][-1]):.4f}",
    }


def save_curve_rows(result: Dict[str, object], task: DatasetTask) -> List[Dict[str, object]]:
    alpha = result["alpha"]
    if result["method"] in {"adaptive", "bandit", "cluster_bandit"}:
        alpha_label = "adaptive"
        if result["method"] == "bandit":
            alpha_label = "bandit"
        if result["method"] == "cluster_bandit":
            alpha_label = "cluster_bandit"
    elif alpha is None:
        alpha_label = "NA"
    else:
        alpha_label = f"{float(alpha):.2f}"
    rows = []
    counts = result["evaluated_counts"]
    recall = result["mean_recall"]
    hit_rate = result["mean_hit_rate"]
    alpha_curve = result["mean_alpha"]
    for i in range(len(counts)):
        rows.append(
            {
                "dataset": task.display_name,
                "dataset_key": task.key,
                "method": result["method"],
                "alpha": alpha_label,
                "evaluated": int(counts[i]),
                "screening_recall": f"{float(recall[i]):.4f}",
                "batch_hit_rate": f"{float(hit_rate[i]):.4f}",
                "current_alpha": f"{float(alpha_curve[i]):.4f}",
            }
        )
    return rows


def parse_alphas(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def choose_tasks(raw: str) -> List[DatasetTask]:
    if raw == "all":
        return list(DATASETS.values())
    keys = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}. Available: {list(DATASETS)}")
    return [DATASETS[k] for k in keys]


def maybe_plot_alpha_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        mpl_cache = output_dir / ".matplotlib"
        ensure_dir(mpl_cache)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    df = pd.DataFrame(rows)
    sweep = df[df["method"] == "weighted"].copy()
    if sweep.empty:
        return
    sweep["alpha_float"] = sweep["alpha"].astype(float)
    sweep["mean_ausc_float"] = sweep["mean_ausc"].astype(float)

    plt.figure(figsize=(8, 5))
    for dataset, group in sweep.groupby("dataset"):
        group = group.sort_values("alpha_float")
        plt.plot(group["alpha_float"], group["mean_ausc_float"], marker="o", label=dataset)
    plt.xlabel("alpha: exploration weight")
    plt.ylabel("Mean AUSC")
    plt.title("Dataset-dependent alpha sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_sweep_ausc.png", dpi=160)
    plt.close()

    best = sweep.sort_values("mean_ausc_float", ascending=False).groupby("dataset", as_index=False).first()
    best["observed_prevalence_float"] = best["observed_prevalence"].astype(float)
    plt.figure(figsize=(7, 5))
    plt.scatter(best["observed_prevalence_float"], best["alpha_float"], s=80)
    for _, row in best.iterrows():
        plt.annotate(row["dataset"], (row["observed_prevalence_float"], row["alpha_float"]), fontsize=8)
    plt.xlabel("Observed hit rate / prevalence")
    plt.ylabel("Best fixed alpha")
    plt.title("Best alpha versus dataset hit rate")
    plt.tight_layout()
    plt.savefig(output_dir / "best_alpha_vs_hit_rate.png", dpi=160)
    plt.close()


def save_best_alpha_table(rows: List[Dict[str, object]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    sweep = df[df["method"] == "weighted"].copy()
    if sweep.empty:
        return
    sweep["mean_ausc_float"] = sweep["mean_ausc"].astype(float)
    best = sweep.sort_values("mean_ausc_float", ascending=False).groupby("dataset", as_index=False).first()
    best_rows = best[
        [
            "dataset",
            "dataset_key",
            "expected_hit_rate_group",
            "observed_prevalence",
            "alpha",
            "mean_ausc",
            "final_recall",
            "final_hit_rate",
        ]
    ].to_dict("records")
    write_csv(best_rows, output_dir / "best_fixed_alpha_by_dataset.csv")


def run_experiments(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    tasks = choose_tasks(args.datasets)
    alphas = parse_alphas(args.alphas)
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    all_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []

    for task in tasks:
        print(f"\n=== Dataset: {task.display_name} ({task.expected_hit_rate}) ===", flush=True)
        df = load_task_dataframe(task, data_dir)
        if args.max_molecules and len(df) > args.max_molecules:
            df, _ = train_test_split(
                df,
                train_size=args.max_molecules,
                stratify=df[task.label_col],
                random_state=0,
            )
            df = df.reset_index(drop=True)
        x, y = build_features(df, task)
        prevalence = float(np.mean(y))
        dataset_rows.append(
            {
                "dataset": task.display_name,
                "dataset_key": task.key,
                "n_molecules": len(y),
                "n_actives": int(np.sum(y == 1)),
                "observed_prevalence": f"{prevalence:.4f}",
                "expected_hit_rate_group": task.expected_hit_rate,
                "label_column": task.label_col,
            }
        )

        if args.include_random or "random" in methods:
            print("Running random context baseline", flush=True)
            result = run_condition(x, y, task.display_name, "random", None, args)
            all_rows.append(result_to_row(result, task, prevalence))
            curve_rows.extend(save_curve_rows(result, task))

        if "exploitation" in methods:
            print("Running pure exploitation baseline (alpha=0)", flush=True)
            result = run_condition(x, y, task.display_name, "exploitation", 0.0, args)
            all_rows.append(result_to_row(result, task, prevalence))
            curve_rows.extend(save_curve_rows(result, task))

        if "weighted" in methods:
            for alpha in alphas:
                print(f"Running weighted acquisition alpha={alpha:.2f}", flush=True)
                result = run_condition(x, y, task.display_name, "weighted", alpha, args)
                all_rows.append(result_to_row(result, task, prevalence))
                curve_rows.extend(save_curve_rows(result, task))

        if "adaptive" in methods:
            print("Running adaptive alpha acquisition", flush=True)
            result = run_condition(x, y, task.display_name, "adaptive", None, args)
            all_rows.append(result_to_row(result, task, prevalence))
            curve_rows.extend(save_curve_rows(result, task))

        if "bandit" in methods:
            print("Running bandit adaptive alpha acquisition", flush=True)
            result = run_condition(x, y, task.display_name, "bandit", None, args)
            all_rows.append(result_to_row(result, task, prevalence))
            curve_rows.extend(save_curve_rows(result, task))

        if "cluster_bandit" in methods:
            print("Running cluster-weighted bandit acquisition", flush=True)
            result = run_condition(x, y, task.display_name, "cluster_bandit", None, args)
            all_rows.append(result_to_row(result, task, prevalence))
            curve_rows.extend(save_curve_rows(result, task))

        write_csv(all_rows, output_dir / "all_results.csv")
        write_csv(curve_rows, output_dir / "screening_curves_long.csv")
        write_csv(dataset_rows, output_dir / "dataset_summary.csv")

    alpha_sweep_rows = [r for r in all_rows if r["method"] == "weighted"]
    adaptive_rows = [r for r in all_rows if r["method"] == "adaptive"]
    bandit_rows = [r for r in all_rows if r["method"] == "bandit"]
    cluster_bandit_rows = [r for r in all_rows if r["method"] == "cluster_bandit"]
    write_csv(alpha_sweep_rows, output_dir / "fixed_alpha_sweep_results.csv")
    write_csv(adaptive_rows, output_dir / "adaptive_alpha_results.csv")
    write_csv(bandit_rows, output_dir / "bandit_alpha_results.csv")
    write_csv(cluster_bandit_rows, output_dir / "cluster_bandit_results.csv")
    save_best_alpha_table(all_rows, output_dir)
    maybe_plot_alpha_summary(all_rows, output_dir)

    print(f"\nFinished. Results written to {output_dir}", flush=True)


def list_datasets() -> None:
    for task in DATASETS.values():
        print(
            f"{task.key:8s} | {task.display_name:16s} | label={task.label_col:10s} "
            f"| expected hit rate={task.expected_hit_rate}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results_screening_alpha")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'.")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--model", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--methods",
        default="exploitation,weighted,adaptive,bandit",
        help="Comma-separated methods: random,exploitation,weighted,adaptive,bandit,cluster_bandit.",
    )
    parser.add_argument("--adaptive-start", type=float, default=0.5)
    parser.add_argument("--adaptive-eta", type=float, default=0.5)
    parser.add_argument("--adaptive-min", type=float, default=0.0)
    parser.add_argument("--adaptive-max", type=float, default=1.0)
    parser.add_argument("--bandit-alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--bandit-exploration", type=float, default=0.35)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--cluster-penalty", type=float, default=0.10)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=5000,
        help="Stratified cap per dataset for manageable course-project runtime. Use 0 for no cap.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small budget and one seed for smoke testing.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    if args.quick:
        args.n_seeds = 1
        args.n0 = min(args.n0, 30)
        args.budget = min(args.budget, 120)
        args.batch_size = min(args.batch_size, 20)

    run_experiments(args)


if __name__ == "__main__":
    main()
