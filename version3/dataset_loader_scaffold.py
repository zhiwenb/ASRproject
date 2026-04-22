#!/usr/bin/env python3
"""
dataset_loader_scaffold.py

Load local molecular datasets from ./data and create scaffold splits.

Supported datasets in ./data:
- hiv.csv
- bace.csv
- bbbp.csv
- clintox.csv.gz
- tox21.csv.gz
- muv.csv.gz

Notes:
- Uses Bemis-Murcko scaffold split to make the screening problem harder.
- For multi-task datasets (tox21, muv), choose one label column explicitly.
- Keeps only binary labels {0,1}.
- Drops rows with missing SMILES or missing labels.
- Converts SMILES to Morgan fingerprints for later experiments.

Example:
    python dataset_loader_scaffold.py --dataset muv --label-col MUV-466
    python dataset_loader_scaffold.py --dataset tox21 --label-col NR-AR
    python dataset_loader_scaffold.py --dataset hiv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# -----------------------------
# Dataset definitions
# -----------------------------

@dataclass(frozen=True)
class DatasetTask:
    key: str
    filename: str
    smiles_col: str
    default_label_col: str
    is_multitask: bool = False


DATASETS: Dict[str, DatasetTask] = {
    "hiv": DatasetTask(
        key="hiv",
        filename="hiv.csv",
        smiles_col="smiles",
        default_label_col="HIV_active",
        is_multitask=False,
    ),
    "bace": DatasetTask(
        key="bace",
        filename="bace.csv",
        smiles_col="mol",
        default_label_col="Class",
        is_multitask=False,
    ),
    "bbbp": DatasetTask(
        key="bbbp",
        filename="bbbp.csv",
        smiles_col="smiles",
        default_label_col="p_np",
        is_multitask=False,
    ),
    "clintox": DatasetTask(
        key="clintox",
        filename="clintox.csv.gz",
        smiles_col="smiles",
        default_label_col="CT_TOX",
        is_multitask=True,
    ),
    "tox21": DatasetTask(
        key="tox21",
        filename="tox21.csv.gz",
        smiles_col="smiles",
        default_label_col="NR-AR",
        is_multitask=True,
    ),
    "muv": DatasetTask(
        key="muv",
        filename="muv.csv.gz",
        smiles_col="smiles",
        default_label_col="MUV-466",
        is_multitask=True,
    ),
}


# -----------------------------
# Basic helpers
# -----------------------------

def read_local_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # pandas can read .csv and .csv.gz directly
    return pd.read_csv(path)


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


# -----------------------------
# Dataset loading
# -----------------------------

def get_task(dataset_key: str) -> DatasetTask:
    dataset_key = dataset_key.lower()
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_key]


def load_dataset_dataframe(
    data_dir: Path,
    dataset_key: str,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, DatasetTask, str]:
    task = get_task(dataset_key)
    df = read_local_csv(data_dir / task.filename)

    chosen_label_col = label_col if label_col is not None else task.default_label_col

    missing = [c for c in [task.smiles_col, chosen_label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_key}' is missing columns {missing}. "
            f"Available columns include: {list(df.columns)[:30]}"
        )

    out = df[[task.smiles_col, chosen_label_col]].copy()
    out = out.rename(columns={task.smiles_col: "smiles", chosen_label_col: "label"})
    out = out.dropna(subset=["smiles", "label"]).copy()

    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"]).copy()
    out = out[out["label"].isin([0, 1])].copy()
    out["label"] = out["label"].astype(int)

    # remove invalid molecules
    mols = out["smiles"].astype(str).apply(Chem.MolFromSmiles)
    valid_mask = mols.notnull()
    out = out.loc[valid_mask].reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid rows remain for dataset '{dataset_key}' and label '{chosen_label_col}'.")

    return out, task, chosen_label_col


def build_features(
    df: pd.DataFrame,
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    smiles_list = df["smiles"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()
    x = np.asarray(
        [smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in smiles_list],
        dtype=np.uint8,
    )
    return x, y


# -----------------------------
# Scaffold split
# -----------------------------

def generate_scaffold_groups(smiles_list: Sequence[str]) -> List[List[int]]:
    scaffold_to_indices: Dict[str, List[int]] = {}

    for idx, smi in enumerate(smiles_list):
        scaffold = smiles_to_scaffold(smi)
        # Treat invalid/empty scaffold as its own singleton bucket
        if scaffold == "":
            scaffold = f"NO_SCAFFOLD_{idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    # Sort scaffold groups by size descending, then earliest index
    groups = sorted(
        scaffold_to_indices.values(),
        key=lambda g: (-len(g), g[0]),
    )
    return groups


def scaffold_split_indices(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("frac_train + frac_valid + frac_test must sum to 1.")

    n_total = len(smiles_list)
    train_cutoff = frac_train * n_total
    valid_cutoff = (frac_train + frac_valid) * n_total

    groups = generate_scaffold_groups(smiles_list)

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for group in groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(valid_idx) + len(group) <= valid_cutoff:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(valid_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def make_active_learning_pool_from_scaffold_split(
    df: pd.DataFrame,
    frac_pool: float = 0.8,
    frac_test: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isclose(frac_pool + frac_test, 1.0):
        raise ValueError("frac_pool + frac_test must sum to 1.")

    train_idx, _, test_idx = scaffold_split_indices(
        df["smiles"].tolist(),
        frac_train=frac_pool,
        frac_valid=0.0,
        frac_test=frac_test,
    )

    pool_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return pool_df, test_df


# -----------------------------
# Optional utility for AL init
# -----------------------------

def initialize_pool_indices(y_pool: np.ndarray, n0: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if n0 >= len(y_pool):
        raise ValueError("n0 must be smaller than pool size.")
    if n0 < 2:
        raise ValueError("n0 must be at least 2.")

    pos_idx = np.flatnonzero(y_pool == 1)
    neg_idx = np.flatnonzero(y_pool == 0)

    if len(pos_idx) == 0:
        raise ValueError("Pool contains no positives.")
    if len(neg_idx) == 0:
        raise ValueError("Pool contains no negatives.")

    first_pos = int(rng.choice(pos_idx))
    first_neg = int(rng.choice(neg_idx))

    remaining = np.setdiff1d(np.arange(len(y_pool)), np.array([first_pos, first_neg]))
    rest = rng.choice(remaining, size=n0 - 2, replace=False)

    init_idx = np.concatenate([[first_pos, first_neg], rest])
    return np.sort(init_idx)


# -----------------------------
# Summary / CLI
# -----------------------------

def summarize_split(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    n_pos = int(df["label"].sum())
    prevalence = (n_pos / n) if n > 0 else 0.0
    print(f"{name:>8s} | n={n:6d} | positives={n_pos:6d} | prevalence={prevalence:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load local datasets and make scaffold splits.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--label-col", type=str, default=None,
                        help="Override default label column, useful for tox21/muv multitask data.")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--save-prefix", type=str, default=None,
                        help="If set, save split CSVs with this prefix.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df, task, chosen_label = load_dataset_dataframe(
        data_dir=data_dir,
        dataset_key=args.dataset,
        label_col=args.label_col,
    )

    print(f"\nLoaded dataset: {task.key}")
    print(f"File: {task.filename}")
    print(f"Label column: {chosen_label}")
    summarize_split("full", df)

    pool_df, test_df = make_active_learning_pool_from_scaffold_split(df, frac_pool=0.8, frac_test=0.2)

    print("\nScaffold split (for harder experiments):")
    summarize_split("pool", pool_df)
    summarize_split("test", test_df)

    # Build features once so you know this works
    x_pool, y_pool = build_features(pool_df, radius=args.radius, n_bits=args.n_bits)
    x_test, y_test = build_features(test_df, radius=args.radius, n_bits=args.n_bits)

    print("\nFeature shapes:")
    print(f"X_pool: {x_pool.shape}, y_pool: {y_pool.shape}")
    print(f"X_test: {x_test.shape}, y_test: {y_test.shape}")

    # Try a default AL init
    if len(np.unique(y_pool)) == 2 and len(y_pool) > 50:
        init_idx = initialize_pool_indices(y_pool, n0=50, seed=0)
        init_pos = int(y_pool[init_idx].sum())
        print(f"\nInitial labeled pool example: n0=50, positives={init_pos}")
    else:
        print("\nSkipping initial-pool demo because scaffold pool is too small or lacks both classes.")

    if args.save_prefix:
        out_dir = Path(".")
        pool_path = out_dir / f"{args.save_prefix}_pool_scaffold.csv"
        test_path = out_dir / f"{args.save_prefix}_test_scaffold.csv"
        pool_df.to_csv(pool_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"\nSaved:")
        print(f"  {pool_path}")
        print(f"  {test_path}")


if __name__ == "__main__":
    main()