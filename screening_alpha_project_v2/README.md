# Dataset-Dependent Screening Alpha Project

This folder is the clean v2 version of the project. It reframes ligand-based virtual screening as an active-molecule recovery problem rather than a classifier-training problem.

## Folder Structure

```text
screening_alpha_project_v2/
  code/
    screening_alpha_experiments.py
  data/
    bace.csv
    bbbp.csv
    hiv.csv                  # downloaded when the full experiment is run
    clintox.csv.gz           # downloaded when the full experiment is run
    tox21.csv.gz             # downloaded when the full experiment is run
  results/
    all_results.csv
    fixed_alpha_sweep_results.csv
    adaptive_alpha_results.csv
    best_fixed_alpha_by_dataset.csv
    dataset_summary.csv
    screening_curves_long.csv
    alpha_sweep_ausc.png
    best_alpha_vs_hit_rate.png
  report/
    report_outline.tex
```

## Research Question

The project asks whether different molecular screening datasets require different exploration-exploitation tradeoffs.

The acquisition function is:

```text
score(x) = (1 - alpha) * P(active | x) + alpha * U(x)
```

where `P(active | x)` is the surrogate model's predicted active probability and `U(x)` is predictive entropy.

## Methods

The experiment compares:

- `random`: context baseline.
- `exploitation`: pure hit-seeking baseline, equivalent to alpha = 0.
- `weighted`: fixed-alpha acquisition with alpha in `{0, 0.1, 0.25, 0.5, 0.75, 1.0}`.
- `adaptive`: online alpha update based on recent batch hit rate.
- `bandit`: online UCB bandit over candidate alpha values.
- `cluster_bandit`: UCB bandit alpha selection with cluster-weighted batch diversity.

The adaptive rule is:

```text
alpha_{t+1} = clip(alpha_t + eta * (batch_hit_rate - current_evaluated_hit_rate), alpha_min, alpha_max)
```

If the most recent batch underperforms the current evaluated-set hit rate, alpha decreases and the policy becomes more exploitative. If the batch finds more hits than expected, alpha increases and allows more exploration.

The bandit method treats each candidate alpha as an arm and uses batch hit rate as the reward:

```text
UCB(alpha_i) = mean_reward(alpha_i) + c * sqrt(log(t) / n_i)
```

This lets the method learn dataset-specific alpha preferences online instead of manually fixing alpha in advance.

The cluster-weighted bandit extension first clusters Morgan fingerprints with MiniBatchKMeans, then applies a within-batch penalty when multiple selected molecules come from the same cluster:

```text
adjusted_score(x) = acquisition_score(x) - lambda * count_selected_from_cluster(C(x))
```

This encourages structurally broader batches, but can trade off against pure hit recovery if the penalty is too strong.

## Datasets

The full experiment uses five datasets/tasks with different hit-rate regimes:

| Key | Dataset/task | Expected hit-rate group | Label column |
|---|---|---|---|
| `hiv` | HIV | very low | `HIV_active` |
| `clintox` | ClinTox CT_TOX | low | `CT_TOX` |
| `tox21` | Tox21 NR-AR | low-medium | `NR-AR` |
| `bace` | BACE | medium | `Class` |
| `bbbp` | BBBP | high | `p_np` |

The actual observed hit rate is written to `results/dataset_summary.csv`.

## Run Commands

From this folder:

```bash
PYTHONPATH=/Users/bianzhiwen/Desktop/CMU/project/.vendor /Users/bianzhiwen/opt/anaconda3/bin/python code/screening_alpha_experiments.py --datasets all --data-dir data --output-dir results --n-seeds 3 --budget 500 --batch-size 25 --include-random --rf-estimators 100
```

Quick test:

```bash
PYTHONPATH=/Users/bianzhiwen/Desktop/CMU/project/.vendor /Users/bianzhiwen/opt/anaconda3/bin/python code/screening_alpha_experiments.py --datasets bace,bbbp --data-dir data --output-dir results_smoke --quick --include-random
```

Note: this environment currently needs the `PYTHONPATH=/Users/bianzhiwen/Desktop/CMU/project/.vendor` prefix because the base Anaconda pandas/numpy installation has a binary-version mismatch.

## Main Outputs

- `results/all_results.csv`: all methods and datasets.
- `results/fixed_alpha_sweep_results.csv`: fixed alpha sweep only.
- `results/adaptive_alpha_results.csv`: adaptive alpha only.
- `results/best_fixed_alpha_by_dataset.csv`: best fixed alpha for each dataset.
- `results/dataset_summary.csv`: dataset sizes and observed hit rates.
- `results/screening_curves_long.csv`: long-format curves for plotting and report tables.
- `results/alpha_sweep_ausc.png`: AUSC versus alpha across datasets.
- `results/best_alpha_vs_hit_rate.png`: best fixed alpha versus observed hit rate.
- `report/RESULTS_SUMMARY.md`: concise result interpretation for the final report.
- `results_bandit/bandit_alpha_results.csv`: UCB bandit adaptive alpha results.
- `results_bandit/bandit_comparison.csv`: comparison against best fixed alpha and simple adaptive alpha.
- `results_cluster_bandit_p003/cluster_bandit_results.csv`: cluster-weighted bandit results with a mild cluster penalty.
- `results_cluster_bandit_p003/cluster_bandit_penalty_comparison.csv`: comparison across best fixed alpha, bandit, and cluster-bandit.

The old interrupted 300-tree run is archived in `archive/partial_300tree_results/`. The completed five-dataset run is in `results/`.
