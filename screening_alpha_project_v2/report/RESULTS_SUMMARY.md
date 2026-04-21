# Results Summary

These results are from the completed five-dataset run using the clean v2 project folder.

## Run Configuration

- Model: Random Forest surrogate
- Trees: 100
- Seeds: 3
- Initial evaluated set: 50 molecules
- Total evaluation budget: 500 molecules
- Batch size: 25
- Large-dataset cap: 5000 molecules with stratified subsampling
- Output directory: `results/`

## Dataset Hit Rates

| Dataset | Hit-rate group | Molecules | Actives | Observed hit rate |
|---|---:|---:|---:|---:|
| HIV | very low | 5000 | 175 | 0.0350 |
| ClinTox CT_TOX | low | 1484 | 112 | 0.0755 |
| Tox21 NR-AR | low-medium | 5000 | 213 | 0.0426 |
| BACE | medium | 1513 | 691 | 0.4567 |
| BBBP | high | 2050 | 1567 | 0.7644 |

## Best Fixed Alpha by Dataset

| Dataset | Best alpha | Mean AUSC | Final recall | Final batch hit rate |
|---|---:|---:|---:|---:|
| HIV | 0.00 | 0.2814 | 0.4595 | 0.0667 |
| ClinTox CT_TOX | 0.50 | 0.5414 | 0.8222 | 0.0667 |
| Tox21 NR-AR | 0.25 | 0.4668 | 0.5608 | 0.0133 |
| BACE | 0.25 | 0.3610 | 0.6510 | 0.7467 |
| BBBP | 0.00 | 0.2078 | 0.3785 | 0.9600 |

## Adaptive Alpha Results

| Dataset | Adaptive AUSC | Final recall | Final alpha |
|---|---:|---:|---:|
| HIV | 0.2737 | 0.4571 | 0.5182 |
| ClinTox CT_TOX | 0.5336 | 0.8037 | 0.2815 |
| Tox21 NR-AR | 0.4273 | 0.5569 | 0.0000 |
| BACE | 0.2936 | 0.5353 | 0.7666 |
| BBBP | 0.1767 | 0.3224 | 0.4879 |

## Bandit Adaptive Alpha Results

The UCB bandit method treats each candidate $\alpha$ as an arm and uses the batch hit rate as the reward. This avoids choosing a fixed $\alpha$ manually while still adapting the exploration--exploitation balance online.

| Dataset | Bandit AUSC | Bandit final recall | Final mean alpha |
|---|---:|---:|---:|
| HIV | 0.2720 | 0.4214 | 0.6667 |
| ClinTox CT_TOX | 0.5316 | 0.8185 | 0.3333 |
| Tox21 NR-AR | 0.4660 | 0.5667 | 0.3667 |
| BACE | 0.3590 | 0.6624 | 0.2833 |
| BBBP | 0.1901 | 0.3503 | 0.2000 |

Compared with the simple adaptive rule, bandit alpha improves AUSC on BACE, BBBP, and Tox21, and nearly matches the best fixed-alpha AUSC on BACE and Tox21. On BACE and Tox21, bandit alpha also achieves higher final recall than the best fixed-alpha setting. It still does not consistently beat the best fixed alpha in AUSC, but the best fixed alpha is selected with hindsight, while the bandit strategy chooses alpha online.

## Cluster-Weighted Bandit Extension

We also tested a cluster-weighted bandit acquisition rule. The method clusters Morgan fingerprints with MiniBatchKMeans and applies a within-batch cluster penalty so that a batch is less likely to select many molecules from the same cluster. This is motivated by virtual screening practice: useful hits should ideally be both active and structurally diverse.

With a mild cluster penalty of 0.03, the results were:

| Dataset | Cluster-bandit AUSC | Cluster-bandit final recall | Final mean alpha |
|---|---:|---:|---:|
| HIV | 0.2694 | 0.4476 | 0.5000 |
| ClinTox CT_TOX | 0.4957 | 0.7852 | 0.3667 |
| Tox21 NR-AR | 0.4513 | 0.5529 | 0.2833 |
| BACE | 0.3537 | 0.6480 | 0.3333 |
| BBBP | 0.1898 | 0.3543 | 0.0667 |

The cluster penalty did not improve mean AUSC over the non-cluster bandit in most datasets. It slightly improved final recall on BBBP and improved recall relative to non-cluster bandit on HIV when a stronger penalty was used, but it also reduced hit recovery on ClinTox and Tox21. This suggests that cluster diversity is useful as a secondary objective, but the penalty must be tuned carefully so that diversity does not override hit-seeking too strongly.

## Main Interpretation

The results support the new framing that virtual screening should be treated as active-molecule recovery rather than classifier training. Pure exploitation is a very strong baseline because directly ranking by predicted active probability is well aligned with the screening objective.

The fixed-alpha sweep shows that the best exploration weight is dataset-dependent. HIV and BBBP prefer pure exploitation (`alpha = 0.00`). Tox21 and BACE prefer mild exploration (`alpha = 0.25`). ClinTox prefers a moderate exploration weight (`alpha = 0.50`), although the difference from pure exploitation is small.

The adaptive alpha methods work as proof-of-concept dataset-aware strategies. The simple hit-rate update rule does not consistently beat the best fixed alpha and can move too far toward exploration on BACE and BBBP. The bandit version is stronger: it substantially improves over the simple adaptive rule on BACE and Tox21 and gives competitive recall without manually tuning alpha for each dataset. The cluster-weighted version adds a useful diversity mechanism, but in the current implementation it trades off some AUSC for diversity pressure.

## Report-Ready Conclusion

Across five ligand-based screening tasks with different observed hit rates, acquisition performance depends strongly on the exploration-exploitation balance. Pure uncertainty sampling is not a reliable screening objective. Instead, the strongest policies are usually exploitative or mildly exploratory, emphasizing predicted activity while allowing limited uncertainty-driven exploration. The bandit adaptive strategy provides a stronger dataset-aware framework than the simple adaptive rule: it learns alpha online and is competitive with the best fixed-alpha strategy on several datasets, although further improvement is needed to consistently beat hindsight-selected fixed alpha.
