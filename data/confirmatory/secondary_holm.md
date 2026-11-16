# Post-hoc Holm across the 12 secondary contrasts (robustness check; not preregistered)

| contrast | comms | mean [95% CI] | d_z | p (perm) | Holm-adjusted |
|---|---|---|---|---|---|
| H1 (degraded comms) | degraded | -0.53 [-2.25, +1.19] | -0.03 | 0.54 | 1 |
| H2 (ideal comms) | ideal | -0.05 [-0.20, +0.08] | -0.04 | 0.48 | 1 |
| control ETA, naive beliefs: A2 - P2 | ideal | -0.77 [-2.59, +1.14] | -0.04 | 0.42 | 1 |
| control ETA, naive beliefs: A2 - P2 | degraded | -0.63 [-2.31, +1.06] | -0.04 | 0.46 | 1 |
| age-aware, graph ETA: A1 - P2 | degraded | -0.12 [-1.31, +1.02] | -0.01 | 0.84 | 1 |
| marginal vs selfish: P2 - P1 | ideal | -1.07 [-2.34, +0.14] | -0.08 | 0.096 | 0.77 |
| marginal vs selfish: P2 - P1 | degraded | -0.01 [-1.41, +1.37] | -0.00 | 0.98 | 1 |
| marginal vs greedy: P2 - P0 | ideal | +33.35 [+30.90, +35.84] | +1.30 | 5e-05 | 0.0006 |
| marginal vs greedy: P2 - P0 | degraded | +32.62 [+30.11, +35.17] | +1.27 | 5e-05 | 0.0006 |
| centralized reference: P4 - P3 | ideal | +7.47 [+5.62, +9.30] | +0.40 | 5e-05 | 0.0006 |
| centralized reference: P4 - P3 | degraded | +8.26 [+6.47, +10.11] | +0.44 | 5e-05 | 0.0006 |
| cost of degraded comms for P3 | degraded-ideal | -0.79 [-2.41, +0.78] | -0.05 | 0.33 | 1 |

Permutation p-values with 20,000 sign flips have resolution 1/20,001 ≈ 5e-5.
