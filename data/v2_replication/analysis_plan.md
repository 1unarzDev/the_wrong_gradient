# v2 replication: analysis plan (written 2026-11-15T14:37:25-06:00, before any replication result was viewed)

Data: seeds 2000–2199 (fresh; never used in development), 7 strategies x {degraded, ideal} radio, sensor perception,
rb2 source hash 94a01fe3f247d56f (ablation switches added afterwards, defaults identical).

Family (6 paired contrasts, Holm within the family, separately for each outcome):
  1. P3 − A1, ideal radio      (control-aware ETA; v1 H1 analogue)
  2. P3 − A2, degraded radio   (age-aware beliefs; v1 H2 analogue)
  3. P4 − P3, ideal radio
  4. P4 − P3, degraded radio
  5. P1 − P3, ideal radio
  6. P1 − P3, degraded radio
Outcomes: total points (primary for this exploratory replication); task points excluding the return bonus (secondary).
Test: sign-flip permutation (100k), BCa/percentile bootstrap CI over courses, d_z. Everything is exploratory
(not preregistered with a registry), but the family and outcomes are fixed here before viewing results.
