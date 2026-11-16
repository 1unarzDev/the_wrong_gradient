# Two-boat replication: analysis plan (written 2026-11-15T17:00:48-06:00, before any two-boat result was viewed)
Same 200 courses as the four-boat replication (seeds 2000–2199; the course is identical for 2 and 4 boats),
7 strategies x {degraded, ideal} radio, sensor perception, n_boats = 2.
Code changes since the four-boat run (none affects n = 4): ablation switches (defaults on), course size fixed at the
4-boat standard for fleets of <= 4, two-boat start spacing 5 m (3.2 m caused a mutual-yield deadlock at the channel).
Family and outcomes identical to data/v2_replication/analysis_plan.md (6 contrasts; total points primary,
task points secondary; Holm within each outcome; sign-flip test; bootstrap CIs; d_z).
Additional descriptive comparison (not tested): 2 vs 4 boats per strategy on the same courses.
