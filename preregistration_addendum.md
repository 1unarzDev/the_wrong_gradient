# Preregistration addendum (deviations), written 2026-11-15T01:54:13-06:00

The original preregistration (`preregistration.md`, sha256 in `data/confirmatory/PREREG_SHA256`) is unchanged.

## Deviation 1 — world-generator failure, fixed before any outcome was viewed
- The first confirmatory launch aborted after 70 of 5600 episodes because task placement raised
  `RuntimeError: task placement failed` for 9 of the 400 held-out worlds
  (20012, 20049, 20083, 20096, 20098, 20115, 20179, 20341, 20383). The 50 pilot worlds never hit it.
- None of the 70 completed episodes were inspected; the rows were deleted (`data/confirmatory/aborted_run1/`
  keeps the log only).
- Fix (`src/tasks.py`): placement constraints (task spacing 11 m, channel padding 10 m) are relaxed in two
  stages **only after** the original 400 attempts fail. Verified by fingerprinting all 450 pilot + held-out
  worlds before/after: 441/441 previously generated worlds are bit-identical; the 9 failing worlds now generate.
  This is a property of the world generator (identical for all policies), decided without outcome data.
- New source hash: `7f9b823a3adcfddc` (was `31e257dced64a4e4`). Seeds, N, hypotheses, outcome, tests unchanged.
