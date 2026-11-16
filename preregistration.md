# Preregistration — *The Wrong Gradient* (confirmatory study)

**Frozen:** 2026-11-15T01:51:01-06:00, before any held-out (confirmatory) seed was simulated.
**Source hash** (sha256 over `src/**/*.py`, first 16 hex): `31e257dced64a4e4`
**Frozen config file:** `configs/confirmatory.yaml` (its sha256 is recorded in
`data/confirmatory/manifest.json` when the run starts).
The sha256 of this file is written to `data/confirmatory/PREREG_SHA256` immediately after freezing.

## 1. Status of earlier data
All data in `data/pilot/` (seeds 1000–1049) are **pilot only**: debugging, calibration, parameter
choice and power analysis. Pilot rounds and every decision are listed in `PROGRESS.md`.
Pilot-informed choices:
- the high-turn bottleneck distribution (fixed a priori from the handoff's description, then checked
  on the pilot for feasibility): `r_max = 0.45 rad/s, tau_r = 0.8 s, current_strength = 0.6 m/s`,
  barrier with channels on;
- the graph-ETA speed is calibrated to be median-unbiased on pilot data: `graph_speed = 1.475 m/s`
  (a deliberately strong, respectful baseline);
- the age-aware belief time constant `age_tau_s = 12 s` (best of {3, 6, 12, 25} for P3 on the pilot;
  A1 uses the same value).

## 2. Hypotheses (confirmatory family = {H1, H2})
- **H1 (control-aware cost estimation).** In the 4-boat high-turn bottleneck distribution with
  **ideal communication**, P3 (marginal contribution, control-aware ETA, age-aware beliefs) has a
  different mean episode welfare than A1 (identical except graph-distance ETA).
  Two-sided test; the anticipated direction is P3 > A1.
- **H2 (information model).** In the same distribution with **degraded communication** (Gilbert–Elliott
  bursty loss + 0.5–3 s latency), P3 has a different mean welfare than A2 (identical except
  age-aware belief compensation disabled). Two-sided; anticipated direction P3 > A2.
- **H3 (exploratory, not confirmatory).** The H1/H2 contrasts depend on turn agility, the bottleneck,
  communication quality, and fleet size (`configs/factorial.yaml`, `configs/scale.yaml`).

Rationale for the comm condition per hypothesis: H1 is a physics/cost-model question and is isolated
under ideal communication (no belief confound); H2 is an information question and only exists under
loss/latency. Both contrasts are also reported in the other comm condition as **secondary** results.

## 3. Primary outcome (frozen)
Net episode welfare per world:
`W = (task points collected before deadlines) − 0.01 · Σ_boats ∫ (T_L² + T_R²) dt − 5 · (#boat–boat contacts + #boat–obstacle contacts)`.
Episode length 240 s. W can be negative; no ratio-based "PoA" is computed for the dynamic game.

Secondary outcomes (descriptive, no multiplicity claims): points, completion rate, expired tasks,
redundant visits, target switches, makespan, distance, energy, contacts, near misses, mean message age
used, decision CPU time, ETA error by turn complexity.

## 4. Design
- Unit of replication: an independent world seed (map, currents, disturbance, sensor and packet tapes).
  Seeds **20000–20399 (N = 400)**, never simulated before this freeze.
- Every world is evaluated by all seven policies (P0, P1, P2, A1, A2, P3, P4) under both ideal and
  degraded communication, with common random numbers (named independent RNG streams).
- N chosen by `analysis/power.py` from pilot data only (`data/power.json`): the smallest N in
  {100, 200, 240, 300, 400, 500} with Holm-adjusted power ≥ 0.90 for both H1 and H2 at a planning
  effect d_z = 0.25 **and** 95 % CI half-width ≤ 2.0 points for H1 → **N = 400**.
  (If H2's true effect were as small as its pilot estimate, d_z ≈ −0.06, power would be ≈ 0.2; a null
  H2 result is therefore a plausible outcome and will be reported as such.)

## 5. Analysis plan
- Per-world paired differences d_w = W_P3 − W_comparator.
- Test: two-sided paired sign-flip randomization test, 100 000 random sign vectors, p = (1+#)/(1+B).
- Multiplicity: Holm step-down across {H1, H2} at familywise α = 0.05. Report unadjusted and adjusted p.
- Effect sizes: mean and median difference, paired d_z = mean/sd; 95 % BCa bootstrap CIs (20 000
  resamples of worlds) for mean, median and d_z; fraction of worlds with d > 0, < 0, = 0.
- Sensitivity (reported, not used for decisions): paired t-test and Wilcoxon signed-rank.
- Failure handling: any episode that raises an error is reported; no world is excluded for its outcome.
- No change to scenarios, metrics, penalty weights, seed sets or the test family after viewing results.
- All other contrasts (P1 vs P2 "selfish vs marginal", P4 vs P3 "centralized reference", etc.),
  H3 experiments and the scaling study are **exploratory** and labelled as such.
