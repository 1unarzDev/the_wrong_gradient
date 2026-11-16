# Changelog: revision of *The Wrong Gradient*

## Scientific changes
- **Competition facts corrected.**
  - Tasks and scoring were checked against the RoboBoat 2025 handbook (§3.2, §4.2); the simulator
    implements that task set. The 2026 handbook allows up to two vehicles per team (§5.1) with a new task
    scenario.
  - The paper now cites both editions.
  - It frames two boats as the competition-relevant fleet, four boats as the scaling test, and one boat
    as a reference point.
- **Physical mismatch made the centrepiece.** A coupling study of the Braess return rush:
  - arrival-spacing injection, matching the deterministic queue law;
  - a service-interval law, h ≈ 5.4 s + L/1.65 m/s;
  - a sweep over the downstream/upstream service-interval ratio ρ, with end points re-validated on fresh
    seeds;
  - a **four-cycle circulation test** on the measured route game. The separable game is exactly 0; the
    measured queue game reaches 21.7 s against noise of about 3.6 s.
  - A **strict improvement cycle**, which rules out even an improvement-preserving (ordinal) potential for the measured finite game; the nonzero circulation rules out an exact one.
- **New comparison strategies.**
  - **P5**: an information-limited central planner (a leader boat plans on its own map and on radio
    positions, and assignments are delivered by radio). It separates optimization from privileged
    information.
  - **R3**: reservations that expire at the claimant's announced completion time.
  - **L3**: a heartbeat lease, added after a diagnostic showed that announced arrivals are times to
    *reach* a task, not to finish it. Exploratory.
- **Robustness.** The key contrasts were re-run with a harsher radio and with slower boats.
- **Statistics.**
  - All reported numbers were reproduced from the raw episode files.
  - Every test family is defined.
  - Planned studies and exploratory ones are distinguished (Table 3).
  - Intervals are unadjusted and p-values are Holm-adjusted.
  - The sign-flip symmetry assumption was checked against paired t-tests.

## Editorial changes
- The paper is reorganized around one argument: individual optimization → exact potential →
  externality → marginal-cost correction (discrete FTC) → potential as conservative field → physical
  model mismatch.
- **New Fig. 1:** staircase vs rectangle with the marginal-cost L-strips that telescope to the fleet cost.
- **New circulation section:** four-cycle test as a discrete Clairaut condition, with the Lie-bracket
  analogy stated carefully.
- **Results organized by question:**
  - when the game describes the fleet;
  - whether better physics or information models help;
  - when coordination structure matters.
- Figures were consolidated: the Braess prediction/measurement panels and the circulation now form one
  figure, and the coupling study is its own figure.

## Software
- `rb2/viz/braess_demo.py`: side-by-side replay of the return rush, shortcut closed versus open, for
  single-file channels or slow zones.
- `rb2/policies2.py`: the R3 and L3 strategies. The leader-planner mode (P5) is in `rb2/sim2.py`.
- `analysis/run_v2.py`: one seeded runner for all new experiments. `analysis/round3_stats.py` reproduces
  the statistics, and `analysis/coupling.py` reproduces the coupling study.
- `paper/preview.pdf`: a two-page mathematical preview.
