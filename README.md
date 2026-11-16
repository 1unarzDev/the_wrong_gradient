# The Wrong Gradient

**Potential games, marginal costs, and where the game stops describing the boats**
Liam Kenji Bray

When each autonomous boat does what is best for itself, the fleet descends the wrong landscape. Selfish
moves lower Rosenthal's potential while the fleet pays the total cost; a marginal-cost toll, the discrete
derivative of an edge's total cost, makes the two equal. This repository contains the exact game-theory
results, a simulator of a fleet on courses modelled on the RoboBoat autonomy challenge, the coordination
strategies, and every experiment behind the paper.

- **Paper:** [`paper/main.pdf`](paper/main.pdf)
- **Two-page mathematical preview:** [`paper/preview.pdf`](paper/preview.pdf)
- **Changes in this revision:** [`CHANGELOG.md`](CHANGELOG.md)

## Contents

### Exact layer (`src/game_exact.py`)
The Braess harbour game, Rosenthal's potential, the externality decomposition, marginal-cost tolls, and the
Monderer–Shapley four-cycle test. All of it is verified by exhaustive enumeration in the tests.

### Simulator (`rb2/`)
- **Courses and scoring:** procedurally generated courses with the six RoboBoat 2025 tasks and the
  handbook's §4.2 scoring.
- **Boats:** twin-thruster dynamics (RK4).
- **Perception and communication:** simulated camera and lidar, and per-boat team maps shared over a lossy
  radio.
- **Coordination:** task executors and a coordination layer: sensed deconfliction, recon sectors,
  bonus-aware values, queue-aware return, and bay inference.

### Strategies
`src/policies.py` (frozen with the preregistered study), `rb2/policies2.py`, `rb2/central.py` and `rb2/sim2.py`:

| Code | Strategy |
|---|---|
| P0 | Greedy |
| P1 | Selfish race |
| P2, A1, A2, P3 | Team-aware, with graph or dynamics travel times and naive or age-aware beliefs |
| R3 | Reservations |
| L3 | Heartbeat lease |
| P4 | Central oracle |
| P5 | Leader on radio |

### Return-rush Braess study
- `rb2/braess2.py` builds and runs the return rush: single-file channels or slow zones.
- `analysis/return_rush.py` calibrates and tests the congestion game.
- `analysis/coupling.py` measures cross-route coupling, the coupling transition, and the four-cycle
  circulation of the measured game.

## Setup
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

## Watch it
```bash
.venv/bin/python -m rb2.viz.app2 --seed 11 --policy P4 --ghost P3   # a fleet working the course
.venv/bin/python -m rb2.viz.braess_demo --course queue              # return rush, single-file: paradox vanishes
.venv/bin/python -m rb2.viz.braess_demo --course slowzone           # return rush, slow zones: paradox appears
```
- **Recorded demos** (in `figures/`):
  - `fleet_demo.mp4`: the central planner on course 2170 (2.5 min), with the team-aware fleet as outline boats;
  - `braess_replay_queue.mp4` and `braess_replay_slowzone.mp4`: the return rush with the shortcut closed and
    open, for single-file channels and for slow zones.
- **`app2` keys:** Space pause, 1/2/3 speed, Tab next boat, G cycle the outline (phantom) fleet, P planned
  paths, L lidar, C camera cone, D detections, F follow camera, R new course.
- **Headless video:** add `--record <path>` to either program.

## Reproduce
Every run is seeded. Strategies are compared on identical courses, currents, sensor noise and radio-loss
tapes.

**Preregistered study (abstract course, 400 worlds):**
```bash
.venv/bin/python -m src.experiment --config configs/confirmatory.yaml --headless
.venv/bin/python analysis/run_stats.py --input data/confirmatory/episodes.csv
```

**RoboBoat course: replication (4 and 2 boats), statistics and ablation:**
```bash
PYTHONPATH=. .venv/bin/python analysis/v2_leaderboard.py 200 degraded,ideal 2000 data/v2_replication
NBOATS=2 PYTHONPATH=. .venv/bin/python analysis/v2_leaderboard.py 200 degraded,ideal 2000 data/v2_two_boats
PYTHONPATH=. .venv/bin/python analysis/v2_stats.py
PYTHONPATH=. .venv/bin/python analysis/v2_ablation.py 100
```

**New strategies, lease, one-boat reference and robustness** (plans in `data/analysis_plan_round3.md`):
```bash
R="env PYTHONPATH=. .venv/bin/python analysis/run_v2.py"
$R --out data/v2_new_policies_n4 --pols P5,R3 --comms degraded,ideal --nboats 4
$R --out data/v2_new_policies_n2 --pols P5,R3 --comms degraded,ideal --nboats 2
$R --out data/v2_lease_n4 --pols L3 --comms degraded,ideal --nboats 4
$R --out data/v2_lease_n2 --pols L3 --comms degraded,ideal --nboats 2
$R --out data/v2_one_boat --pols P2,A2 --nboats 1
$R --out data/v2_yields_n4 --pols P3,A2,P4 --nboats 4
$R --out data/v2_robust/harsh_radio --pols P3,A2,R3,P4 --seeds 3100:3200 --dyn '{"p_good_to_bad": 0.25, "drop_good": 0.3, "latency_max_s": 5.0}'
$R --out data/v2_robust/slow_boats --pols P3,A2,R3,P4 --seeds 3100:3200 --dyn '{"v_max": 1.6}'
PYTHONPATH=. .venv/bin/python analysis/round3_stats.py
```

**Return rush and coupling:**
```bash
PYTHONPATH=. .venv/bin/python analysis/return_rush.py              # single-file calibration and test
PYTHONPATH=. .venv/bin/python analysis/return_rush.py --slowzone   # slow-zone variant
PYTHONPATH=. .venv/bin/python analysis/coupling.py                 # coupling, transition, circulation
```

**Figures and paper:**
```bash
.venv/bin/python analysis/make_figures.py braess
.venv/bin/python analysis/v2_figures.py arch physical league ablation
PYTHONPATH=. .venv/bin/python analysis/decision_figure.py && PYTHONPATH=. .venv/bin/python analysis/tasks_figure.py
cd paper && tectonic main.tex && tectonic preview.tex
```

## Headline results
- **Exact:** in the four-boat harbour every selfish move lowers the potential and raises the fleet cost
  (price of anarchy 17/14). Marginal-cost tolls make the potential equal the fleet cost.
- **The physical return rush is not a congestion game.**
  - A single-file channel spaces boats out, so the next queue's delay depends on how boats arrive.
  - The predicted Braess paradox (+14 s) reverses: the fleet is 6.7 s faster with the shortcut open.
  - Four-cycle circulations reach 21.7 s against about 3.6 s of noise, and there is a strict improvement
    cycle, so the measured game has no exact potential and not even an improvement-preserving one.
  - With slow-zone congestion, the paradox appears: +63 s measured against +43 s predicted.
- **Stale information:** every rule that discounts teammates' claims causes duplicated trips. Duplicates per
  run go 2.0 (never discount) < 3.0 (lease) < 4.1 (reservations) < 7.1 (age-aware), and task points fall in
  the same order.
- **Fleet size and information:**
  - Coordination strategy matters with four boats and barely with two, the number the 2026 rules allow.
  - A central planner's advantage comes mostly from privileged information: the radio-fed leader trails
    the oracle by 244 task points with four boats.
