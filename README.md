# The Wrong Gradient

Selfish route choice in a fleet of autonomous boats, from exact game theory to a physics simulator.

- `src/game_exact.py`: the Braess harbour game, Rosenthal's potential, the externality decomposition,
  marginal-cost tolls, and the Monderer–Shapley four-cycle test, checked by exhaustive enumeration.
- `src/`: a simulator of twin-thruster boats (RK4 dynamics, currents, planner and line-of-sight guidance,
  lossy radio) with decentralized and central task-allocation strategies, and a Pygame front end.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.visualizer --scenario braess
```
