"""v1 freeze: the finished study must stay bit-reproducible while v2 (rb2/) is built beside it."""
import pandas as pd
import pytest

from src.config import config_from, load_yaml
from src.experiment import source_hash
from src.sim import Simulation

PINNED_SOURCE_HASH = "af5adf4f79996eb1"   # src/ must not change; v2 lives in rb2/
SKIP = {"cpu_ms_mean", "cpu_ms_max", "wall_s", "condition"}


def test_source_hash_pinned():
    assert source_hash() == PINNED_SOURCE_HASH, "src/ changed: v1 must stay frozen (put v2 code in rb2/)"


@pytest.mark.parametrize("seed", [20000, 20001, 20002])
@pytest.mark.parametrize("policy", ["P3", "P4"])
@pytest.mark.parametrize("cond", ["ideal", "degraded"])
def test_confirmatory_rows_reproduce(seed, policy, cond):
    spec = load_yaml("data/confirmatory/confirmatory.frozen.yaml")
    ref = pd.read_csv("data/confirmatory/episodes.csv").set_index(["seed", "policy", "condition"]).loc[(seed, policy, cond)]
    m = Simulation(config_from({**spec["base"], "comm_mode": cond, "policy": policy}), seed).run()
    for k, v in m.items():
        if k in SKIP or k not in ref.index:
            continue
        if isinstance(v, float):
            assert v == pytest.approx(float(ref[k]), rel=0, abs=1e-9) or (pd.isna(ref[k]) and v != v), k
        else:
            assert str(v) == str(ref[k]), k
