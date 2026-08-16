"""Independent, named RNG streams derived from one world seed.

Every exogenous source of randomness (map, tasks, currents, disturbances, sensors,
communication tapes, tie-breaks) gets its own stream so that changing the POLICY can
never shift the random numbers seen by the world (common random numbers)."""
import zlib

import numpy as np

STREAMS = ("map", "tasks", "currents", "disturb", "sensors", "comms", "tiebreak")


def stream(world_seed: int, name: str) -> np.random.Generator:
    if name not in STREAMS:
        raise KeyError(name)
    key = zlib.crc32(name.encode())
    return np.random.default_rng(np.random.SeedSequence(entropy=int(world_seed),
                                                        spawn_key=(key,)))
