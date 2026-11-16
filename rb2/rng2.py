"""Named, independent RNG streams for v2 (same scheme as src/rng.py, 'v2:' prefixed names)."""
import zlib

import numpy as np

STREAMS = ("course", "current", "disturb", "sensors_cam", "sensors_lidar", "comms", "light", "tiebreak")


def stream(seed: int, name: str) -> np.random.Generator:
    if name not in STREAMS:
        raise KeyError(name)
    key = zlib.crc32(("v2:" + name).encode())
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=(key,)))
