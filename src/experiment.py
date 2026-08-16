"""Headless batch experiments (same engine as the GUI, rendering disabled).

  python -m src.experiment --config configs/pilot.yaml --headless
  python -m src.experiment --config configs/confirmatory.yaml --headless

YAML schema:
  name: pilot
  out_dir: data/pilot
  seeds: {start: 1000, count: 50}      # or explicit list
  policies: [P0, P1, P2, A1, A2, P3, P4]
  base: {n_boats: 4, ...}              # SimConfig overrides shared by all conditions
  conditions:                          # each condition = label + overrides
    - {label: degraded, overrides: {comm_mode: degraded}}
  workers: 18

Each job (seed, condition, policy) appends one row to episodes.csv immediately, so an
interrupted run resumes where it stopped.  ETA predictions vs realized completion times
go to eta_log.csv.  A manifest.json records config hashes, source hash, versions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

from .config import config_from, load_yaml


def source_hash() -> str:
    h = hashlib.sha256()
    root = os.path.dirname(os.path.abspath(__file__))
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".py"):
                with open(os.path.join(dp, f), "rb") as fh:
                    h.update(f.encode()); h.update(fh.read())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "none"


def expand_seeds(spec):
    if isinstance(spec, dict):
        return list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))
    return [int(s) for s in spec]


def _run_job(job):
    seed, label, overrides, policy = job
    from .sim import Simulation
    cfg = config_from({**overrides, "policy": policy})
    t0 = time.perf_counter()
    sim = Simulation(cfg, seed)
    m = sim.run()
    m["wall_s"] = time.perf_counter() - t0
    m["condition"] = label
    eta = [dict(d, seed=seed, condition=label, policy=policy) for d in sim.decisions]
    return m, eta


def run(config_path: str, workers: int = None, limit: int = None):
    spec = load_yaml(config_path)
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    seeds = expand_seeds(spec["seeds"])
    if limit:
        seeds = seeds[:limit]
    base = spec.get("base", {})
    conds = spec.get("conditions") or [{"label": "default", "overrides": {}}]
    jobs = []
    hashes = {}
    for c in conds:
        ov = {**base, **c.get("overrides", {})}
        for p in spec["policies"]:
            hashes[f"{c['label']}/{p}"] = config_from({**ov, "policy": p}).hash()
        for s in seeds:
            for p in spec["policies"]:
                jobs.append((s, c["label"], ov, p))
    ep_path = os.path.join(out, "episodes.csv")
    eta_path = os.path.join(out, "eta_log.csv")
    done = set()
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for r in csv.DictReader(f):
                done.add((int(r["seed"]), r["condition"], r["policy"]))
    todo = [j for j in jobs if (j[0], j[1], j[3]) not in done]
    manifest = dict(name=spec.get("name"), config_file=config_path,
                    config_file_sha256=hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
                    source_hash=source_hash(), git_commit=git_commit(),
                    python=sys.version.split()[0], platform=platform.platform(),
                    seeds=[seeds[0], seeds[-1], len(seeds)], policies=spec["policies"],
                    conditions=conds, base=base, config_hashes=hashes,
                    started=time.strftime("%Y-%m-%dT%H:%M:%S"), n_jobs=len(jobs),
                    already_done=len(jobs) - len(todo))
    try:
        import numpy, scipy, pandas, pygame  # noqa
        manifest["versions"] = dict(numpy=numpy.__version__, scipy=scipy.__version__,
                                    pandas=pandas.__version__, pygame=pygame.version.ver)
    except Exception:
        pass
    mpath = os.path.join(out, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    if old and old.get("source_hash") != manifest["source_hash"] and done:
        print(f"WARNING: source changed since earlier rows ({old.get('source_hash')} -> "
              f"{manifest['source_hash']}); rows would mix code versions.", file=sys.stderr)
        if not os.environ.get("ALLOW_MIXED"):
            raise SystemExit("refusing to resume with different source; set ALLOW_MIXED=1 to override")
    manifest["runs"] = old.get("runs", []) + [dict(started=manifest["started"], todo=len(todo),
                                                   source_hash=manifest["source_hash"])]
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"{spec.get('name')}: {len(jobs)} jobs, {len(todo)} to run, workers={workers or spec.get('workers', 8)}")
    if not todo:
        return ep_path
    ep_new = not os.path.exists(ep_path)
    eta_new = not os.path.exists(eta_path)
    fe = open(ep_path, "a", newline="")
    ft = open(eta_path, "a", newline="")
    we = wt = None
    t0 = time.time()
    with Pool(workers or spec.get("workers", 8)) as pool:
        for k, (m, eta) in enumerate(pool.imap_unordered(_run_job, todo, chunksize=1)):
            if we is None:
                we = csv.DictWriter(fe, fieldnames=list(m.keys()))
                if ep_new:
                    we.writeheader()
            we.writerow(m); fe.flush()
            if eta:
                if wt is None:
                    wt = csv.DictWriter(ft, fieldnames=list(eta[0].keys()))
                    if eta_new:
                        wt.writeheader()
                wt.writerows(eta); ft.flush()
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)}  {el:.0f}s elapsed, ~{el / (k + 1) * (len(todo) - k - 1):.0f}s left",
                      flush=True)
    fe.close(); ft.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(manifest, open(mpath, "w"), indent=2)
    return ep_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--headless", action="store_true", help="(always headless; flag kept for clarity)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N seeds (smoke test)")
    a = ap.parse_args(argv)
    run(a.config, a.workers, a.limit)


if __name__ == "__main__":
    main()
