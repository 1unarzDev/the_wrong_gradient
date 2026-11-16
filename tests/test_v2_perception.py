"""Perception layer for course v2: sensors, tracker, team map (fast)."""
import math
import time

import numpy as np
import pytest

from rb2.course import Course, Obj
from rb2.generate import generate_course
from rb2.geom import Obstacles
from rb2.sensors import (BANNER_TRUTH_BASE, CAM_CLASSES, Camera, Detection, Lidar, confusion_matrix, p_detect)
from rb2.teammap import Message2, TeamMap
from rb2.tracking import Tracker


def mini(objs, seed=11):
    """Open-water test course with the given objects (list of (cls, x, y, kw))."""
    from rb2.course import BUOY_RADIUS
    out = []
    for i, (cls, x, y, kw) in enumerate(objs):
        r = kw.pop("r", BUOY_RADIUS.get(cls, 0.8))
        out.append(Obj(i, cls, x, y, r, **kw))
    return Course(seed, 60.0, 60.0, (0.0, 0.0, 0.0), out, [], "circle", "red")


def ci_ok(k, n, p, z=4.0):
    return abs(k / n - p) <= z * math.sqrt(p * (1 - p) / n) + 1e-9


# ---- camera ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d,off_deg", [(3.0, 0.0), (10.0, 20.0), (16.0, 40.0)])
def test_detection_rate_matches_p_detect(d, off_deg):
    off = math.radians(off_deg)
    c = mini([("yellow", 20 + d * math.cos(off), 20 + d * math.sin(off), {})])
    cam, obs = Camera(), Obstacles(c)
    n = 1500
    k = sum(bool(cam.sense(0.0, t, 0, (20.0, 20.0, 0.0), c, obs)) for t in range(n))
    assert ci_ok(k, n, float(p_detect(d, off)))


def test_outside_fov_and_range_never_detected():
    c = mini([("yellow", 20, 30, {}), ("red", 40, 20, {})])          # 90 deg off-axis; 20 m away
    cam = Camera()
    assert not any(cam.sense(0.0, t, 0, (20.0, 20.0, 0.0), c) for t in range(200))


def test_occluded_object_never_detected():
    c = mini([("vessel_black", 28, 20, dict(heading=math.pi / 2, length=1.6)), ("yellow", 32, 20, {})])
    cam, lid, obs = Camera(), Lidar(), Obstacles(c)
    seen_v = 0
    for t in range(400):
        dets = cam.sense(0.0, t, 0, (20.0, 20.0, 0.0), c, obs)
        assert all(det.truth != 1 for det in dets)
        seen_v += any(det.truth == 0 for det in dets)
        assert all(det.truth != 1 for det in lid.sense(0.0, t, 0, (20.0, 20.0, 0.0), c, obs))
    assert seen_v > 200
    # control: move the vessel away -> the buoy becomes visible
    c2 = mini([("vessel_black", 28, 25, dict(heading=math.pi / 2, length=1.6)), ("yellow", 32, 20, {})])
    assert any(det.truth == 1 for t in range(50) for det in cam.sense(0.0, t, 0, (20.0, 20.0, 0.0), c2))


@pytest.mark.parametrize("cls,d", [("yellow", 10.0), ("red", 16.0), ("blue", 12.0)])
def test_confusion_frequencies(cls, d):
    c = mini([(cls, 20 + d, 20, {})])
    cam, obs = Camera(), Obstacles(c)
    counts = np.zeros(len(CAM_CLASSES))
    for t in range(4000):
        for det in cam.sense(0.0, t, 0, (20.0, 20.0, 0.0), c, obs):
            counts[CAM_CLASSES.index(max(det.probs, key=det.probs.get))] += 1
            assert abs(sum(det.probs.values()) - 1) < 1e-9
    n = counts.sum()
    row = confusion_matrix(d)[CAM_CLASSES.index(cls)]
    assert n > 1000
    for j, p in enumerate(row):
        assert ci_ok(counts[j], n, p, z=4.5) or abs(counts[j] / n - p) < 0.01, (CAM_CLASSES[j], counts[j] / n, p)
    if cls == "red":   # long range: orange vessel confusion exists
        assert row[CAM_CLASSES.index("vessel_orange")] > 0.03


def test_determinism_and_crn():
    c = generate_course(4)
    obs = Obstacles(c)
    pose = (c.start[0], c.start[1] + 2, math.pi / 2)
    a = Camera().sense(1.0, 17, 2, pose, c, obs)
    cam = Camera()
    for t in range(30):                                    # unrelated calls in between (another "policy")
        cam.sense(0.0, t, 2, (30.0, 30.0, 1.0), c, obs)
    b = cam.sense(1.0, 17, 2, pose, c)                     # also with a fresh obstacle build
    assert a == b and len(a) > 0
    other = Camera().sense(1.0, 17, 3, pose, c, obs)
    assert [(d.x, d.y) for d in other] != [(d.x, d.y) for d in a]
    la, lb = Lidar().sense(1.0, 17, 2, pose, c, obs), Lidar().sense(1.0, 17, 2, pose, c)
    assert la == lb and len(la) > 0
    pa, pb = Lidar().scan_points(pose, c, obs, 17, 2), Lidar().scan_points(pose, c, None, 17, 2)
    assert np.array_equal(pa, pb)
    r = np.hypot(pa[:, 0] - pose[0], pa[:, 1] - pose[1])
    assert len(pa) > 10 and r.max() < 20.2


def test_sensing_cost():
    c = generate_course(2)
    obs = Obstacles(c)
    cam, lid = Camera(), Lidar()
    rng = np.random.default_rng(0)
    poses = [(rng.uniform(5, 75), rng.uniform(5, 50), rng.uniform(-3, 3)) for _ in range(300)]
    t0 = time.perf_counter()
    for k, p in enumerate(poses):
        cam.sense(0.0, k, 0, p, c, obs)
        lid.sense(0.0, k, 0, p, c, obs)
    per = (time.perf_counter() - t0) / len(poses)
    assert per < 2e-3, per


# ---- tracker ---------------------------------------------------------------------------------------------------
def _drive(c, tracker, passes, cam, lid, obs, t0=0.0, tick0=0, boat=0, extra=None):
    """Boat drives straight lines along y = 20 (east then west); camera 5 Hz, lidar 10 Hz."""
    t, tick = t0, tick0
    for k in range(passes):
        xs = np.arange(2.0, 58.0, 0.12) if k % 2 == 0 else np.arange(58.0, 2.0, -0.12)
        psi = 0.0 if k % 2 == 0 else math.pi
        for x in xs:
            pose = (float(x), 20.0, psi)
            dets = lid.sense(t, tick, boat, pose, c, obs)
            if tick % 2 == 0:
                dets += cam.sense(t, tick, boat, pose, c, obs)
            if extra:
                dets += extra(t, tick)
            tracker.update(dets, t)
            t += 0.1; tick += 1
    return t, tick


def test_tracker_counts_yellow_without_duplicates():
    xs = [12.0, 19.0, 27.5, 36.0, 44.0]
    objs = [("yellow", x, 20 + (2.5 if i % 2 else -2.5), {}) for i, x in enumerate(xs)]
    objs += [("red", 15.0, 26.0, {}), ("green", 30.0, 14.0, {}), ("black", 50.0, 24.0, {})]
    c = mini(objs)
    obs, cam, lid = Obstacles(c), Camera(), Lidar()
    tr = Tracker()
    rng = np.random.default_rng(5)

    def clutter(t, tick):     # isolated false camera detections claiming 'yellow'
        if tick % 7:
            return []
        p = {k: 0.02 for k in CAM_CLASSES}; p["yellow"] = 1 - 0.02 * (len(CAM_CLASSES) - 1)
        return [Detection(t, 0, "cam", float(rng.uniform(2, 58)), float(rng.uniform(5, 35)), 0.3, p, {}, -1)]

    t, tick = _drive(c, tr, 2, cam, lid, obs, extra=clutter)
    assert tr.count("yellow") == 5
    t, tick = _drive(c, tr, 2, cam, lid, obs, t, tick, extra=clutter)     # repeat views
    assert tr.count("yellow") == 5
    assert tr.count("black") == 1 and tr.count("red") == 1
    poly = [(0, 0), (30, 0), (30, 40), (0, 40)]
    assert tr.count("yellow", poly) == 3
    assert tr.count("yellow", lambda x, y: x > 30) == 2
    # every confirmed yellow track is one distinct true object (evaluation-only truth)
    truths = [max(t_.truth_votes, key=t_.truth_votes.get) for t_ in tr.confirmed() if t_.best_class() == "yellow"]
    assert sorted(truths) == [0, 1, 2, 3, 4]


def test_unconfirmed_tracks_age_out():
    tr = Tracker()
    p = {k: 0.0 for k in CAM_CLASSES}; p["yellow"] = 1.0
    tr.update([Detection(0.0, 0, "cam", 5.0, 5.0, 0.2, p, {}, -1)], 0.0)
    assert len(tr.tracks) == 1 and not tr.tracks[0].confirmed
    tr.update([], 4.5)
    assert tr.tracks == []


# ---- team map ---------------------------------------------------------------------------------------------------
def test_teammap_merge_same_object_once():
    from src.comms import CommTape, Network
    from src.config import SimConfig
    c = mini([("yellow", 30.0, 22.5, {}), ("black", 40.0, 17.5, {})])
    obs, cam, lid = Obstacles(c), Camera(), Lidar()
    maps = [TeamMap(b, c.width, c.height) for b in range(2)]
    for b, m in enumerate(maps):
        _drive(c, m.tracker, 1, cam, lid, obs, boat=b, tick0=1000 * b)
        m.mark_coverage((30.0, 20.0, 0.0))
    assert all(m.tracker.count("yellow") == 1 for m in maps)
    net = Network(CommTape(0, SimConfig(n_boats=2, comm_mode="ideal")))
    for b, m in enumerate(maps):
        net.broadcast(0, Message2(b, 10.0, 30.0, 20.0, 0.0, 1.0, None, 0.0, frozenset(), m.digest(10.0)), 2)
    got = net.deliver(10.0)
    assert len(got) == 2
    for r, msg in got:
        maps[r].merge(msg.digest, msg.sender, msg.t_sent)
    for m in maps:
        assert len(m.remote) >= 2
        assert len(m.objects("yellow")) == 1 and m.count("yellow") == 1 and m.count("black") == 1
        assert {s for s, _ in m.objects("yellow")[0].members} == {"own", 1 - m.boat}
    # coverage: told tiles reduce the unexplored set; repeat digests are idempotent
    m0 = maps[0]
    n_before = len(m0.unexplored_tiles())
    m0.merge(maps[1].digest(11.0), 1, 11.0)
    assert m0.count("yellow") == 1 and len(m0.unexplored_tiles()) <= n_before


def test_banner_identified_within_5m():
    for seed in (1, 2, 3):
        c = generate_course(seed)
        m = c.module("T3")[0]
        bay = next(b for b in m.bays if (b.shape, b.color) == (c.day_shape, c.day_color) and not b.occupied)
        out = bay.heading + math.pi
        bx, by = bay.banner[0] + 4.5 * math.cos(out), bay.banner[1] + 4.5 * math.sin(out)
        tm = TeamMap(0, c.width, c.height)
        cam, obs = Camera(), Obstacles(c)
        for k in range(15):
            pose = (bx, by, bay.heading)
            tm.observe(k * 0.2, pose, cam.sense(k * 0.2, k, 0, pose, c, obs))
        bays = tm.banner_bays()
        x, y, probs = min(bays, key=lambda b: math.hypot(b[0] - bay.banner[0], b[1] - bay.banner[1]))
        assert math.hypot(x - bay.banner[0], y - bay.banner[1]) < 0.5
        best = max(probs, key=probs.get)
        assert best == (c.day_shape, c.day_color) and probs[best] > 0.9, (seed, best, probs[best])


def test_banner_truth_ids_are_offset():
    c = generate_course(1)
    m = c.module("T3")[0]
    bay = m.bays[m.data["match"]]
    out = bay.heading + math.pi
    pose = (bay.banner[0] + 3 * math.cos(out), bay.banner[1] + 3 * math.sin(out), bay.heading)
    dets = [d for k in range(10) for d in Camera().sense(0.0, k, 0, pose, c)]
    assert any(d.truth == BANNER_TRUTH_BASE + m.data["match"] and "banner" in d.attrs for d in dets)


def test_light_state_follows_truth():
    c = mini([("light_panel", 26.0, 20.0, dict(r=0.25))])
    cam, obs = Camera(), Obstacles(c)
    tm = TeamMap(0, c.width, c.height)
    pose = (20.0, 20.0, 0.0)
    states = []
    for k in range(60):
        truth = "red" if k < 30 else "green"
        t = k * 0.2
        tm.observe(t, pose, cam.sense(t, k, 0, pose, c, obs, light=truth))
        states.append(tm.light_state()[0] if tm.light_state() else None)
    assert states[25:30] == ["red"] * 5
    assert states[-10:] == ["green"] * 10
    # a teammate learns the state from the digest
    tm2 = TeamMap(1, c.width, c.height)
    tm2.merge(tm.digest(12.0), 0, 12.0)
    assert tm2.light_state()[0] == "green"
