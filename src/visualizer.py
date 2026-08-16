"""Interactive Pygame-CE front end.  The SAME Simulation engine as headless experiments
is stepped here at fixed dt; rendering only reads its state.

  python -m src.visualizer --scenario braess
  python -m src.visualizer --scenario mission --fleet 4 --seed 42 --policy control-aware
  python -m src.visualizer --scenario compare --seed 42 --policy P2 --policy-b P3
  python -m src.visualizer --scenario scale --fleet 12
Headless screenshots:
  python -m src.visualizer --scenario mission --sim-time 60 --screenshot figures/shot.png
By default the world uses the study's frozen distribution (configs/confirmatory.yaml);
--world agile uses the more manoeuvrable default boats.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pygame

from .config import config_from, load_yaml
from .policies import POLICY_INFO
from .sim import Simulation
from .viz import fx
from .viz import theme as T
from .viz.braess_view import BraessScene
from .viz.mission_view import MissionView, card, draw_panel, sparkline, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIASES = {"control-aware": "P3", "graph": "P2", "selfish": "P1", "greedy": "P0",
           "central": "P4", "centralized": "P4"}
POLICIES = ["P0", "P1", "P2", "A1", "A2", "P3", "P4"]
COMMS = ["ideal", "degraded", "none"]
SPEEDS = [("×¼", 0.25), ("×1", 1.0), ("×4", 4.0), ("×16", 16.0)]
MODES = ["braess", "mission", "scale", "compare"]
MODE_TITLE = {"braess": "Exact game", "mission": "Mission", "scale": "Larger fleet", "compare": "Side by side"}
TOP_H, BOT_H, PANEL_W = 52, 0, 400
GHOST_SETS = [[], ["P2"], ["P2", "P4"], ["P1", "P2", "P4"]]
GHOST_COLORS = {"P0": (200, 200, 200), "P1": (255, 170, 160), "P2": (240, 244, 248), "A1": (190, 235, 255),
                "A2": (255, 225, 160), "P3": (255, 235, 170), "P4": (150, 235, 140)}


def study_base(world):
    if world == "agile":
        return {}
    try:
        return dict(load_yaml(os.path.join(ROOT, "configs", "confirmatory.yaml")).get("base", {}))
    except Exception:
        return {}


class App:
    def __init__(self, args):
        self.args = args
        pygame.init()
        flags = 0 if args.screenshot else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((args.width, args.height), flags)
        pygame.display.set_caption("The wrong gradient")
        self.clock = pygame.time.Clock()
        self.mode = args.scenario
        self.seed = args.seed
        self.fleet = args.fleet
        self.policy = ALIASES.get(args.policy, args.policy)
        self.policy_b = ALIASES.get(args.policy_b, args.policy_b)
        self.comm = args.comm
        self.base = study_base(getattr(args, "world", "study"))
        self.speed_idx = 1
        self.paused = False
        self.help = False
        self.toast = None
        self.braess = BraessScene()
        self.views = []
        self.replay = []
        self.belief_idx = None
        self.ghost_idx = 2
        self.follow = True
        self.build()

    # -------------------------------------------------------------------------------
    def layout(self):
        w, h = self.screen.get_size()
        if self.mode == "compare":
            return pygame.Rect(0, TOP_H, w, h - TOP_H - BOT_H), None
        pw = min(PANEL_W, max(300, w // 3))
        return pygame.Rect(0, TOP_H, w - pw, h - TOP_H - BOT_H), pygame.Rect(w - pw, TOP_H, pw, h - TOP_H - BOT_H)

    def make_sim(self, policy, fleet=None):
        ov = {**self.base, "n_boats": fleet or self.fleet, "policy": policy, "comm_mode": self.comm}
        if self.args.episode:
            ov["episode_s"] = self.args.episode
        return Simulation(config_from(ov), self.seed)

    def _rects(self):
        view, _ = self.layout()
        if self.mode != "compare":
            return [view]
        return [pygame.Rect(view.x, view.y, view.w // 2 - 1, view.h),
                pygame.Rect(view.x + view.w // 2 + 1, view.y, view.w // 2 - 1, view.h)]

    def build(self):
        rects = self._rects()
        self.views, self.replay = [], []
        if self.mode in ("mission", "scale"):
            self.views = [MissionView(self.make_sim(self.policy), rects[0])]
            ghosts = [g for g in GHOST_SETS[self.ghost_idx] if g != self.policy]
            self.views[0].ghosts = [(self.make_sim(g), g, GHOST_COLORS[g]) for g in ghosts]
            self.views[0].follow = self.follow
        elif self.mode == "compare":
            self.views = [MissionView(self.make_sim(self.policy), rects[0], f"{self.policy}  {POLICY_INFO[self.policy]}"),
                          MissionView(self.make_sim(self.policy_b), rects[1], f"{self.policy_b}  {POLICY_INFO[self.policy_b]}",
                                      label_right=True)]

    def relayout(self):
        for v, r in zip(self.views, self._rects()):
            v.resize(r)

    def say(self, s, color=T.INK):
        self.toast = (s, color, time.time())

    # -------------------------------------------------------------------------------
    def handle(self, ev):
        if ev.type == pygame.QUIT:
            return False
        if ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            self.relayout(); return True
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                return False
            if k == pygame.K_TAB:
                self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
                self.fleet = 8 if self.mode == "scale" and self.fleet < 8 else (self.args.fleet if self.mode != "scale" else self.fleet)
                self.build(); return True
            if k == pygame.K_h:
                self.help = not self.help; return True
            if k == pygame.K_F2:
                self.snapshot(); return True
            if self.mode == "braess":
                self.braess.handle_key(k); return True
            if k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                self.speed_idx = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 0}[k]
                self.say(f"Speed {SPEEDS[self.speed_idx][0]}")
            elif k == pygame.K_n:
                for v in self.views:
                    for _ in range(int(0.1 / v.sim.cfg.dt)):
                        v.sim.step()
                self.paused = True
            elif k == pygame.K_r:
                self.seed += 1; self.build(); self.say(f"New world {self.seed}")
            elif k == pygame.K_p:
                self.policy = POLICIES[(POLICIES.index(self.policy) + 1) % len(POLICIES)]
                self.build(); self.say(f"{self.policy}: {POLICY_INFO[self.policy]}")
            elif k == pygame.K_y and self.mode == "compare":
                self.policy_b = POLICIES[(POLICIES.index(self.policy_b) + 1) % len(POLICIES)]
                self.build(); self.say(f"Right side {self.policy_b}")
            elif k == pygame.K_c:
                self.comm = COMMS[(COMMS.index(self.comm) + 1) % len(COMMS)]
                self.build(); self.say(f"Comms {self.comm}")
            elif k == pygame.K_b:
                n = self.views[0].sim.n if self.views else 0
                self.belief_idx = 0 if self.belief_idx is None else (self.belief_idx + 1 if self.belief_idx + 1 < n else None)
                for v in self.views:
                    v.belief_of = self.belief_idx
                self.say("Beliefs hidden" if self.belief_idx is None else f"Showing what boat {self.belief_idx + 1} believes")
            elif k == pygame.K_g:
                on = not (self.views and self.views[0].show_graph)
                for v in self.views:
                    v.show_options = v.show_graph = on
                self.say("Planning overlay on" if on else "Planning overlay off")
            elif k == pygame.K_k and self.mode in ("mission", "scale"):
                self.ghost_idx = (self.ghost_idx + 1) % len(GHOST_SETS)
                self.build()
                g = GHOST_SETS[self.ghost_idx]
                self.say("No phantom fleets" if not g else "Phantom fleets: " + ", ".join(g))
            elif k == pygame.K_d:
                for v in self.views:
                    v.show_detail = not getattr(v, "show_detail", False)
            elif k == pygame.K_f:
                self.follow = not self.follow
                for v in self.views:
                    v.follow = self.follow
                    if not self.follow:
                        v.cam.zoom, v.cam.cx, v.cam.cy = 1.0, v.sim.world.width / 2, v.sim.world.height / 2
                self.say("Camera follows the fleet" if self.follow else "Whole course")
            elif k == pygame.K_t:
                for v in self.views:
                    v.show_graph = not v.show_graph
            elif k == pygame.K_v:
                for v in self.views:
                    v.show_sensors = not v.show_sensors
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.fleet = min(16, self.fleet + 1); self.build(); self.say(f"Fleet of {self.fleet}")
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.fleet = max(2, self.fleet - 1); self.build(); self.say(f"Fleet of {self.fleet}")
            elif k == pygame.K_e:
                self.export_replay()
            elif k == pygame.K_o:
                self.open_failure_case()
        if ev.type == pygame.MOUSEWHEEL and self.views:
            for v in self.views:
                if v.cam.rect.collidepoint(pygame.mouse.get_pos()):
                    mx, my = pygame.mouse.get_pos()
                    wx, wy = v.cam.to_world(mx, my)
                    v.cam.zoom = max(1.0, min(4.0, v.cam.zoom * (1.15 if ev.y > 0 else 1 / 1.15)))
                    nx, ny = v.cam.to_world(mx, my)
                    v.cam.cx += wx - nx; v.cam.cy += wy - ny
        if ev.type == pygame.MOUSEMOTION and ev.buttons[0] and self.views:
            for v in self.views:
                if v.cam.rect.collidepoint(ev.pos):
                    v.cam.cx -= ev.rel[0] / v.cam.scale; v.cam.cy += ev.rel[1] / v.cam.scale
        return True

    def snapshot(self, path=None):
        os.makedirs("figures/screenshots", exist_ok=True)
        path = path or f"figures/screenshots/{self.mode}_{int(time.time())}.png"
        pygame.image.save(self.screen, path)
        self.say(f"Saved {path}")
        return path

    def export_replay(self):
        if not self.views:
            return
        os.makedirs("replays", exist_ok=True)
        out = [dict(seed=v.sim.seed, config=v.sim.cfg.to_dict(), config_hash=v.sim.cfg.hash(), t=v.sim.t,
                    samples=self.replay, events=[e.__dict__ for e in v.sim.events if e.kind != "decide"],
                    note="Deterministic: Simulation(config, seed) reproduces this episode exactly.")
               for v in self.views]
        path = f"replays/replay_{self.mode}_seed{self.seed}_{self.policy}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(out, f, default=float)
        self.say(f"Saved {path}")

    def open_failure_case(self):
        p = os.path.join(ROOT, "data/failure_cases.json")
        if not os.path.exists(p):
            self.say("No failure cases yet. Run analysis/run_stats.py first.", T.WARN); return
        cases = json.load(open(p))
        if not cases:
            return
        self._fc = (getattr(self, "_fc", -1) + 1) % len(cases)
        c = cases[self._fc]
        self.seed, self.comm = int(c["seed"]), c["comm_mode"]
        self.policy, self.policy_b = c["policy_a"], c["policy_b"]
        self.fleet = int(c.get("n_boats", 4))
        self.mode = "compare"; self.build()
        self.say(f"Failure case {self._fc + 1} of {len(cases)}: {c.get('note', '')}", T.WARN)

    # -------------------------------------------------------------------------------
    def step_sims(self, real_dt):
        if self.mode == "braess" or self.paused or not self.views:
            return 0.0
        sim0 = self.views[0].sim
        target = sim0.t + SPEEDS[self.speed_idx][1] * real_dt
        before = sim0.t
        while sim0.t + 1e-9 < target and not sim0.done:
            for v in self.views:
                v.sim.step()
                for g in v.ghosts:
                    g[0].step()
        if int(sim0.t * 2) != int(before * 2):
            self.replay.append([round(sim0.t, 2)] + [[round(float(a), 2) for a in v.sim.S[:3].ravel()] for v in self.views])
        return sim0.t - before

    def draw_top(self):
        w = self.screen.get_width()
        pygame.draw.rect(self.screen, T.PAPER, (0, 0, w, TOP_H))
        pygame.draw.line(self.screen, T.RULE, (0, TOP_H - 1), (w, TOP_H - 1))
        text(self.screen, "The wrong gradient", (24, TOP_H // 2), 17, T.INK, True, "ml")
        x = 230
        for m in MODES:
            act = m == self.mode
            r = text(self.screen, MODE_TITLE[m], (x, TOP_H // 2), 14, T.INK if act else (90, 104, 130), act, "ml")
            if act:
                pygame.draw.line(self.screen, T.ACCENT, (r.x, TOP_H - 3), (r.right, TOP_H - 3), 2)
            x = r.right + 30
        if self.mode != "braess" and self.views:
            t = self.views[0].sim.t
            clock = f"{int(t // 60)}:{int(t % 60):02d}"
            if self.paused:
                clock = "paused  " + clock
            elif self.speed_idx != 1:
                clock = f"{SPEEDS[self.speed_idx][0]}  " + clock
            text(self.screen, clock, (w - 24, TOP_H // 2), 15, T.WARN if self.paused else T.INK, False, "mr")

    def draw_timeline(self):
        if BOT_H <= 0:
            return
        w, h = self.screen.get_size()
        y0 = h - BOT_H
        pygame.draw.rect(self.screen, T.PAPER, (0, y0, w, BOT_H))
        pygame.draw.line(self.screen, T.RULE, (0, y0), (w, y0))
        if self.mode == "braess" or not self.views:
            text(self.screen, "Every number here is computed with exact fractions and checked by exhaustive "
                 "enumeration (src/game_exact.py).", (24, y0 + BOT_H // 2), 12, T.INK2, False, "ml")
            return
        sim = self.views[0].sim
        E = sim.cfg.episode_s
        x0, x1 = 24, (w if self.mode == "compare" else w - PANEL_W) - 24
        X = lambda t: x0 + (x1 - x0) * t / E
        n = len(sim.tasks)
        hh = max(1, (BOT_H - 18) // max(1, n))
        for k, tk in enumerate(sim.tasks):
            y = y0 + 9 + k * hh
            pygame.draw.line(self.screen, T.RULE, (X(tk.release), y), (X(min(E, tk.deadline)), y), max(1, hh - 1))
            if tk.completed_at is not None:
                pygame.draw.line(self.screen, T.BOAT_COLORS[tk.completed_by % 16], (X(tk.release), y),
                                 (X(tk.completed_at), y), max(1, hh - 1))
        cx = X(sim.t)
        pygame.draw.line(self.screen, T.INK, (cx, y0 + 6), (cx, h - 6), 1)

    def draw_help(self):
        w, h = self.screen.get_size()
        r = pygame.Rect(0, 0, 600, 440); r.center = (w // 2, h // 2)
        card(self.screen, r)
        rows = [("Tab", "switch view: exact game, mission, larger fleet, side by side"),
                ("Space", "pause or resume"), ("1 2 3, 4", "speed ×1, ×4, ×16, or ×¼"), ("N", "step 0.1 s"),
                ("P / Y", "change policy (left / right side)"), ("C", "comms: ideal, degraded, none"),
                ("R", "new world"), ("+ / −", "fleet size"),
                ("G", "planning overlay: routes, time ticks, options weighed"),
                ("K", "phantom fleets: same course, other strategies"),
                ("B", "show what one boat believes about the others"),
                ("T", "remaining-time labels"), ("D", "diagnostic detail in the side panel"), ("V", "sensor range"), ("O", "open a recorded failure case"),
                ("F", "follow camera or whole course"), ("Wheel, drag", "zoom and pan"), ("F2, E", "screenshot, export replay")]
        text(self.screen, "Controls", (r.x + 28, r.y + 22), 18, T.INK, True)
        y = r.y + 58
        for k, d in rows:
            text(self.screen, k, (r.x + 28, y), 13, T.INK, True)
            text(self.screen, d, (r.x + 150, y), 13, T.INK2)
            y += 24

    def draw_compare_overlay(self, view):
        ms = [v.live_metrics() for v in self.views]
        for v, m in zip(self.views, ms):
            rr = v.cam.rect
            r = pygame.Rect(0, 0, 300, 64)
            r.bottomleft = (rr.x + 16, rr.bottom - 16)
            card(self.screen, r)
            text(self.screen, f"{m['welfare']:+.1f}", (r.x + 16, r.y + 10), 24, T.INK, True)
            text(self.screen, f"{m['points']} points, {m['completed']} done, {m['expired']} expired",
                 (r.x + 110, r.y + 12), 12, T.INK2)
            text(self.screen, f"{sum(mm.switches for mm in v.sim.minds)} target switches, {m['collisions']} contacts",
                 (r.x + 110, r.y + 32), 12, T.INK2)
        d = ms[1]["welfare"] - ms[0]["welfare"]
        r = pygame.Rect(0, 0, 260, 70); r.midbottom = (view.centerx, view.bottom - 16)
        card(self.screen, r)
        text(self.screen, "Right minus left", (r.centerx, r.y + 14), 12, T.INK2, False, "c")
        text(self.screen, f"{d:+.1f}", (r.centerx, r.y + 42), 24, T.INK, True, "c")

    def draw(self, real_dt, sim_dt):
        self.screen.fill(T.PAPER)
        view, panel = self.layout()
        if self.mode == "braess":
            self.braess.update(real_dt)
            self.braess.draw(self.screen, view, panel)
        else:
            for v in self.views:
                v.update(real_dt, sim_dt)
                v.draw(self.screen)
            if self.mode == "compare":
                pygame.draw.line(self.screen, T.RULE, (view.centerx, view.y), (view.centerx, view.bottom), 1)
                self.draw_compare_overlay(view)
            else:
                draw_panel(self.screen, panel, self.views[0])
            if self.views[0].sim.done:
                text(self.screen, "Episode complete. Press R for a new world.", (view.centerx, view.y + 26), 15, T.INK, True, "c")
        self.draw_top()
        self.draw_timeline()
        if self.toast and time.time() - self.toast[2] < 2.5:
            s, c, _ = self.toast
            img = T.font(13).render(s, True, c)
            r = img.get_rect().inflate(28, 16); r.midtop = (view.centerx, view.y + 14)
            card(self.screen, r)
            self.screen.blit(img, (r.x + 14, r.y + 8))
        if self.help:
            self.draw_help()

    def run(self):
        if self.args.screenshot:
            return self.run_scripted()
        running = True
        while running:
            real_dt = min(0.05, self.clock.tick(60) / 1000.0)
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            if self.layout()[0].size != (self.views[0].cam.rect.size if self.views and self.mode != "compare" else self.layout()[0].size):
                self.relayout()
            self.draw(real_dt, self.step_sims(real_dt))
            pygame.display.flip()
        pygame.quit()

    def run_scripted(self):
        """Headless: fast-forward to --sim-time, render a few frames, save."""
        a = self.args
        if self.mode == "braess":
            if a.marginal:
                self.braess.handle_key(pygame.K_m)
            if a.cycle:
                self.braess.show_cycle = True
            for _ in range(a.braess_moves):
                self.braess.step()
            if self.braess.caption:
                self.braess.caption = (*self.braess.caption[:3], 0.0)
            self.braess.t = 0.8
            for _ in range(max(1, a.frames)):
                self.draw(1 / 60, 0)
        else:
            sim0 = self.views[0].sim
            lead = max(0.0, a.sim_time - 6.0)
            while sim0.t < lead:
                for v in self.views:
                    v.sim.step()
                    for g in v.ghosts:
                        g[0].step()
            for v in self.views:
                v.ev_idx = len(v.sim.events)
                v.cam.zoom = a.zoom
                v.belief_of = a.beliefs if a.beliefs >= 0 else None
            for _ in range(a.frames):
                d = 0.0
                if sim0.t < a.sim_time:
                    for _k in range(int(5 * (1 / 60) / sim0.cfg.dt)):
                        for v in self.views:
                            v.sim.step()
                            for g in v.ghosts:
                                g[0].step()
                    d = 5 / 60
                if a.focus_boat >= 0:
                    for v in self.views:
                        v.cam.cx, v.cam.cy = float(v.sim.S[0, a.focus_boat]), float(v.sim.S[1, a.focus_boat])
                self.draw(1 / 60, d)
            if a.help_overlay:
                self.help = True; self.draw(0, 0)
        os.makedirs(os.path.dirname(a.screenshot) or ".", exist_ok=True)
        pygame.image.save(self.screen, a.screenshot)
        print("saved", a.screenshot)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=MODES, default="mission")
    ap.add_argument("--fleet", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", default="P3")
    ap.add_argument("--policy-b", default="P2")
    ap.add_argument("--comm", choices=COMMS, default="degraded")
    ap.add_argument("--world", choices=["study", "agile"], default="study")
    ap.add_argument("--episode", type=float, default=None)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--sim-time", type=float, default=40.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--braess-moves", type=int, default=1)
    ap.add_argument("--marginal", action="store_true")
    ap.add_argument("--cycle", action="store_true")
    ap.add_argument("--help-overlay", action="store_true")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--focus-boat", type=int, default=-1)
    ap.add_argument("--beliefs", type=int, default=-1, help="show what boat k (0-based) believes")
    args = ap.parse_args(argv)
    args.policy = ALIASES.get(args.policy, args.policy)
    args.policy_b = ALIASES.get(args.policy_b, args.policy_b)
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # force headless even if the session sets wayland
    App(args).run()


if __name__ == "__main__":
    main()
