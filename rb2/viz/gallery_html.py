"""Write figures/v2_preview/index.html: a browsable gallery of the checkpoint previews."""
import html
import os
import time

OUT = "figures/v2_preview"
ITEMS = [
    ("tasks_gallery", "The six RoboBoat tasks",
     "Each official task up close, with a boat's camera cone, lidar ring and returns, and detection labels with confidence. "
     "T3 shows the banner of the day with an empty matching bay, a decoy, and occupied bays; T4 the light panel; T5 a water stream on a triangle and a ball drop on a plus."),
    ("courses", "Procedural courses",
     "Three seeds at two difficulties on open water: Poisson-disc module placement, a Catmull-Rom spline path with yellow buoys inside it, "
     "a cube-grid dock on an edge, a compact speed challenge, scattered delivery vessels; every task entry validated as reachable."),
    ("return_rush", "Return rush (Braess paradox as a course)",
     "After the tasks, every boat races to the home gate. Marinas squeeze one half of each corridor into a single-file channel next to the "
     "oil-spill boom; the other halves are open water. Opening a channel through the boom tempts every boat onto both slow channels."),
    ("boats", "Boats", "Team colours chosen to read on pale water and never be confused with buoy colours; the number stays upright at any heading."),
]


def main():
    files = sorted(os.listdir(OUT))
    cards = []
    if "lidar_variants.mp4" in files:
        cards.append(f"""<section><h2>Lidar: three reserved treatments (pick one)</h2><p>{html.escape(
            "Same scene, real 10 Hz scan timing, no beam drawn. A: live points that refresh almost instantly. "
            "B: the same scan with a faint 1.5 s memory, so the picture accumulates. C: no points, just a faint outline of what the lidar can see.")}</p>
<video src="lidar_variants.mp4?v={int(time.time())}" autoplay loop muted playsinline style="width:100%;border-radius:6px"></video></section>""")
    if False and "lidar_sweep.mp4" in files:
        cards.append(f"""<section><h2>Sensor vision: the lidar sweep</h2><p>{html.escape(
            "A sweeping beam (slowed down so the eye can follow it). Each return appears where the laser touches a real "
            "surface (dock edges, hulls, buoys) and fades. Lidar gives positions only; colours come from the camera cone.")}</p>
<video src="lidar_sweep.mp4?v={int(time.time())}" autoplay loop muted playsinline style="width:100%;border-radius:6px"></video></section>""")
    for key, title, desc in ITEMS:
        f = next((x for x in files if x.startswith(key) and x.endswith(".png")), None)
        if not f:
            continue
        cards.append(f"""<section><h2>{html.escape(title)}</h2><p>{html.escape(desc)}</p>
<a href="{f}"><img src="{f}?v={int(time.time())}" alt="{html.escape(title)}"></a></section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>RoboBoat v2 preview</title>
<style>
:root{{--bg:#eef3f7;--ink:#1c2834;--ink2:#5a6e80;--card:#fff;--rule:#d6e0e8}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
header{{padding:28px 32px 8px}} h1{{margin:0;font-size:24px}} header p{{color:var(--ink2);margin:6px 0 0}}
main{{padding:8px 32px 40px;display:grid;gap:22px}}
section{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:16px 18px}}
h2{{margin:0 0 4px;font-size:18px}} section p{{margin:0 0 12px;color:var(--ink2);max-width:80ch}}
img{{width:100%;height:auto;border-radius:6px;display:block}}
</style></head><body><header><h1>RoboBoat course v2: checkpoint preview</h1>
<p>Rendered headlessly from the procedural generator ({time.strftime('%Y-%m-%d %H:%M')}). Click an image for full size.</p></header>
<main>{''.join(cards)}</main></body></html>"""
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print(os.path.abspath(os.path.join(OUT, "index.html")))


if __name__ == "__main__":
    main()
