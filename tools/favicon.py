# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate static/favicon.svg: a mountain ridge as a graph inside a hex sticker.

Mesh = Delaunay triangulation of ridge, base and interior points, clipped to the
mountain outline. The red route is the Dijkstra shortest path from the middle of
the base to the summit.

    uv run tools/favicon.py
"""

import heapq
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

INK, PAPER, BLUE, RED = "#26251f", "#faf8f3", "#2e5a87", "#b5462f"

# Ridge, left to right (64x64 grid, y down). Summit is the highest point.
RIDGE = [
    (11, 46), (14, 39), (18, 31), (22, 35), (26, 26), (29, 29),
    (34, 12),
    (38, 22), (42, 18), (45, 26), (49, 23), (52, 33), (55, 46),
]
# Foot of the mountain, left to right: zigzag so the base reads as receding foothills.
BASE = [(18, 50), (24, 46), (30, 51), (36, 47), (42, 51.5), (48, 47.5)]
SUMMIT = RIDGE[6]
START = (30, 51)


def inside(poly, p):
    x, y = p
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def main() -> None:
    outline = RIDGE + BASE[::-1]
    edges_outline = list(zip(outline, outline[1:] + outline[:1]))

    # interior points: jittered grid, kept away from the outline
    rng = np.random.default_rng(7)
    interior = []
    for gy in np.arange(16, 44, 6.0):
        for gx in np.arange(13, 56, 6.5):
            p = (gx + rng.uniform(-1.4, 1.4) + (3 if int(gy) % 12 else 0), gy + rng.uniform(-1.2, 1.2))
            if inside(outline, p) and min(seg_dist(p, a, b) for a, b in edges_outline) > 3.2:
                interior.append(p)

    pts = np.array(RIDGE + BASE + interior, dtype=float)
    tri = Delaunay(pts)
    edges = set()
    for s in tri.simplices:
        if inside(outline, tuple(pts[s].mean(axis=0))):
            for i in range(3):
                a, b = sorted((s[i], s[(i + 1) % 3]))
                edges.add((a, b))
    # drop chords that cut through the air across a notch in the ridge
    on_outline = {tuple(sorted((tuple(map(float, a)), tuple(map(float, b))))) for a, b in edges_outline}
    edges = {
        (a, b) for a, b in edges
        if tuple(sorted((tuple(pts[a]), tuple(pts[b])))) in on_outline
        or inside(outline, tuple((pts[a] + pts[b]) / 2))
    }

    idx = {tuple(map(float, p)): i for i, p in enumerate(pts)}
    start, goal = idx[tuple(map(float, START))], idx[tuple(map(float, SUMMIT))]
    adj: dict[int, list[tuple[float, int]]] = {}
    for a, b in edges:
        w = float(np.linalg.norm(pts[a] - pts[b]))
        adj.setdefault(a, []).append((w, b))
        adj.setdefault(b, []).append((w, a))
    dist, prev, pq = {start: 0.0}, {}, [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        for w, v in adj[u]:
            if d + w < dist.get(v, 1e9):
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path_edges = {tuple(sorted(e)) for e in zip(path, path[1:])}

    ridge_ids = [idx[tuple(map(float, p))] for p in RIDGE]
    ridge_edges = {tuple(sorted(e)) for e in zip(ridge_ids, ridge_ids[1:])}
    base_ids = [ridge_ids[0]] + [idx[tuple(map(float, p))] for p in BASE] + [ridge_ids[-1]]
    base_edges = {tuple(sorted(e)) for e in zip(base_ids, base_ids[1:])}

    # Fit: centre the graph's bounding box in the hex and scale it uniformly so every
    # node (incl. radius) keeps GAP units clear of the hex border's inner edge.
    # Pointy-top hex centred at (32, 32): |dx| <= a and |dx|/2 + |dy|*sqrt(3)/2 <= a.
    HEX_APOTHEM, HEX_STROKE, NODE_R, GAP = 25.5, 3.2, 2.7, 2.5
    a = HEX_APOTHEM - HEX_STROKE / 2 - GAP - NODE_R
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centred = pts - (lo + hi) / 2
    dx, dy = np.abs(centred[:, 0]), np.abs(centred[:, 1])
    scale = min(1.0, a / dx.max(), a / (dx / 2 + dy * np.sqrt(3) / 2).max())
    pts = centred * scale + 32
    summit_xy = pts[goal]

    f = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
    line = lambda a, b: f'<line x1="{f(pts[a][0])}" y1="{f(pts[a][1])}" x2="{f(pts[b][0])}" y2="{f(pts[b][1])}"/>'
    mesh = [line(a, b) for a, b in sorted(edges) if (a, b) not in ridge_edges | base_edges | path_edges]
    outline_lines = [line(a, b) for a, b in sorted(ridge_edges | base_edges)]
    route = " ".join(f"{f(pts[i][0])},{f(pts[i][1])}" for i in reversed(path))
    nodes = []
    for i, (x, y) in enumerate(pts):
        if i == goal:
            continue
        on_path, on_ridge = i in path, i in ridge_ids
        r = 1.5 if on_path or on_ridge else 1.1
        nodes.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}"{" fill=" + chr(34) + RED + chr(34) if on_path else ""}/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <!-- generated by tools/favicon.py -->
  <polygon points="32,2.8 57.5,17.5 57.5,46.5 32,61.2 6.5,46.5 6.5,17.5" fill="{PAPER}" stroke="{INK}" stroke-width="3.2" stroke-linejoin="round"/>
  <g stroke="{INK}" stroke-width="0.8" stroke-opacity="0.45" stroke-linecap="round">
    {chr(10).join("    " + m for m in mesh).strip()}
  </g>
  <g stroke="{INK}" stroke-width="1.6" stroke-linecap="round">
    {chr(10).join("    " + m for m in outline_lines).strip()}
  </g>
  <polyline points="{route}" fill="none" stroke="{RED}" stroke-width="1.9" stroke-linejoin="round" stroke-linecap="round"/>
  <g fill="{INK}">
    {chr(10).join("    " + n for n in nodes).strip()}
  </g>
  <circle cx="{f(summit_xy[0])}" cy="{f(summit_xy[1])}" r="2.7" fill="{BLUE}"/>
</svg>
"""
    out = Path(__file__).resolve().parent.parent / "static" / "favicon.svg"
    out.write_text(svg)
    print(f"{len(pts)} nodes, {len(edges)} edges, route {len(path) - 1} hops, scale {scale:.3f} -> {out}")


if __name__ == "__main__":
    main()
