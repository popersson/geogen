"""Shape descriptors: feature sizes, size ratios, angle statistics, symmetry.

Distances involving arcs are measured from arc sample points (never from
their chords), so a clearance is never under-estimated by the sampling; with
the default 7.5-degree step the over-estimate is below 0.002 * radius.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Geometry, Loop, _EPS, arc_radius


def point_segment_distances(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distances from points P (m,2) to segments A->B (s,2): result (m,s)."""
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    L2 = np.where(L2 < 1e-30, 1e-30, L2)
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip((AP * AB[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    D = AP - t[..., None] * AB[None]
    return np.sqrt((D ** 2).sum(axis=2))


def loop_items(lp: Loop, max_step_deg: float = 7.5) -> list:
    """Per edge: (sample points, straight segment or None, bbox, circle or None).

    Arcs and splines are represented by sample points only (never by chords), so
    distances are never under-estimated.  ``circle`` is (cx, cy, r) for arcs, used
    to recognise arcs of one and the same circle (a 4-arc hole), whose mutual
    distance is not a feature size."""
    items = []
    for i, e in enumerate(lp.evaluators()):
        pts = e.polyline(max_step_deg)
        segs = pts[None, :, :] if e.kind == "line" else None
        circ = None
        if e.kind == "arc":
            circ = (round(float(e.center[0]), 6), round(float(e.center[1]), 6), round(float(e.radius), 6))
        b = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        items.append((pts, segs, b, circ))
    return items


def _bbox_gap(b1, b2) -> float:
    dx = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    dy = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return math.hypot(dx, dy)


def item_distance(a, b) -> float:
    Pa, Sa = a[0], a[1]
    Pb, Sb = b[0], b[1]
    d = float(np.sqrt(((Pa[:, None, :] - Pb[None, :, :]) ** 2).sum(axis=2)).min())
    if Sb is not None:
        d = min(d, float(point_segment_distances(Pa, Sb[:, 0], Sb[:, 1]).min()))
    if Sa is not None:
        d = min(d, float(point_segment_distances(Pb, Sa[:, 0], Sa[:, 1]).min()))
    return d


def loops_min_distance(lpA: Loop, lpB: Loop, max_step_deg: float = 7.5, stop_below: float | None = None) -> float:
    """Minimum distance between two loops (exact for straight edges, sampled for arcs)."""
    IA, IB = loop_items(lpA, max_step_deg), loop_items(lpB, max_step_deg)
    best = math.inf
    for a in IA:
        for b in IB:
            if _bbox_gap(a[2], b[2]) >= best:
                continue
            best = min(best, item_distance(a, b))
            if stop_below is not None and best < stop_below:
                return best
    return best


def min_ligament(geom: Geometry, max_step_deg: float = 7.5) -> float:
    """Smallest distance between two non-adjacent edges of the same loop
    (index distance >= 2, arcs of one circle excluded) or between edges of
    different loops.  Captures thin ligaments, narrow notches, slot widths
    and hole clearances (empty gaps count as features too)."""
    items = []
    for li, lp in enumerate(geom.loops):
        for ei, it in enumerate(loop_items(lp, max_step_deg)):
            items.append((li, ei, lp.n, it))
    best = math.inf
    for a in range(len(items)):
        la, ea, na, ia = items[a]
        for b in range(a + 1, len(items)):
            lb, eb, _, ib = items[b]
            if la == lb:
                d = abs(ea - eb)
                if d == 1 or d == na - 1:
                    continue
                if ia[3] is not None and ia[3] == ib[3]:     # arcs of the same circle
                    continue
            if _bbox_gap(ia[2], ib[2]) >= best:
                continue
            best = min(best, item_distance(ia, ib))
    return best


def feature_size(geom: Geometry, max_step_deg: float = 7.5) -> float:
    """min(shortest straight edge, smallest arc radius, thinnest ligament)."""
    straight, radii = [], []
    for lp in geom.loops:
        for e in lp.evaluators():
            if e.kind == "line":
                straight.append(e.length)
            elif e.kind == "arc":
                radii.append(e.radius)
    vals = [min_ligament(geom, max_step_deg)]
    if straight:
        vals.append(min(straight))
    if radii:
        vals.append(min(radii))
    return float(min(vals))


def _mirror_symmetric(P: np.ndarray, Q: np.ndarray, tol: float) -> bool:
    for i in range(0, len(Q), 256):
        d = np.sqrt(((Q[i:i + 256, None, :] - P[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        if d.max() > tol:
            return False
    return True


def symmetry_flags(geom: Geometry, max_step_deg: float = 10.0, rel_tol: float = 1e-6) -> tuple[bool, bool]:
    """Mirror symmetry about the vertical and the horizontal axis through the bbox centre."""
    x0, y0, x1, y1 = geom.bbox()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    P = np.vstack(geom.discretize(max_step_deg))
    tol = rel_tol * max(x1 - x0, y1 - y0)
    Px = P.copy()
    Px[:, 0] = 2 * cx - Px[:, 0]
    Py = P.copy()
    Py[:, 1] = 2 * cy - Py[:, 1]
    return _mirror_symmetric(P, Px, tol), _mirror_symmetric(P, Py, tol)


def compute(geom: Geometry, max_step_deg: float = 7.5) -> dict:
    x0, y0, x1, y1 = geom.bbox()
    w, h = x1 - x0, y1 - y0
    straight, radii, curved_lens, arc_abs, spline_turns = [], [], [], [], []
    for lp in geom.loops:
        for e in lp.evaluators():
            if e.kind == "line":
                straight.append(e.length)
            elif e.kind == "arc":
                radii.append(e.radius)
                curved_lens.append(e.length)
                arc_abs.append(abs(e.theta))
            else:
                curved_lens.append(e.length)
                spline_turns.append(e.turning_deg())
    lig = min_ligament(geom, max_step_deg)
    min_edge = min(straight) if straight else math.inf
    min_radius = min(radii) if radii else math.inf
    f_min = min(min_edge, min_radius, lig)
    max_edge = max(straight) if straight else max(curved_lens)

    angle_hist: dict = {}
    n_concave = n_acute = n_smooth = 0
    for lp in geom.loops:
        for a in lp.interior_angles():
            q = round(a / 45.0) * 45
            key = str(int(q)) if abs(a - q) < 1e-6 else "other"
            angle_hist[key] = angle_hist.get(key, 0) + 1
            if a > 180 + 1e-6:
                n_concave += 1
            if a < 90 - 1e-6:
                n_acute += 1
            if abs(a - 180) < 1e-6:
                n_smooth += 1
    arc_hist: dict = {}
    for a in arc_abs:
        q = round(a / 15.0) * 15
        key = str(int(q)) if abs(a - q) < 1e-6 else "other"
        arc_hist[key] = arc_hist.get(key, 0) + 1
    sym_x, sym_y = symmetry_flags(geom)
    n_vertices = sum(lp.n for lp in geom.loops)
    area = geom.area()
    return {
        "bbox": [x0, y0, x1, y1], "width": w, "height": h, "bbox_max": max(w, h),
        "area": area, "area_fraction": area / (w * h) if w * h > 0 else 0.0,
        "n_loops": len(geom.loops), "n_holes": len(geom.loops) - 1,
        "n_vertices": n_vertices, "n_corners": n_vertices - n_smooth, "n_smooth": n_smooth,
        "n_arcs": len(radii), "n_straight": len(straight), "n_splines": len(spline_turns),
        "max_spline_turn": max(spline_turns) if spline_turns else None,
        "n_concave": n_concave, "n_acute": n_acute,
        "min_edge": min_edge if straight else None, "max_edge": max_edge,
        "min_arc_radius": min_radius if radii else None, "min_ligament": lig,
        "f_min": f_min,
        "ratio_edge": max_edge / f_min, "ratio_bbox": max(w, h) / f_min,
        "interior_angles": dict(sorted(angle_hist.items())), "arc_angles": dict(sorted(arc_hist.items())),
        "sym_x": bool(sym_x), "sym_y": bool(sym_y),
        "n_chains": len(geom.chains), "n_regions": len(geom.regions()),
    }
