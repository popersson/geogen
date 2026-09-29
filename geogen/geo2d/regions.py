"""Internal boundaries: chains that split a domain into regions.

The loops and chains of a Geometry form a planar graph.  Tracing its faces
(walk along every directed edge, at each vertex take the first edge clockwise
from the way you came, so the face stays on the left) gives the regions and
the infinite/hole faces; ``label_regions`` writes the result into the per-edge
region of every loop and the ``(left, right)`` regions of every chain, and
``region_geometry`` returns one region as an ordinary Geometry.  A chain with
the same region on both sides is a single cut through a ring: that region's
outer loop runs along the cut twice.

``cut`` inserts a straight or circular chain between two points (splitting the
edges its ends land on) and ``cut_line`` cuts along a whole lattice line.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Chain, Geometry, Loop
from .validate import _seg_hits, point_in_polygon


def _tol(geom: Geometry, tol):
    if tol is not None:
        return float(tol)
    x0, y0, x1, y1 = geom.bbox()
    return 1e-9 * max(x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- planar graph

class _Graph:
    def __init__(self, geom: Geometry, tol: float):
        self.geom, self.tol = geom, tol
        self.edges = []
        for li, lp in enumerate(geom.loops):
            for ei, e in enumerate(lp.evaluators()):
                self.edges.append({"kind": "loop", "idx": li, "ei": ei, "ev": e, "left": lp.edge_region(ei), "right": -1})
        for ci, ch in enumerate(geom.chains):
            for ei, e in enumerate(ch.evaluators()):
                self.edges.append({"kind": "chain", "idx": ci, "ei": ei, "ev": e, "left": ch.regions[0], "right": ch.regions[1]})
        self.vpos: list = []
        self.tail = np.zeros(2 * len(self.edges), int)
        self.head = np.zeros(2 * len(self.edges), int)
        self.out_dir = np.zeros((2 * len(self.edges), 2))
        self.in_dir = np.zeros((2 * len(self.edges), 2))
        for k, ed in enumerate(self.edges):
            e = ed["ev"]
            a, b = self.vertex(e.p0), self.vertex(e.p1)
            t0, t1 = e.tangent(0.0), e.tangent(1.0)
            self.tail[2 * k], self.head[2 * k], self.out_dir[2 * k], self.in_dir[2 * k] = a, b, t0, t1
            self.tail[2 * k + 1], self.head[2 * k + 1], self.out_dir[2 * k + 1], self.in_dir[2 * k + 1] = b, a, -t1, -t0
        self.outgoing = [[] for _ in self.vpos]
        for h in range(2 * len(self.edges)):
            self.outgoing[self.tail[h]].append(h)
        for v in range(len(self.vpos)):
            self.outgoing[v].sort(key=lambda h: math.atan2(self.out_dir[h][1], self.out_dir[h][0]) % (2 * math.pi))

    def vertex(self, p) -> int:
        p = np.asarray(p, float)
        for i, q in enumerate(self.vpos):
            if abs(q[0] - p[0]) <= self.tol and abs(q[1] - p[1]) <= self.tol:
                return i
        self.vpos.append(p.copy())
        return len(self.vpos) - 1

    def left_label(self, h: int) -> int:
        ed = self.edges[h // 2]
        return ed["left"] if h % 2 == 0 else ed["right"]

    def next(self, h: int) -> int:
        v = self.head[h]
        back = -self.in_dir[h]
        ang_back = math.atan2(back[1], back[0]) % (2 * math.pi)
        twin = h ^ 1
        cands = [g for g in self.outgoing[v] if g != twin]
        if not cands:
            return twin
        angs = [math.atan2(self.out_dir[g][1], self.out_dir[g][0]) % (2 * math.pi) for g in cands]
        below = [(a, g) for a, g in zip(angs, cands) if a < ang_back - 1e-12]
        if below:                                     # first edge clockwise from the way back
            return max(below)[1]
        return max(zip(angs, cands))[1]               # wrap around

    def cycles(self) -> list:
        seen = np.zeros(2 * len(self.edges), bool)
        out = []
        for h0 in range(2 * len(self.edges)):
            if seen[h0]:
                continue
            cyc, h = [], h0
            while not seen[h]:
                seen[h] = True
                cyc.append(h)
                h = self.next(h)
            poly = np.vstack([self.polyline(h)[:-1] for h in cyc])
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            out.append({"hes": cyc, "area": area, "poly": poly})
        return out

    def polyline(self, h: int, step: float = 10.0) -> np.ndarray:
        q = self.edges[h // 2]["ev"].polyline(step)
        return q if h % 2 == 0 else q[::-1]

    def interior_point(self, h: int, side: float = 1.0) -> np.ndarray:
        """Point a little to the left (side=+1) or right of the middle of half-edge h."""
        ed = self.edges[h // 2]["ev"]
        m = ed.point(0.5)
        t = ed.tangent(0.5) * (1.0 if h % 2 == 0 else -1.0)
        eps = 1e-6 * max(1.0, ed.length)
        return m + side * eps * np.array([-t[1], t[0]])


def _faces(geom: Geometry, tol):
    """Material faces of the planar graph: [{'outer': cycle, 'holes': [cycles], 'point': p}], plus
    the list of problems found (chains with a non-material side)."""
    G = _Graph(geom, tol)
    cycles = G.cycles()
    area_tol = 1e-12 * max(abs(c["area"]) for c in cycles) if cycles else 0.0
    outer_poly = geom.outer.discretize(10.0)
    hole_polys = [h.discretize(10.0) for h in geom.holes]

    def in_material(p):
        return point_in_polygon(p, outer_poly) and not any(point_in_polygon(p, hp) for hp in hole_polys)

    faces, problems = [], []
    for c in cycles:
        if c["area"] <= area_tol:
            continue
        kinds = {(G.edges[h // 2]["kind"], h % 2) for h in c["hes"]}
        if ("loop", 0) in kinds and ("loop", 1) in kinds:
            problems.append("a face has material on one side and the outside on another: a chain crosses the boundary")
        material = ("loop", 0) in kinds or (("loop", 1) not in kinds and in_material(G.interior_point(c["hes"][0])))
        if material:
            faces.append({"outer": c, "holes": [], "hes": c["hes"]})
    # inner boundaries (negative cycles: holes; zero-area cycles: dangling chains) belong to the
    # smallest face containing a point just left of them
    for c in cycles:
        if c["area"] > area_tol:
            continue
        p = G.interior_point(c["hes"][0], +1.0)
        best = None
        for f in faces:
            if point_in_polygon(p, f["outer"]["poly"]) and (best is None or f["outer"]["area"] < best["outer"]["area"]):
                best = f
        if best is not None:
            best["holes"].append(c)
    order = sorted(range(len(faces)), key=lambda i: (round(faces[i]["outer"]["poly"][:, 0].min(), 9),
                                                   round(faces[i]["outer"]["poly"][:, 1].min(), 9)))
    return G, [faces[i] for i in order], problems


def label_regions(geom: Geometry, tol: float | None = None) -> int:
    """Trace the faces and write region ids (0 .. R-1, ordered by position) into the loops'
    per-edge ``region`` and the chains' ``regions``.  Returns R.  Raises ValueError when a chain
    has empty space (or a hole) on one of its sides."""
    G, faces, problems = _faces(geom, _tol(geom, tol))
    side = {}                                            # half-edge -> region id
    for r, f in enumerate(faces):
        for h in f["hes"]:
            side[h] = r
        for c in f["holes"]:
            for h in c["hes"]:
                side[h] = r
    for lp in geom.loops:
        lp.region = None
    regions_per_loop = {li: np.zeros(lp.n, int) for li, lp in enumerate(geom.loops)}
    for k, ed in enumerate(G.edges):
        if ed["kind"] == "loop":
            regions_per_loop[ed["idx"]][ed["ei"]] = side.get(2 * k, -1)
        else:
            ch = geom.chains[ed["idx"]]
            left, right = side.get(2 * k, -1), side.get(2 * k + 1, -1)
            if left < 0 or right < 0:
                problems.append(f"chain {ed['idx']} has no material on one side")
            ch.regions = (left, right)
    for li, lp in enumerate(geom.loops):
        if (regions_per_loop[li] < 0).any():
            problems.append(f"loop {li}: an edge has no material on its left")
        lp.region = regions_per_loop[li] if regions_per_loop[li].any() else None
    if problems:
        raise ValueError("; ".join(sorted(set(problems))))
    return len(faces)


def _halfedge_geometry(G: _Graph, h: int):
    """(start point, angle, tag, interior, kind) of half-edge h as traversed."""
    ed = G.edges[h // 2]
    src = G.geom.loops[ed["idx"]] if ed["kind"] == "loop" else G.geom.chains[ed["idx"]]
    ei, fwd = ed["ei"], h % 2 == 0
    x = src.interior[ei] if src.interior is not None else None
    tag = src.tags[ei] if src.tags is not None else ("outer" if ed["kind"] == "loop" and ed["idx"] == 0 else "hole" if ed["kind"] == "loop" else "cut")
    return (G.vpos[G.tail[h]], float(src.angles[ei]) * (1.0 if fwd else -1.0), tag,
            None if x is None else (x if fwd else x[::-1].copy()), src.kind(ei))


def _loops_from_halfedges(G: _Graph, hes: list, region: int) -> tuple:
    """Split a traced cycle into simple closed loops and the doubled edges (traversed in both
    directions: self-adjacent cuts and dangling chains), returned as Loops and Chains."""
    doubled = {h for h in hes if (h ^ 1) in hes}
    chains = []
    for h in sorted(doubled):
        if h % 2 == 0:
            p, ang, tag, inter, kind = _halfedge_geometry(G, h)
            q = G.vpos[G.head[h]]
            chains.append(Chain(np.array([p, q]), np.array([ang]), [tag], None if inter is None else [inter], None, (region, region)))
    rest = [h for h in hes if h not in doubled]
    loops = []
    while rest:
        run = [rest.pop(0)]
        while True:
            nxt = [h for h in rest if G.tail[h] == G.head[run[-1]]]
            if not nxt:
                break
            run.append(nxt[0])
            rest.remove(nxt[0])
        if G.head[run[-1]] != G.tail[run[0]] or len(run) < 3:
            raise ValueError("a region boundary does not close into loops of at least three edges")
        pts, angs, tags, interior, smooth = [], [], [], [], []
        for j, h in enumerate(run):
            p, ang, tag, inter, kind = _halfedge_geometry(G, h)
            pts.append(p)
            angs.append(ang)
            tags.append(tag)
            interior.append(inter)
            prev = run[j - 1]
            prev_kind = _halfedge_geometry(G, prev)[4]
            smooth.append(bool(kind == "spline" and prev_kind == "spline" and float(G.in_dir[prev] @ G.out_dir[h]) > 1 - 1e-9))
        loops.append(Loop(np.array(pts), np.array(angs), tags, interior if any(x is not None for x in interior) else None,
                          smooth if any(smooth) else None, np.full(len(pts), region)))
    return loops, chains


def region_geometry(geom: Geometry, r: int, tol: float | None = None) -> Geometry:
    """Region r as a standalone Geometry: its outer loop, its holes, and as internal chains the
    cuts that have region r on both sides (a single cut through a ring) and dangling chains.
    Region ids inside the result are renumbered to 0."""
    G, faces, _ = _faces(geom, _tol(geom, tol))
    if r < 0 or r >= len(faces):
        raise IndexError(f"region {r} does not exist ({len(faces)} regions)")
    f = faces[r]
    outer, chains = _loops_from_halfedges(G, f["outer"]["hes"], 0)
    holes = []
    for c in f["holes"]:
        lps, chs = _loops_from_halfedges(G, c["hes"], 0)
        holes += lps
        chains += chs
    outer.sort(key=lambda lp: -lp.signed_area())
    loops = outer + holes
    if loops[0].signed_area() <= 0:
        raise ValueError("region has no counter-clockwise outer loop")
    return Geometry(loops, {"region": r, "parent": geom.meta.get("seed")}, chains)


def n_regions(geom: Geometry, tol: float | None = None) -> int:
    return len(_faces(geom, _tol(geom, tol))[1])


# --------------------------------------------------------------------------- validation

def _all_segments(geom: Geometry, step: float = 22.5):
    segs = []                                            # (kind, idx, ei, a, b)
    for li, lp in enumerate(geom.loops):
        for ei, q in enumerate(lp.discretize_edges(step)):
            for a, b in zip(q[:-1], q[1:]):
                segs.append(("loop", li, ei, a, b))
    for ci, ch in enumerate(geom.chains):
        for ei, q in enumerate(ch.discretize_edges(step)):
            for a, b in zip(q[:-1], q[1:]):
                segs.append(("chain", ci, ei, a, b))
    return segs


def _is_vertex(geom: Geometry, p, tol, exclude_chain: int | None = None) -> bool:
    """Is p (within tol) a vertex of a loop or of a chain other than ``exclude_chain``?"""
    p = np.asarray(p, float)
    for lp in geom.loops:
        if (np.abs(lp.points - p).max(axis=1) <= tol).any():
            return True
    for ci, ch in enumerate(geom.chains):
        if ci == exclude_chain:
            continue
        if (np.abs(ch.points - p).max(axis=1) <= tol).any():
            return True
    return False


def validate_regions(geom: Geometry, tol: float | None = None) -> list[str]:
    """Conformity (chain ends at vertices or inside the material), no crossing or touching between
    chains and anything else, and region labels consistent with the traced faces."""
    tol = _tol(geom, tol)
    problems: list[str] = []
    outer_poly = geom.outer.discretize(10.0)
    hole_polys = [h.discretize(10.0) for h in geom.holes]
    from .snap import locate
    for ci, ch in enumerate(geom.chains):
        for p in (ch.points[0], ch.points[-1]):
            if _is_vertex(geom, p, tol, exclude_chain=ci):
                continue
            hits = [(li, ei, t) for li, ei, t in locate(geom, p, tol) if tol < t < 1 - tol and li != -ci - 1]
            if hits:
                problems.append(f"chain {ci}: an end lies in the middle of an edge; split that edge first")
            elif not (point_in_polygon(p, outer_poly) and not any(point_in_polygon(p, hp) for hp in hole_polys)):
                problems.append(f"chain {ci}: an end lies outside the material")
    segs = _all_segments(geom)
    chain_ids = [k for k, s in enumerate(segs) if s[0] == "chain"]
    A = np.array([s[3] for s in segs])
    B = np.array([s[4] for s in segs])
    for k in chain_ids:
        kind, ci, ei, a, b = segs[k]
        hit = _seg_hits(a, b, A, B, 1e-9)
        for j in np.nonzero(hit)[0]:
            if j == k:
                continue
            # touching at a shared vertex of the graph is fine
            other = segs[j]
            shared = any(np.abs(np.asarray(x) - np.asarray(y)).max() <= tol
                         for x in (a, b) for y in (other[3], other[4]))
            if shared and _is_vertex(geom, a if any(np.abs(np.asarray(a) - np.asarray(y)).max() <= tol for y in (other[3], other[4])) else b, tol):
                continue
            if other[0] == kind and other[1] == ci and abs(other[2] - ei) <= 1:
                continue                                   # consecutive pieces of the same chain
            problems.append(f"chain {ci}: crosses or touches another edge")
            break
    if problems:
        return problems
    try:
        G, faces, probs = _faces(geom, tol)
        problems += probs
        side = {}
        for r, f in enumerate(faces):
            for h in f["hes"] + [h for c in f["holes"] for h in c["hes"]]:
                side[h] = r
        for k, ed in enumerate(G.edges):
            if ed["kind"] == "loop":
                if geom.loops[ed["idx"]].edge_region(ed["ei"]) != side.get(2 * k, -1):
                    problems.append("loop edge regions do not match the traced faces; call label_regions()")
                    break
            else:
                if geom.chains[ed["idx"]].regions != (side.get(2 * k, -1), side.get(2 * k + 1, -1)):
                    problems.append("chain regions do not match the traced faces; call label_regions()")
                    break
    except ValueError as exc:
        problems.append(str(exc))
    return sorted(set(problems))


# --------------------------------------------------------------------------- cutting

def _attach(geom: Geometry, p, tol):
    """Make p a vertex of the graph: split the loop or chain edge it lies on (lines and arcs).
    Returns (geometry, exact point)."""
    from .snap import locate
    p = np.asarray(p, float)
    if _is_vertex(geom, p, tol):
        return geom, p
    hits = locate(geom, p, tol)
    for li, ei, t in hits:
        if tol < t < 1 - tol:
            if li >= 0:
                lp = geom.loops[li]
                q = lp.edge(ei).point(t)
                loops = list(geom.loops)
                loops[li] = lp.split_edge(ei, t)
                return Geometry(loops, dict(geom.meta), list(geom.chains)), q
            ch = geom.chains[-li - 1]
            q = ch.edge(ei).point(t)
            chains = list(geom.chains)
            chains[-li - 1] = ch.split_edge(ei, t)
            return Geometry(list(geom.loops), dict(geom.meta), chains), q
    outer_poly = geom.outer.discretize(10.0)
    if not point_in_polygon(p, outer_poly) or any(point_in_polygon(p, h.discretize(10.0)) for h in geom.holes):
        raise ValueError(f"cut end {p.tolist()} is outside the material")
    return geom, p                                       # a dangling end inside the material


def _copy(geom: Geometry) -> Geometry:
    """Independent copy (labelling writes into loops and chains in place)."""
    loops = [Loop(lp.points.copy(), lp.angles.copy(), None if lp.tags is None else list(lp.tags),
                  None if lp.interior is None else [None if x is None else x.copy() for x in lp.interior],
                  None if lp.smooth is None else list(lp.smooth), None if lp.region is None else lp.region.copy())
             for lp in geom.loops]
    chains = [Chain(ch.points.copy(), ch.angles.copy(), None if ch.tags is None else list(ch.tags),
                    None if ch.interior is None else [None if x is None else x.copy() for x in ch.interior],
                    None if ch.smooth is None else list(ch.smooth), ch.regions) for ch in geom.chains]
    return Geometry(loops, dict(geom.meta), chains)


def cut(geom: Geometry, p, q, angle: float = 0.0, tol: float | None = None) -> Geometry:
    """New geometry with an internal chain from p to q (straight, or a circular arc with the given
    included angle).  The ends are attached to the boundary or to existing chains, splitting the
    edges they land on; regions are re-labelled.  The input geometry is left untouched."""
    tol = _tol(geom, tol)
    g = _copy(geom)
    g, p = _attach(g, p, tol)
    g, q = _attach(g, q, tol)
    chain = Chain(np.array([p, q]), np.array([angle]), ["cut"])
    g.chains.append(chain)
    problems = validate_regions(g, tol)
    problems = [pr for pr in problems if "label_regions" not in pr]
    if problems:
        raise ValueError("; ".join(problems))
    label_regions(g, tol)
    return g


def _line_hits(geom: Geometry, axis: int, value: float, tol: float):
    """Exact intersections of the lattice line {x[axis] = value} with all loop and chain edges."""
    other = 1 - axis
    pts = []
    for src in list(geom.loops) + list(geom.chains):
        for e in src.evaluators():
            if e.kind == "line":
                a, b = e.p0[axis], e.p1[axis]
                if abs(b - a) < 1e-15:
                    continue
                t = (value - a) / (b - a)
                if -1e-12 <= t <= 1 + 1e-12:
                    pts.append(e.p0 + min(max(t, 0.0), 1.0) * (e.p1 - e.p0))
            elif e.kind == "arc":
                d = value - e.center[axis]
                if abs(d) <= e.radius + tol:
                    h = math.sqrt(max(e.radius ** 2 - d ** 2, 0.0))
                    for sgn in (1, -1):
                        c = np.zeros(2)
                        c[axis] = value
                        c[other] = e.center[other] + sgn * h
                        t, dist = e.locate(c)
                        if dist <= 1e-9 * max(1.0, e.radius):
                            pts.append(e.point(t))
            else:                                            # spline: polyline crossings, refined onto the curve
                q = e.polyline(5.0)
                for a, b in zip(q[:-1], q[1:]):
                    if (a[axis] - value) * (b[axis] - value) <= 0 and a[axis] != b[axis]:
                        s = (value - a[axis]) / (b[axis] - a[axis])
                        c = a + s * (b - a)
                        t, _ = e.locate(c)
                        pts.append(e.point(t))
    uniq = []
    for p in sorted(pts, key=lambda p: p[other]):
        if not uniq or abs(uniq[-1][other] - p[other]) > tol:
            uniq.append(p)
    return uniq


def cut_line(geom: Geometry, axis: int, value: float, tol: float | None = None) -> Geometry:
    """Cut along the whole line x[axis] = value: one chain per material segment of the line."""
    tol = _tol(geom, tol)
    g = geom
    hits = _line_hits(g, axis, value, tol)
    outer_poly = g.outer.discretize(10.0)
    hole_polys = [h.discretize(10.0) for h in g.holes]
    for a, b in zip(hits[:-1], hits[1:]):
        m = 0.5 * (a + b)
        if point_in_polygon(m, outer_poly) and not any(point_in_polygon(m, hp) for hp in hole_polys):
            g = cut(g, a, b, 0.0, tol)
    return g
