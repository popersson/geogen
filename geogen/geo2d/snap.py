"""Placing mesh nodes on curved boundary edges, coordinates only.

``split(geom, A, B, s)`` returns the point at arc-length fraction ``s`` between
two boundary nodes A and B that lie on the same edge; ``snap_chain`` moves the
interior nodes of a boundary chain onto the edge so that chord-length fractions
become arc-length fractions; ``snap_mesh`` does that for every boundary chain of
a mesh whose nodes include all control points.  All of them use the edge
evaluators of curves.py, so straight edges, arcs and splines behave alike.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .geometry import Geometry


def _tol(geom: Geometry, tol: float | None) -> float:
    if tol is not None:
        return float(tol)
    x0, y0, x1, y1 = geom.bbox()
    return 1e-9 * max(x1 - x0, y1 - y0)


def _sources(geom: Geometry):
    """(index, object) for loops (0, 1, ...) and chains (-1, -2, ...)."""
    return [(li, lp) for li, lp in enumerate(geom.loops)] + [(-ci - 1, ch) for ci, ch in enumerate(geom.chains)]


def locate(geom: Geometry, p, tol: float | None = None) -> list[tuple[int, int, float]]:
    """All (loop, edge, t) on which the point lies (two entries at a control point); a negative
    loop index -k-1 means chain k."""
    tol = _tol(geom, tol)
    p = np.asarray(p, float)
    out = []
    for li, lp in _sources(geom):
        for ei, e in enumerate(lp.evaluators()):
            x0, y0, x1, y1 = e.bbox()
            if p[0] < x0 - tol or p[0] > x1 + tol or p[1] < y0 - tol or p[1] > y1 + tol:
                continue
            t, d = e.locate(p)
            if d <= tol:
                out.append((li, ei, t))
    return out


def project(geom: Geometry, p) -> tuple[int, int, float, np.ndarray]:
    """Nearest boundary point: (loop, edge, t, point)."""
    p = np.asarray(p, float)
    best = None
    for li, lp in _sources(geom):
        for ei, e in enumerate(lp.evaluators()):
            t, d = e.locate(p)
            if best is None or d < best[0]:
                best = (d, li, ei, t)
    _, li, ei, t = best
    return li, ei, t, geom.point(li, ei, t)


def _common_edge(geom: Geometry, A, B, tol: float | None):
    la = locate(geom, A, tol)
    lb = locate(geom, B, tol)
    common = [(li, ei, ta, tb) for li, ei, ta in la for lj, ej, tb in lb if (li, ei) == (lj, ej)]
    if not common:
        raise ValueError(f"points {np.asarray(A).tolist()} and {np.asarray(B).tolist()} do not lie on a common edge "
                         f"(A on {[(l, e) for l, e, _ in la]}, B on {[(l, e) for l, e, _ in lb]})")
    if len(common) > 1:                      # both are control points of a 2-edge chain: take the shorter edge
        common.sort(key=lambda c: geom.edge(c[0], c[1]).length)
    return common[0]


def split(geom: Geometry, A, B, s: float = 0.5, tol: float | None = None) -> np.ndarray:
    """Point at arc-length fraction ``s`` (from A towards B) of the edge containing A and B.
    Straight edges return the linear interpolant, so the mesher needs one code path."""
    li, ei, ta, tb = _common_edge(geom, A, B, tol)
    return geom.point(li, ei, ta + s * (tb - ta))


def _chain_params(pts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Edge parameters for a chain: chord-length fractions along ``pts`` mapped onto [t0, t1]."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    frac = cum / cum[-1] if cum[-1] > 0 else np.linspace(0, 1, len(pts))
    return t0 + frac * (t1 - t0)


def _chain_on_edge(edge, pts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Place the interior points of ``pts`` on ``edge`` between parameters t0 and t1 so that
    their arc-length fractions equal their chord-length fractions along ``pts``."""
    out = edge.point(_chain_params(pts, t0, t1))
    out[0], out[-1] = pts[0], pts[-1]
    return out


def snap_chain(geom: Geometry, pts, tol: float | None = None) -> np.ndarray:
    """``pts`` is an ordered boundary chain whose first and last points lie on one edge;
    returns the chain with its interior points moved onto that edge."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts.copy()
    li, ei, ta, tb = _common_edge(geom, pts[0], pts[-1], tol)
    return _chain_on_edge(geom.edge(li, ei), pts, ta, tb)


def boundary_edges(elements) -> list[tuple[int, int]]:
    """Edges that belong to exactly one element (element = sequence of node indices)."""
    cnt: Counter = Counter()
    for el in elements:
        el = [int(v) for v in el]
        for k in range(len(el)):
            a, b = el[k], el[(k + 1) % len(el)]
            cnt[(min(a, b), max(a, b))] += 1
    return [e for e, c in cnt.items() if c == 1]


def snap_mesh(geom: Geometry, nodes, elements=None, tol: float | None = None):
    """Move every boundary node of a mesh onto the true boundary.

    Accepts either a :class:`geo2d.mesh.Mesh` (returns a new Mesh whose ``data`` holds
    the boundary link ``bnd_loop``, ``bnd_edge``, ``bnd_t`` (-1 / -1 / nan for interior
    nodes) and whose ``meta["chains"]`` lists the boundary chains) or ``nodes`` (N, 2)
    plus ``elements`` (sequences of node indices; returns ``(new_nodes, chains)``).
    The mesh must contain every control point as a node (matched within ``tol``).
    ``chains`` = ``[(loop, edge, [node ids from start to end]), ...]``.
    """
    from .mesh import Mesh
    mesh = nodes if isinstance(nodes, Mesh) else None
    if mesh is not None:
        nodes, elements = mesh.nodes, mesh.elements()
    nodes = np.array(nodes, float)
    tol = _tol(geom, tol)
    adj = defaultdict(list)
    for a, b in boundary_edges(elements):
        adj[a].append(b)
        adj[b].append(a)
    ctrl: dict = {}
    for li, lp in enumerate(geom.loops):
        for vi, p in enumerate(lp.points):
            d = np.hypot(nodes[:, 0] - p[0], nodes[:, 1] - p[1])
            k = int(np.argmin(d))
            if d[k] > tol:
                raise ValueError(f"control point {p.tolist()} of loop {li} is not a mesh node")
            ctrl[(li, vi)] = k
    ctrl_nodes = set(ctrl.values())
    bnd_loop = np.full(len(nodes), -1, int)
    bnd_edge = np.full(len(nodes), -1, int)
    bnd_t = np.full(len(nodes), np.nan)
    chains = []
    for li, lp in enumerate(geom.loops):
        for ei in range(lp.n):
            a, b = ctrl[(li, ei)], ctrl[(li, (ei + 1) % lp.n)]
            chain = None
            for nb in adj[a]:
                path, prev, cur = [a, nb], a, nb
                while cur not in ctrl_nodes and len(path) <= len(nodes):
                    nxt = [x for x in adj[cur] if x != prev]
                    if len(nxt) != 1:
                        break
                    prev, cur = cur, nxt[0]
                    path.append(cur)
                if cur == b:
                    chain = path
                    break
            if chain is None:
                raise ValueError(f"no boundary chain from control point {ei} to {(ei + 1) % lp.n} of loop {li}")
            ts = _chain_params(nodes[chain], 0.0, 1.0)
            if len(chain) > 2:
                new = lp.edge(ei).point(ts)
                nodes[chain[1:-1]] = new[1:-1]
            bnd_loop[chain[:-1]], bnd_edge[chain[:-1]], bnd_t[chain[:-1]] = li, ei, ts[:-1]
            chains.append((li, ei, [int(v) for v in chain]))
    if mesh is None:
        return nodes, chains
    out = mesh.copy()
    out.nodes = nodes
    out.data.update({"bnd_loop": bnd_loop, "bnd_edge": bnd_edge, "bnd_t": bnd_t})
    out.meta["chains"] = [[li, ei, ids] for li, ei, ids in chains]
    return out
