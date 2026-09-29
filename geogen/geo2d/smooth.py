"""Mesh smoothing with frozen boundary nodes.

* ``laplacian``: free node -> mean of its edge neighbours (vectorised).
* ``smart_laplacian``: same target, but a move is accepted only if the minimum
  corner quality of the adjacent elements does not decrease (Freitag).
* ``optimize``: per-node simultaneous untangling and smoothing (Escobar et al.):
  minimise sum_k (|a_k|^2 + |b_k|^2) sin(theta_n) / (2 h(J_k)) over the corners
  of the adjacent elements, where h(J) = (J + sqrt(J^2 + delta^2)) / 2 regularises
  the corner Jacobian so that inverted corners are pushed out instead of
  blowing up.  Newton steps with finite-difference derivatives and backtracking.
  For an untangled quad mesh this is the usual condition-number smoothing; for
  triangles and polygons it is the corner-based generalisation (mean ratio for
  triangles may be added later).

All functions return a new Mesh; ``fixed`` is a node mask or id list
(default: the boundary nodes).
"""
from __future__ import annotations

import math

import numpy as np

from .mesh import Mesh


def _fixed_mask(mesh: Mesh, fixed) -> np.ndarray:
    if fixed is None:
        return mesh.is_boundary()
    fixed = np.asarray(fixed)
    if fixed.dtype == bool:
        return fixed.copy()
    m = np.zeros(mesh.n_nodes, bool)
    m[fixed] = True
    return m


def laplacian(mesh: Mesh, iters: int = 20, omega: float = 1.0, fixed=None, tol: float = 1e-10) -> Mesh:
    """Edge-based Laplacian smoothing of the free nodes."""
    out = mesh.copy()
    free = ~_fixed_mask(mesh, fixed)
    e = mesh.edges()
    deg = np.bincount(e.ravel(), minlength=mesh.n_nodes).astype(float)
    deg[deg == 0] = 1.0
    X = out.nodes
    scale = math.sqrt(abs(mesh.signed_areas()).sum())
    for _ in range(iters):
        s = np.zeros_like(X)
        np.add.at(s, e[:, 0], X[e[:, 1]])
        np.add.at(s, e[:, 1], X[e[:, 0]])
        move = omega * (s / deg[:, None] - X)
        move[~free] = 0.0
        X += move
        if np.abs(move).max() < tol * scale:
            break
    return out


class _NodeCorners:
    """Corner triplets (prev, node, next) and element sizes of all corners adjacent to each node."""

    def __init__(self, mesh: Mesh, tangents=None):
        c = mesh.corners()
        order = np.argsort(c["elem"], kind="stable")
        counts = np.bincount(c["elem"], minlength=mesh.n_elements)
        self._eoff = np.concatenate([[0], np.cumsum(counts)])
        self._ecorner = order
        self._tri = np.column_stack([c["prev"], c["node"], c["next"]])
        self._sin = np.sin(math.pi * (c["n"] - 2) / c["n"])
        self._noff, self._nel = mesh.node_elements()
        self._dir = _corner_directions(self._tri, tangents)

    def of_node(self, i: int):
        els = self._nel[self._noff[i]:self._noff[i + 1]]
        ids = np.concatenate([self._ecorner[self._eoff[e]:self._eoff[e + 1]] for e in els])
        dirs = None if self._dir is None else self._dir[ids]
        return self._tri[ids], self._sin[ids], dirs


def _corner_directions(tri: np.ndarray, tangents):
    """(K, 2, 2) unit directions for the (a, b) sides, NaN where the side is straight.

    `tangents` maps (node, neighbour) -> the unit direction the element's side
    leaves `node` along, for sides that are CURVED. Every other side keeps its
    chord direction and is left NaN here.
    """
    if not tangents:
        return None
    out = np.full((len(tri), 2, 2), np.nan)
    for k, (prev, node, nxt) in enumerate(tri):
        a = tangents.get((int(node), int(nxt)))
        if a is not None:
            out[k, 0] = a
        b = tangents.get((int(node), int(prev)))
        if b is not None:
            out[k, 1] = b
    return out if np.isfinite(out).any() else None


def _corner_terms(P: np.ndarray, dirs=None):
    """P (K, 3, 2) corner points (prev, node, next) -> J, |a|^2 + |b|^2.

    With `dirs`, a side that is CURVED takes its direction from the tangent
    there while keeping the chord LENGTH. On a curved edge the element's side
    follows the arc, so the direction it leaves the corner is the tangent, and
    building J from the chord measures an element nobody has. Length stays on
    the chord, so |a| and |b| still describe the straight-sided polygonal patch
    and aspect ratio is untouched -- which is why only J moves here and
    |a|^2 + |b|^2 does not.
    """
    a = P[:, 2] - P[:, 1]
    b = P[:, 0] - P[:, 1]
    L2 = (a ** 2).sum(axis=1) + (b ** 2).sum(axis=1)
    if dirs is not None:
        a = _retarget(a, dirs[:, 0])
        b = _retarget(b, dirs[:, 1])
    J = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return J, L2


def _retarget(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """`v` turned onto `direction` wherever one is given, keeping |v|."""
    use = np.isfinite(direction).all(axis=1)
    if not use.any():
        return v
    out = v.copy()
    lengths = np.linalg.norm(v[use], axis=1, keepdims=True)
    unit = direction[use]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    out[use] = lengths * unit
    return out


def _min_quality(P, sn, dirs=None) -> float:
    J, L2 = _corner_terms(P, dirs)
    return float((2 * J / L2 / sn).min())


def smart_laplacian(mesh: Mesh, iters: int = 10, fixed=None, tol: float = 1e-10,
                    tangents=None) -> Mesh:
    """Laplacian target, accepted per node only if the local minimum corner quality does not drop."""
    out = mesh.copy()
    free = np.nonzero(~_fixed_mask(mesh, fixed))[0]
    nc = _NodeCorners(mesh, tangents)
    noff, nb = mesh.node_neighbors()
    X = out.nodes
    scale = math.sqrt(abs(mesh.signed_areas()).sum())
    local = [nc.of_node(i) for i in range(mesh.n_nodes)]
    for _ in range(iters):
        maxmove = 0.0
        for i in free:
            tri, sn, dirs = local[i]
            target = X[nb[noff[i]:noff[i + 1]]].mean(axis=0)
            P = X[tri]
            q_old = _min_quality(P, sn, dirs)
            mask = tri == i
            Pn = P.copy()
            Pn[mask] = target
            if _min_quality(Pn, sn, dirs) >= q_old - 1e-12:
                maxmove = max(maxmove, float(np.hypot(*(target - X[i]))))
                X[i] = target
        if maxmove < tol * scale:
            break
    return out


def _objective(P, sn, delta: float, dirs=None) -> float:
    J, L2 = _corner_terms(P, dirs)
    h = 0.5 * (J + np.sqrt(J * J + delta * delta))
    if np.any(h <= 0):
        return math.inf
    return float((L2 * sn / (2.0 * h)).sum())


def _node_tangent_terms(K: int, dirs):
    """Per-corner tangent bookkeeping for one node, computed ONCE per node.

    Returns (use_a, unit_a, use_b, unit_b): which corners have a curved (a, b)
    side and the unit direction it leaves the corner along. These are constant
    with respect to the node being moved (see `_optimize_node`), so hoisting
    them out of the objective is exact -- the old code rebuilt them on every
    evaluation, and that rebuilding was over half the smoother's time.
    """
    use_a = np.zeros(K, bool)
    use_b = np.zeros(K, bool)
    unit_a = np.zeros((K, 2))
    unit_b = np.zeros((K, 2))
    if dirs is not None:
        use_a[:] = np.isfinite(dirs[:, 0]).all(axis=1)
        use_b[:] = np.isfinite(dirs[:, 1]).all(axis=1)
        if use_a.any():
            d = dirs[:, 0][use_a]
            unit_a[use_a] = d / np.linalg.norm(d, axis=1, keepdims=True)
        if use_b.any():
            d = dirs[:, 1][use_b]
            unit_b[use_b] = d / np.linalg.norm(d, axis=1, keepdims=True)
    return use_a, unit_a, use_b, unit_b


def _node_objective_numpy(P, sn, use_a, unit_a, use_b, unit_b) -> float:
    """The per-node objective, same arithmetic and operation order as
    `_corner_terms` + `_objective` -- bitwise identical to the old evaluation --
    with the tangent bookkeeping taken as given."""
    a = P[:, 2] - P[:, 1]
    b = P[:, 0] - P[:, 1]
    L2 = (a ** 2).sum(axis=1) + (b ** 2).sum(axis=1)
    if use_a.any():
        a = a.copy()
        a[use_a] = np.linalg.norm(a[use_a], axis=1, keepdims=True) * unit_a[use_a]
    if use_b.any():
        b = b.copy()
        b[use_b] = np.linalg.norm(b[use_b], axis=1, keepdims=True) * unit_b[use_b]
    J = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    jmin = float(J.min())
    delta = 0.0 if jmin > 0 else math.sqrt(0.1 * float(abs(J).mean()) * (0.1 * float(abs(J).mean()) - jmin))
    h = 0.5 * (J + np.sqrt(J * J + delta * delta))
    if np.any(h <= 0):
        return math.inf
    return float((L2 * sn / (2.0 * h)).sum())


try:
    import numba as _numba

    @_numba.njit(cache=True, fastmath=False)
    def _node_objective_numba(P, sn, use_a, unit_a, use_b, unit_b):  # pragma: no cover - compiled
        K = P.shape[0]
        L2 = np.empty(K)
        J = np.empty(K)
        for k in range(K):
            ax = P[k, 2, 0] - P[k, 1, 0]
            ay = P[k, 2, 1] - P[k, 1, 1]
            bx = P[k, 0, 0] - P[k, 1, 0]
            by = P[k, 0, 1] - P[k, 1, 1]
            L2[k] = (ax * ax + ay * ay) + (bx * bx + by * by)
            if use_a[k]:
                la = math.sqrt(ax * ax + ay * ay)
                ax = la * unit_a[k, 0]
                ay = la * unit_a[k, 1]
            if use_b[k]:
                lb = math.sqrt(bx * bx + by * by)
                bx = lb * unit_b[k, 0]
                by = lb * unit_b[k, 1]
            J[k] = ax * by - ay * bx
        jmin = J[0]
        sabs = 0.0
        for k in range(K):
            if J[k] < jmin:
                jmin = J[k]
            sabs += abs(J[k])
        delta = 0.0
        if jmin <= 0:
            m = 0.1 * (sabs / K)
            delta = math.sqrt(m * (m - jmin))
        f = 0.0
        for k in range(K):
            h = 0.5 * (J[k] + math.sqrt(J[k] * J[k] + delta * delta))
            if h <= 0:
                return math.inf
            f += L2[k] * sn[k] / (2.0 * h)
        return f

    _node_objective = _node_objective_numba
except ImportError:  # pragma: no cover - exercised only without numba
    _node_objective = _node_objective_numpy


def _optimize_node(P0: np.ndarray, mask: np.ndarray, sn: np.ndarray, x0: np.ndarray,
                   newton_iters: int, dirs=None):
    """Minimise the regularised corner distortion over the position x of one node.

    The tangent directions in `dirs` are CONSTANT with respect to x: an arc side
    joins two boundary nodes, so no side of a corner centred on the node being
    moved is ever curved, and where a curved side does appear -- on a corner
    centred at a boundary neighbour -- it points away from x. So the Newton
    machinery below is unchanged, and since it differentiates by finite
    differences it picks the new objective up for free.

    Performance: the objective is evaluated ~50 times per node on ~8 corners,
    so it is all small-array overhead. The tangent masks and unit directions
    are therefore computed once here (`_node_tangent_terms`), the corner terms
    once per evaluation, and the evaluation itself is a compiled kernel when
    numba is installed. The numba path sums in a different order from numpy,
    which the 1e-6 finite-difference Hessian amplifies into a different
    Newton PATH on some nodes; the objective reached agrees to ~1e-6 relative
    and gate verdicts were unchanged on paired meshes. Without numba the
    numpy path reproduces the previous results bit for bit.
    """
    P0 = np.ascontiguousarray(P0, dtype=np.float64)
    K = P0.shape[0]
    a0 = P0[:, 2] - P0[:, 1]
    b0 = P0[:, 0] - P0[:, 1]
    L2mean = float(((a0 ** 2).sum(axis=1) + (b0 ** 2).sum(axis=1)).mean())
    scale = math.sqrt(L2mean / 2.0)
    hstep = 1e-6 * scale
    use_a, unit_a, use_b, unit_b = _node_tangent_terms(K, dirs)
    sn = np.ascontiguousarray(sn, dtype=np.float64)
    P = P0.copy()
    rows, cols = np.nonzero(mask)
    x = np.array(x0, dtype=np.float64)

    def F(xx):
        P[rows, cols] = xx
        return _node_objective(P, sn, use_a, unit_a, use_b, unit_b)

    f = F(x)
    for _ in range(newton_iters):
        ex, ey = np.array([hstep, 0.0]), np.array([0.0, hstep])
        fxp, fxm, fyp, fym = F(x + ex), F(x - ex), F(x + ey), F(x - ey)
        if not all(map(math.isfinite, (fxp, fxm, fyp, fym))):
            hstep *= 0.1
            continue
        g = np.array([(fxp - fxm), (fyp - fym)]) / (2 * hstep)
        hxx = (fxp - 2 * f + fxm) / hstep ** 2
        hyy = (fyp - 2 * f + fym) / hstep ** 2
        fpp, fmm = F(x + ex + ey), F(x - ex - ey)
        hxy = (fpp - fxp - fyp + 2 * f - fxm - fym + fmm) / (2 * hstep ** 2)
        H = np.array([[hxx, hxy], [hxy, hyy]])
        det = hxx * hyy - hxy * hxy
        if det > 0 and hxx > 0:
            d = -np.linalg.solve(H, g)
        else:
            gn = float(np.hypot(*g))
            d = -g / gn * scale * 0.2 if gn > 0 else np.zeros(2)
        alpha, improved = 1.0, False
        for _ in range(12):
            fn = F(x + alpha * d)
            if fn < f:
                x, f, improved = x + alpha * d, fn, True
                break
            alpha *= 0.5
        if not improved or float(np.hypot(*(alpha * d))) < 1e-12 * math.sqrt(L2mean):
            break
    return x


def optimize(mesh: Mesh, iters: int = 5, fixed=None, newton_iters: int = 6,
             tangents=None) -> Mesh:
    """Simultaneous untangling and smoothing by per-node optimisation (Gauss-Seidel sweeps)."""
    out = mesh.copy()
    free = np.nonzero(~_fixed_mask(mesh, fixed))[0]
    nc = _NodeCorners(mesh, tangents)
    X = out.nodes
    local = {int(i): nc.of_node(int(i)) for i in free}
    for _ in range(iters):
        for i in free:
            tri, sn, dirs = local[int(i)]
            P0 = X[tri]
            X[i] = _optimize_node(P0, tri == i, sn, X[i], newton_iters, dirs)
    return out


def smooth(mesh: Mesh, method: str = "smart", **kw) -> Mesh:
    """Dispatcher: method in {"laplacian", "smart", "optimize"}."""
    return {"laplacian": laplacian, "smart": smart_laplacian, "optimize": optimize}[method](mesh, **kw)
