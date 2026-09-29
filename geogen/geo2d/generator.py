"""Seeded generator of lattice-based "generic mechanical part" domains.

Pipeline (see plan_2d_generator.md, section 4):
  1. occupancy raster on an integer lattice: base rectangle, then random
     rectangle cuts / additions / combs anchored on the current boundary,
     each kept only if the material stays connected, pinch-free and without
     enclosed voids;
  2. trace the raster boundary into a counter-clockwise lattice polygon;
  3. corner and edge modifiers with exact geometry: chamfers (45 degrees),
     fillets (90-degree arcs), paired modifiers giving rounded or V-shaped
     ends, semicircular notches and bulges;
  4. interior features: circular / diamond / slot / rectangular holes,
     optionally in linear arrays, placed with >= 1 unit clearance;
  5. optional mirror symmetry, implemented by duplicating every operation.

One lattice unit is the smallest feature size, so a shape with ratio R has
bounding box <= R x R and can be meshed with a uniform mesh of size 1.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import descriptors as _desc
from .geometry import Geometry, Loop, circle_loop, slot_loop
from .validate import loops_intersect, point_in_polygon, self_intersections, validate as _validate

_TOL = 1e-9


@dataclass
class Params:
    ratio: int = 10                  # lattice size = largest bbox side / smallest feature
    aspect: tuple = (1.0, 1.6)       # bbox aspect ratio (range or fixed)
    n_ops: tuple = (2, 5)            # raster boundary operations (cut / add / comb)
    n_mods: tuple = (1, 4)           # corner and edge modifiers (chamfer, fillet, pair, notch, bulge)
    n_holes: tuple = (0, 3)          # interior features
    angles: tuple = (45, 90)         # (90,) disables chamfers and diamonds
    arcs: str = "fillets"            # "none" | "fillets" | "all" (adds semicircular notches / bulges)
    symmetry: float | str = 0.5      # probability of a mirror-symmetric shape, or "none" | "x" | "y" | "xy"
    patterns: bool = True            # linear arrays of holes
    acute: float = 0.2               # probability that a chamfer may leave a 45-degree corner
    comb: float = 0.1                # probability that a raster op is a comb of narrow slots
    hole_types: dict = field(default_factory=lambda: {"circle": 0.6, "slot": 0.25, "rect": 0.15})
    hole_radius: tuple = (1, 2)
    max_tries: int = 40

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    "default": {},
    "plain": dict(angles=(90,), arcs="none", n_mods=(0, 0), n_holes=(0, 0), comb=0.0),
    "polycube": dict(angles=(90,), arcs="none", n_mods=(0, 0), hole_types={"rect": 1.0}, comb=0.0),
    "chamfered": dict(angles=(45, 90), arcs="none", n_mods=(2, 5)),
    "rounded": dict(angles=(90,), arcs="all", n_mods=(2, 5)),
    "holey": dict(n_holes=(2, 5), n_ops=(1, 3), n_mods=(0, 2)),
    "comb": dict(comb=0.6, n_ops=(2, 4)),
    "star": dict(acute=1.0, angles=(45, 90), arcs="none", n_ops=(3, 6), n_mods=(3, 7), n_holes=(0, 1)),
    "fine": dict(ratio=20, n_ops=(4, 8), n_mods=(3, 7), n_holes=(1, 5)),
    # complexity ladder
    "simple": dict(ratio=8, n_ops=(1, 2), n_mods=(0, 2), n_holes=(0, 1), comb=0.0),
    "medium": dict(ratio=12, n_ops=(3, 5), n_mods=(2, 4), n_holes=(1, 3)),
    "complex": dict(ratio=20, n_ops=(5, 9), n_mods=(4, 8), n_holes=(2, 5), arcs="all", comb=0.2),
    # straight lines only (45 and 90 degrees), holes are rectangles and diamonds
    "straight": dict(arcs="none"),
}


# --------------------------------------------------------------------------- sampling helpers

def _sample_int(rng, spec) -> int:
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    lo, hi = spec
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_float(rng, spec) -> float:
    if isinstance(spec, (int, float, np.floating)):
        return float(spec)
    lo, hi = spec
    return float(rng.uniform(lo, hi))


def _choice(rng, weights: dict):
    keys = list(weights)
    w = np.array([float(weights[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def _geom_int(rng, lo: int, hi: int, p: float = 0.5) -> int:
    """Integer in [lo, hi], biased towards lo (geometric with parameter p)."""
    k = lo
    while k < hi and rng.random() > p:
        k += 1
    return k


# --------------------------------------------------------------------------- raster helpers

def _neighbors4(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    out[1:, :] |= a[:-1, :]
    out[:-1, :] |= a[1:, :]
    out[:, 1:] |= a[:, :-1]
    out[:, :-1] |= a[:, 1:]
    return out


def _flood(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    reach = seed & mask
    while True:
        new = (reach | _neighbors4(reach)) & mask
        if np.array_equal(new, reach):
            return reach
        reach = new


def _connected(occ: np.ndarray) -> bool:
    if not occ.any():
        return False
    seed = np.zeros_like(occ)
    seed[tuple(np.argwhere(occ)[0])] = True
    return bool(np.array_equal(_flood(occ, seed), occ))


def _has_pinch(occ: np.ndarray) -> bool:
    a, b = occ[:-1, :-1], occ[1:, 1:]
    c, d = occ[1:, :-1], occ[:-1, 1:]
    return bool(((a & b & ~c & ~d) | (c & d & ~a & ~b)).any())


def _has_void(occ: np.ndarray) -> bool:
    empty = np.pad(~occ, 1, constant_values=True)
    seed = np.zeros_like(empty)
    seed[0, 0] = True
    return bool((empty & ~_flood(empty, seed)).any())


def _raster_ok(occ: np.ndarray) -> bool:
    return occ.any() and _connected(occ) and not _has_pinch(occ) and not _has_void(occ)


def _trace(occ: np.ndarray) -> np.ndarray:
    """Counter-clockwise lattice polygon of a connected, pinch-free, void-free raster."""
    W, H = occ.shape
    nxt: dict = {}
    n_faces = 0
    for xx, yy in np.argwhere(occ):
        x, y = int(xx), int(yy)
        if y == 0 or not occ[x, y - 1]:
            nxt[(x, y)] = (x + 1, y)
            n_faces += 1
        if x == W - 1 or not occ[x + 1, y]:
            nxt[(x + 1, y)] = (x + 1, y + 1)
            n_faces += 1
        if y == H - 1 or not occ[x, y + 1]:
            nxt[(x + 1, y + 1)] = (x, y + 1)
            n_faces += 1
        if x == 0 or not occ[x - 1, y]:
            nxt[(x, y + 1)] = (x, y)
            n_faces += 1
    if len(nxt) != n_faces:
        raise RuntimeError("raster boundary has a pinch")
    start = min(nxt)
    pts = [start]
    cur = nxt[start]
    while cur != start:
        pts.append(cur)
        cur = nxt[cur]
        if len(pts) > n_faces:
            raise RuntimeError("raster boundary does not close")
    if len(pts) != n_faces:
        raise RuntimeError("raster boundary is not a single loop")
    P = np.array(pts, dtype=int)
    n = len(P)
    keep = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            keep.append(i)
    return P[keep]


# --------------------------------------------------------------------------- exact assembly

def _assemble(P: np.ndarray, corner_mods: dict, edge_feats: dict, base_tag: str) -> Loop:
    """Build a Loop from a lattice polygon plus corner modifiers and edge features.

    corner_mods: vertex index -> (kind, k) with kind in {"fillet", "chamfer"}.
    edge_feats:  edge index -> list of (a, r, kind), kind in {"notch", "bulge"};
                 the feature occupies [a, a + 2r] along the edge.
    Works for counter-clockwise and clockwise polygons (material on the left).
    """
    P = np.asarray(P, float)
    n = len(P)
    u, L = [], []
    for i in range(n):
        d = P[(i + 1) % n] - P[i]
        length = math.hypot(d[0], d[1])
        u.append(d / length)
        L.append(length)
    pts, angs, tags = [], [], []

    def emit(p, a, t):
        pts.append(np.asarray(p, float))
        angs.append(float(a))
        tags.append(t)

    for i in range(n):
        k_start = corner_mods[i][1] if i in corner_mods else 0.0
        k_end = corner_mods[(i + 1) % n][1] if (i + 1) % n in corner_mods else 0.0
        if i in corner_mods:
            kind, k = corner_mods[i]
            a_pt = P[i] - k * u[i - 1]
            turn = u[i - 1][0] * u[i][1] - u[i - 1][1] * u[i][0]
            sign = 1.0 if turn > 0 else -1.0
            emit(a_pt, sign * 90.0 if kind == "fillet" else 0.0, kind)
        pos = float(k_start)
        end = L[i] - k_end
        nL = np.array([-u[i][1], u[i][0]])
        for a, r, kind in sorted(edge_feats.get(i, [])):
            if a > pos + _TOL:
                emit(P[i] + pos * u[i], 0.0, base_tag)
            E0 = P[i] + a * u[i]
            M = E0 + r * u[i]
            if kind == "notch":
                emit(E0, -90.0, kind)
                emit(M + r * nL, -90.0, kind)
            else:
                emit(E0, 90.0, kind)
                emit(M - r * nL, 90.0, kind)
            pos = a + 2 * r
        if end > pos + _TOL:
            emit(P[i] + pos * u[i], 0.0, base_tag)
    return Loop(np.array(pts), np.array(angs), tags)


# --------------------------------------------------------------------------- the builder

class _Builder:
    def __init__(self, rng, P: Params, seed: int):
        self.P = P
        self.rng = rng
        self.seed = seed
        self.W = int(P.ratio)
        aspect = _sample_float(rng, P.aspect)
        self.H = max(3, int(round(self.W / aspect)))
        self.occ = np.zeros((self.W, self.H), dtype=bool)
        self.program: list = []
        self.axes = self._choose_symmetry()
        self.allow_chamfer = 45 in tuple(P.angles)
        self.allow_fillet = P.arcs in ("fillets", "all")
        self.allow_edgefeat = P.arcs == "all"

    # ---- symmetry -------------------------------------------------------
    def _choose_symmetry(self) -> set:
        s = self.P.symmetry
        if isinstance(s, str):
            return {"none": set(), "x": {"x"}, "y": {"y"}, "xy": {"x", "y"}}[s]
        if self.rng.random() < float(s):
            return {"x": {"x"}, "y": {"y"}, "xy": {"x", "y"}}[_choice(self.rng, {"x": 0.6, "y": 0.25, "xy": 0.15})]
        return set()

    def _transforms(self):
        """Point transforms of the symmetry group (identity first)."""
        W, H = self.W, self.H
        ts = [lambda p: (p[0], p[1])]
        if "x" in self.axes:
            ts = ts + [(lambda t: (lambda p: (W - t(p)[0], t(p)[1])))(t) for t in ts]
        if "y" in self.axes:
            ts = ts + [(lambda t: (lambda p: (t(p)[0], H - t(p)[1])))(t) for t in ts]
        return ts

    def _images_pt(self, p):
        out = []
        for t in self._transforms():
            q = t((float(p[0]), float(p[1])))
            if q not in out:
                out.append(q)
        return out

    def _images_rect(self, r):
        x0, y0, x1, y1 = r
        out = []
        for t in self._transforms():
            a, b = t((x0, y0)), t((x1, y1))
            q = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            if q not in out:
                out.append(q)
        return out

    # ---- raster stage ---------------------------------------------------
    def _apply_rects(self, rects, value: bool) -> bool:
        cand = self.occ.copy()
        for x0, y0, x1, y1 in rects:
            x0c, x1c = max(0, int(x0)), min(self.W, int(x1))
            y0c, y1c = max(0, int(y0)), min(self.H, int(y1))
            if x1c <= x0c or y1c <= y0c:
                return False
            cand[x0c:x1c, y0c:y1c] = value
        if np.array_equal(cand, self.occ) or not _raster_ok(cand):
            return False
        self.occ = cand
        return True

    def _base(self) -> None:
        rng, W, H = self.rng, self.W, self.H
        for _ in range(100):
            wb = int(rng.integers(math.ceil(0.6 * W), W + 1))
            hb = int(rng.integers(math.ceil(0.5 * H), H + 1))
            x0 = int(rng.integers(0, W - wb + 1))
            y0 = int(rng.integers(0, H - hb + 1))
            rects = self._images_rect((x0, y0, x0 + wb, y0 + hb))
            if self._apply_rects(rects, True):
                self.program.append({"op": "base", "rects": rects})
                return
        raise RuntimeError("could not place a base rectangle")

    def _faces(self):
        occ, W, H = self.occ, self.W, self.H
        faces = []
        for xx, yy in np.argwhere(occ):
            x, y = int(xx), int(yy)
            if x == W - 1 or not occ[x + 1, y]:
                faces.append((x, y, 0))
            if y == H - 1 or not occ[x, y + 1]:
                faces.append((x, y, 1))
            if x == 0 or not occ[x - 1, y]:
                faces.append((x, y, 2))
            if y == 0 or not occ[x, y - 1]:
                faces.append((x, y, 3))
        return faces

    def _is_face(self, f) -> bool:
        x, y, dr = f
        W, H = self.W, self.H
        if not (0 <= x < W and 0 <= y < H) or not self.occ[x, y]:
            return False
        dx, dy = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[dr]
        nx, ny = x + dx, y + dy
        return not (0 <= nx < W and 0 <= ny < H and self.occ[nx, ny])

    @staticmethod
    def _face_rect(face, w: int, d: int, off: int, outward: bool):
        x, y, dr = face
        if dr in (0, 2):
            y0 = y - off
            if dr == 0:
                xs = (x + 1, x + 1 + d) if outward else (x - d + 1, x + 1)
            else:
                xs = (x - d, x) if outward else (x, x + d)
            return (xs[0], y0, xs[1], y0 + w)
        x0 = x - off
        if dr == 1:
            ys = (y + 1, y + 1 + d) if outward else (y - d + 1, y + 1)
        else:
            ys = (y - d, y) if outward else (y, y + d)
        return (x0, ys[0], x0 + w, ys[1])

    def _comb(self, face) -> bool:
        rng = self.rng
        n_teeth = int(rng.integers(3, 6))
        depth = int(rng.integers(1, 3))
        off = int(rng.integers(0, 2 * n_teeth - 1))
        x, y, dr = face
        rects = []
        for t in range(n_teeth):
            tooth = (x, y - off + 2 * t, dr) if dr in (0, 2) else (x - off + 2 * t, y, dr)
            if not self._is_face(tooth):
                return False
            rects += self._images_rect(self._face_rect(tooth, 1, depth, 0, outward=False))
        if self._apply_rects(rects, False):
            self.program.append({"op": "comb", "rects": rects})
            return True
        return False

    def _raster_op(self) -> bool:
        rng = self.rng
        smax = max(2, min(self.W, self.H) // 2)
        for _ in range(self.P.max_tries):
            faces = self._faces()
            face = faces[int(rng.integers(len(faces)))]
            if rng.random() < self.P.comb:
                if self._comb(face):
                    return True
                continue
            kind = "cut" if rng.random() < 0.6 else "add"
            w = _geom_int(rng, 1, smax, 0.35)
            d = _geom_int(rng, 1, smax, 0.4)
            off = int(rng.integers(0, w))
            rects = self._images_rect(self._face_rect(face, w, d, off, outward=(kind == "add")))
            if self._apply_rects(rects, kind == "add"):
                self.program.append({"op": kind, "rects": rects})
                return True
        return False

    # ---- modifier stage -------------------------------------------------
    def _prepare_polygon(self) -> None:
        self.Pv = _trace(self.occ)
        n = len(self.Pv)
        self.n = n
        self.u, self.L = [], []
        for i in range(n):
            d = self.Pv[(i + 1) % n] - self.Pv[i]
            length = float(math.hypot(d[0], d[1]))
            self.u.append(d / length)
            self.L.append(length)
        self.turn = [1 if (self.u[i - 1][0] * self.u[i][1] - self.u[i - 1][1] * self.u[i][0]) > 0 else -1
                     for i in range(n)]
        self.vidx = {(float(p[0]), float(p[1])): i for i, p in enumerate(self.Pv)}
        self.cmods: dict = {}
        self.efeats: dict = {}
        self.used: dict = {i: [] for i in range(n)}   # occupied intervals along each edge

    def _free(self, i: int, s0: float, s1: float, used=None) -> bool:
        used = self.used if used is None else used
        return all(s1 <= a + _TOL or s0 >= b - _TOL for a, b in used[i])

    def _tail_free(self, i: int, used=None) -> float:
        used = self.used if used is None else used
        return self.L[i] - max([b for _, b in used[i]], default=0.0)

    def _head_free(self, i: int, used=None) -> float:
        used = self.used if used is None else used
        return min([a for a, _ in used[i]], default=self.L[i])

    def _edge_of(self, p, q):
        """Index of the straight lattice edge containing both points, or None."""
        for j in range(self.n):
            P0, uj, Lj = self.Pv[j].astype(float), self.u[j], self.L[j]
            ok = True
            for X in (p, q):
                v = np.asarray(X, float) - P0
                s = float(v @ uj)
                if abs(v[0] * uj[1] - v[1] * uj[0]) > _TOL or s < -_TOL or s > Lj + _TOL:
                    ok = False
                    break
            if ok:
                return j
        return None

    def _corner_cells_ok(self, v: int, k: int) -> bool:
        """The k x k square at the corner must be solid material (convex corner)
        or empty space inside the lattice (concave corner)."""
        V = self.Pv[v].astype(float)
        a, b = -self.u[v - 1], self.u[v]
        want = self.turn[v] > 0
        for s in range(k):
            for t in range(k):
                c = V + (s + 0.5) * a + (t + 0.5) * b
                cx, cy = int(math.floor(c[0])), int(math.floor(c[1]))
                if not (0 <= cx < self.W and 0 <= cy < self.H) or bool(self.occ[cx, cy]) != want:
                    return False
        return True

    def _outer_ok(self) -> bool:
        """Exact check of the assembled outer loop: no touching/crossing and ligaments >= 1."""
        try:
            loop = _assemble(self.Pv, self.cmods, self.efeats, "outer")
        except ValueError:
            # the modifiers can consume the outline entirely -- two unit chamfers on
            # opposite corners of a unit-wide rectangle leave fewer than three
            # points (seed 11919, preset "straight") -- and that is an invalid
            # candidate, not an error to raise out of the generator
            return False
        g = Geometry([loop])
        if self_intersections(g) > 0:
            return False
        return _desc.min_ligament(g) >= 1.0 - 1e-6

    def _try_corner_mods(self, mods) -> bool:
        """mods: list of (vertex, kind, k); commit all or nothing."""
        used = {i: list(v) for i, v in self.used.items()}
        cm = dict(self.cmods)
        for v, kind, k in mods:
            v = int(v) % self.n
            if v in cm or not self._corner_cells_ok(v, int(k)):
                return False
            e0, e1 = (v - 1) % self.n, v
            if not self._free(e0, self.L[e0] - k, self.L[e0], used) or not self._free(e1, 0.0, k, used):
                return False
            used[e0].append((self.L[e0] - k, self.L[e0]))
            used[e1].append((0.0, k))
            cm[v] = (kind, float(k))
        old = (self.used, self.cmods)
        self.used, self.cmods = used, cm
        if not self._outer_ok():
            self.used, self.cmods = old
            return False
        return True

    def _pick_kind(self) -> str | None:
        if self.allow_fillet and self.allow_chamfer:
            return "fillet" if self.rng.random() < 0.6 else "chamfer"
        if self.allow_fillet:
            return "fillet"
        if self.allow_chamfer:
            return "chamfer"
        return None

    def _corner_op(self) -> bool:
        rng = self.rng
        kind0 = self._pick_kind()
        if kind0 is None:
            return False
        for _ in range(self.P.max_tries):
            i = int(rng.integers(self.n))
            if i in self.cmods:
                continue
            kind = self._pick_kind()
            e0, e1 = (i - 1) % self.n, i
            kmax = int(min(self._tail_free(e0), self._head_free(e1)))
            if kmax < 1:
                continue
            k = _geom_int(rng, 1, kmax, 0.5)
            # a chamfer that alone consumes a whole edge creates a 45-degree corner
            for e, free in ((e0, self._tail_free(e0)), (e1, self._head_free(e1))):
                if k == free and not self.used[e] and kind == "chamfer" and rng.random() >= self.P.acute:
                    k = int(free) - 1
            if k < 1:
                continue
            mods = []
            for q in self._images_pt(self.Pv[i]):
                if q not in self.vidx:
                    mods = None
                    break
                mods.append((self.vidx[q], kind, k))
            if mods and self._try_corner_mods(mods):
                self.program.append({"op": "corner", "kind": kind, "k": k,
                                     "at": [[int(self.Pv[v][0]), int(self.Pv[v][1])] for v, _, _ in mods]})
                return True
        return False

    def _pair_op(self) -> bool:
        """Same modifier at both ends of a short edge: rounded ends (fillets) or 90-degree tips / V-notches (chamfers)."""
        rng = self.rng
        if self._pick_kind() is None:
            return False
        for _ in range(self.P.max_tries):
            i = int(rng.integers(self.n))
            L = int(round(self.L[i]))
            v0, v1 = i, (i + 1) % self.n
            if L % 2 or L < 2 or L > 6 or v0 in self.cmods or v1 in self.cmods or self.used[i]:
                continue
            if self.turn[v0] != self.turn[v1]:
                continue
            k = L // 2
            kind = self._pick_kind()
            if self._tail_free((i - 1) % self.n) < k or self._head_free((i + 1) % self.n) < k:
                continue
            mods = []
            ok = True
            for t in self._transforms():
                a = t((float(self.Pv[v0][0]), float(self.Pv[v0][1])))
                b = t((float(self.Pv[v1][0]), float(self.Pv[v1][1])))
                if a not in self.vidx or b not in self.vidx:
                    ok = False
                    break
                for q in (a, b):
                    if (self.vidx[q], kind, k) not in mods:
                        mods.append((self.vidx[q], kind, k))
            if ok and self._try_corner_mods(mods):
                self.program.append({"op": "pair", "kind": kind, "k": k,
                                     "at": [[int(self.Pv[v][0]), int(self.Pv[v][1])] for v, _, _ in mods]})
                return True
        return False

    def _cells_ok(self, i: int, a: float, r: int, kind: str) -> bool:
        """Raster clearance for a semicircular notch (material behind) or bulge (empty space in front)."""
        P0, uu = self.Pv[i].astype(float), self.u[i]
        nL = np.array([-uu[1], uu[0]])
        side = nL if kind == "notch" else -nL
        for s in range(int(a) - 1, int(a) + 2 * r + 1):
            for j in range(0, r + 1):
                c = P0 + (s + 0.5) * uu + (j + 0.5) * side
                cx, cy = int(math.floor(c[0])), int(math.floor(c[1]))
                if not (0 <= cx < self.W and 0 <= cy < self.H):
                    return False
                if self.occ[cx, cy] != (kind == "notch"):
                    return False
        return True

    def _edgefeat_op(self) -> bool:
        rng = self.rng
        for _ in range(self.P.max_tries):
            i = int(rng.integers(self.n))
            L = int(round(self.L[i]))
            r = _geom_int(rng, 1, 2, 0.6)
            if L < 2 * r + 2:
                continue
            kind = "notch" if rng.random() < 0.6 else "bulge"
            choices = [a for a in range(1, L - 2 * r) if self._free(i, a - 1, a + 2 * r + 1)]
            if not choices:
                continue
            a = int(choices[int(rng.integers(len(choices)))])
            if not self._cells_ok(i, a, r, kind):
                continue
            feats = []
            ok = True
            used = {j: list(v) for j, v in self.used.items()}
            for t in self._transforms():
                E0 = self.Pv[i] + a * self.u[i]
                E1 = E0 + 2 * r * self.u[i]
                q0, q1 = t((float(E0[0]), float(E0[1]))), t((float(E1[0]), float(E1[1])))
                j = self._edge_of(q0, q1)
                if j is None:
                    ok = False
                    break
                s0 = float((np.asarray(q0) - self.Pv[j]) @ self.u[j])
                s1 = float((np.asarray(q1) - self.Pv[j]) @ self.u[j])
                aj = int(round(min(s0, s1)))
                if (j, aj) in [(f[0], f[1]) for f in feats]:
                    continue
                if not self._free(j, aj - 1, aj + 2 * r + 1, used) or not self._cells_ok(j, aj, r, kind):
                    ok = False
                    break
                used[j].append((aj - 1, aj + 2 * r + 1))
                feats.append((j, aj, r, kind))
            if not ok:
                continue
            old = (self.used, self.efeats)
            self.used = used
            self.efeats = {j: list(v) for j, v in self.efeats.items()}
            for j, aj, rr, kk in feats:
                self.efeats.setdefault(j, []).append((float(aj), int(rr), kk))
            if not self._outer_ok():
                self.used, self.efeats = old
                continue
            self.program.append({"op": kind, "r": r, "at": [[int(self.Pv[j][0]), int(self.Pv[j][1]), aj] for j, aj, _, _ in feats]})
            return True
        return False

    def _mod_op(self) -> bool:
        weights = {}
        if self.allow_fillet or self.allow_chamfer:
            weights.update({"corner": 0.5, "pair": 0.3})
        if self.allow_edgefeat:
            weights["edgefeat"] = 0.25
        if not weights:
            return False
        return {"corner": self._corner_op, "pair": self._pair_op, "edgefeat": self._edgefeat_op}[_choice(self.rng, weights)]()

    # ---- holes ----------------------------------------------------------
    def _rect_hole(self, rect, corner) -> Loop:
        x0, y0, x1, y1 = rect
        P = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], float)   # clockwise
        mods = {}
        if corner in ("fillet", "chamfer") and min(x1 - x0, y1 - y0) >= 2:
            mods = {v: (corner, 1.0) for v in range(4)}
        return _assemble(P, mods, {}, "rect")

    def _block_solid(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """Cells [x0, x1) x [y0, y1) all inside the lattice and material."""
        if x0 < 0 or y0 < 0 or x1 > self.W or y1 > self.H or x1 <= x0 or y1 <= y0:
            return False
        return bool(self.occ[x0:x1, y0:y1].all())

    def _feasible_anchors(self, ex: int, ey: int, w: int = 0, h: int = 0) -> list:
        """Lattice points p such that the block from p-(ex,ey) to p+(w,h)+(ex,ey), padded
        by one clearance cell, is solid material (raster pre-check for hole placement)."""
        out = []
        for x in range(ex + 1, self.W - w - ex):
            for y in range(ey + 1, self.H - h - ey):
                if self._block_solid(x - ex - 1, y - ey - 1, x + w + ex + 1, y + h + ey + 1):
                    out.append((x, y))
        return out

    def _sample_hole(self):
        rng, P = self.rng, self.P
        types = dict(P.hole_types)
        if not self.allow_fillet:                      # no arcs anywhere: rectangles, and diamonds if 45 deg allowed
            types.pop("slot", None)
            if not self.allow_chamfer:
                types.pop("circle", None)
            types = types or {"rect": 1.0}
        kind = _choice(rng, types)
        r = _geom_int(rng, int(P.hole_radius[0]), int(P.hole_radius[1]), 0.6)
        desc = {"op": "hole", "kind": kind}
        w = h = 0
        if kind == "circle":
            diamond = self.allow_chamfer and (not self.allow_fillet or rng.random() < 0.1)
            desc.update(r=r, diamond=diamond)

            def make(c):
                lp = circle_loop(c[0], c[1], r, hole=True, tag="hole")
                return Loop(lp.points, np.zeros(4), ["diamond"] * 4) if diamond else lp
            ex, ey = r, r
        elif kind == "slot":
            half = int(rng.integers(1, 4))
            axis = "x" if rng.random() < 0.6 else "y"
            desc.update(r=r, half=half, axis=axis)

            def make(c):
                return slot_loop(c[0], c[1], r, half, axis, hole=True)
            ex, ey = (half + r, r) if axis == "x" else (r, half + r)
        else:
            w, h = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            corner = _choice(rng, {"none": 0.4, "fillet": 0.35 if self.allow_fillet else 0.0,
                                   "chamfer": 0.25 if self.allow_chamfer else 0.0})
            desc.update(w=w, h=h, corner=corner)

            def make(c):
                return self._rect_hole((c[0], c[1], c[0] + w, c[1] + h), corner)
            ex, ey = 0, 0
        feasible = self._feasible_anchors(ex, ey, w, h)
        if not feasible:
            return None, desc
        fset = set(feasible)
        cx, cy = feasible[int(rng.integers(len(feasible)))]
        centers = [(cx, cy)]
        if P.patterns and rng.random() < 0.4:
            m = int(rng.integers(2, 4))
            axis = 0 if rng.random() < 0.6 else 1
            extent = (2 * ex + w, 2 * ey + h)
            spacing = extent[axis] + int(rng.integers(1, 3))
            cand = [(cx + t * spacing, cy) if axis == 0 else (cx, cy + t * spacing) for t in range(m)]
            while len(cand) > 1 and not all(c in fset for c in cand):
                cand = cand[:-1]
            if len(cand) > 1:
                centers = cand
                desc.update(pattern={"m": len(cand), "axis": "xy"[axis], "spacing": spacing})
        # symmetry images (rect holes are mirrored as rectangles, others by centre)
        loops = []
        for c in centers:
            if kind == "rect":
                for rr in self._images_rect((c[0], c[1], c[0] + w, c[1] + h)):
                    loops.append(self._rect_hole(rr, corner))
            else:
                for q in self._images_pt(c):
                    loops.append(make(q))
        uniq = []
        for lp in loops:
            if not any(lp.n == o.n and np.allclose(lp.points, o.points) for o in uniq):
                uniq.append(lp)
        desc["centers"] = [list(map(int, c)) for c in centers]
        return uniq, desc

    def _holes_ok(self, cands, existing, outer_poly, hole_polys) -> bool:
        """Candidates must lie inside the material, not intersect anything, and keep >= 1 clearance."""
        cand_polys = [lp.discretize(22.5) for lp in cands]
        for lp, cp in zip(cands, cand_polys):
            if not point_in_polygon(cp[0], outer_poly):
                return False
            for hp in hole_polys:
                if point_in_polygon(cp[0], hp) or point_in_polygon(hp[0], cp):
                    return False
            for ex, ep in existing:
                if loops_intersect(cp, ep) or _desc.loops_min_distance(lp, ex, stop_below=0.98) < 0.98:
                    return False
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                if loops_intersect(cand_polys[a], cand_polys[b]) or point_in_polygon(cand_polys[a][0], cand_polys[b]) \
                        or _desc.loops_min_distance(cands[a], cands[b], stop_below=0.98) < 0.98:
                    return False
        return True

    def _holes(self, outer: Loop) -> list[Loop]:
        outer_poly = outer.discretize(22.5)
        existing = [(outer, outer_poly)]
        hole_polys: list = []
        holes: list[Loop] = []
        for _ in range(_sample_int(self.rng, self.P.n_holes)):
            for _ in range(self.P.max_tries):
                cands, desc = self._sample_hole()
                if cands and self._holes_ok(cands, existing, outer_poly, hole_polys):
                    holes += cands
                    for lp in cands:
                        hp = lp.discretize(22.5)
                        existing.append((lp, hp))
                        hole_polys.append(hp)
                    self.program.append(desc)
                    break
        return holes

    # ---- driver ---------------------------------------------------------
    def run(self) -> Geometry:
        self._base()
        for _ in range(_sample_int(self.rng, self.P.n_ops)):
            self._raster_op()
        self._prepare_polygon()
        for _ in range(_sample_int(self.rng, self.P.n_mods)):
            self._mod_op()
        outer = _assemble(self.Pv, self.cmods, self.efeats, "outer")
        holes = self._holes(outer)
        geom = Geometry([outer] + holes)
        x0, y0, _, _ = geom.bbox()
        geom = geom.translated(-round(x0), -round(y0))
        geom.meta = {"seed": int(self.seed), "params": self.P.to_dict(), "lattice": [self.W, self.H],
                     "symmetry": sorted(self.axes), "program": self.program}
        return geom


def generate(seed: int, params: Params | None = None, preset: str | None = None, **overrides) -> Geometry:
    """Generate one valid geometry for ``seed``.  ``preset`` names an entry of PRESETS;
    keyword overrides replace individual Params fields."""
    P = params if params is not None else Params()
    if preset is not None:
        P = P.replace(**PRESETS[preset])
    if overrides:
        P = P.replace(**overrides)
    last = None
    for attempt in range(10):
        rng = np.random.default_rng([int(seed), attempt])
        try:
            geom = _Builder(rng, P, seed).run()
        except RuntimeError as exc:      # degenerate raster; try again with fresh randomness
            last = str(exc)
            continue
        problems = _validate(geom, min_feature=1.0, tol=1e-3)
        if not problems:
            geom.meta["attempt"] = attempt
            geom.meta["descriptors"] = _desc.compute(geom)
            return geom
        last = "; ".join(problems)
    raise RuntimeError(f"generate(seed={seed}) failed after 10 attempts: {last}")


def generate_many(seed: int, n: int, params: Params | None = None, preset: str | None = None,
                  **overrides) -> list[Geometry]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31 - 1, size=n)
    return [generate(int(s), params, preset, **overrides) for s in seeds]
