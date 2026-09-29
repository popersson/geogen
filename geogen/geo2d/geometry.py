"""Planar domains made of straight edges, circular arcs and cubic spline edges.

A ``Loop`` is a closed sequence of control points; edge ``i`` runs from
``points[i]`` to ``points[(i + 1) % n]``.  Three edge types:

* straight: ``angles[i] == 0`` and no interior points;
* arc: ``angles[i]`` is the signed included angle in degrees, positive =
  counter-clockwise turn from start to end, |angle| <= 90 (a circle is a
  square with four +90 edges, four -90 when traversed clockwise);
* spline: ``angles[i] == 0`` and ``interior[i]`` is an (m, 2) array of points;
  the edge is the interpolating cubic spline through start, interior points and
  end.  At a vertex flagged in ``smooth`` the adjacent spline edges share a
  tangent (G1 joint).

Convention: material lies on the left of every edge.  The outer loop of a
``Geometry`` is counter-clockwise and its holes are clockwise.  The control
polygon alone carries the topology; arcs and splines refine it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from . import curves as _curves
from .curves import (arc_bulge, arc_center, arc_length, arc_points, arc_radius,  # noqa: F401  (re-exported)
                     arc_tangents, bulge_angle)

MAX_ARC_DEG = 90.0
_EPS = 1e-9


def _num(x: float):
    xr = round(float(x))
    return int(xr) if abs(float(x) - xr) < 1e-9 else float(x)


# --------------------------------------------------------------------------

@dataclass(eq=False)
class Loop:
    points: np.ndarray
    angles: np.ndarray | None = None
    tags: list | None = None
    interior: list | None = None      # per edge: None or (m, 2) array of spline interior points
    smooth: list | None = None        # per vertex: True = G1 joint between the adjacent edges
    region: np.ndarray | None = None  # per edge: region id on the material (left) side, default 0
    _edges: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, float).reshape(-1, 2)
        n = len(self.points)
        if n < 3:
            raise ValueError("a loop needs at least 3 points")
        if self.region is not None:
            self.region = np.asarray(self.region, int).reshape(-1)
            if len(self.region) != n:
                raise ValueError("region must have one entry per edge")
            if not self.region.any():
                self.region = None
        if self.angles is None:
            self.angles = np.zeros(n)
        self.angles = np.asarray(self.angles, float).reshape(-1)
        if len(self.angles) != n:
            raise ValueError("angles must have one entry per edge")
        if np.any(np.abs(self.angles) > MAX_ARC_DEG + 1e-9):
            raise ValueError("arc angles are limited to +-90 degrees; split larger arcs")
        if self.tags is not None:
            self.tags = list(self.tags)
            if len(self.tags) != n:
                raise ValueError("tags must have one entry per edge")
        if self.interior is not None:
            if len(self.interior) != n:
                raise ValueError("interior must have one entry per edge")
            cleaned = []
            for i, x in enumerate(self.interior):
                if x is None or len(x) == 0:
                    cleaned.append(None)
                    continue
                if abs(self.angles[i]) > _EPS:
                    raise ValueError(f"edge {i}: a spline edge must have angle 0")
                cleaned.append(np.asarray(x, float).reshape(-1, 2))
            self.interior = cleaned if any(c is not None for c in cleaned) else None
        if self.smooth is not None:
            self.smooth = [bool(s) for s in self.smooth]
            if len(self.smooth) != n:
                raise ValueError("smooth must have one entry per vertex")
            if not any(self.smooth):
                self.smooth = None

    # -- basic access -------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.points)

    def kind(self, i: int) -> str:
        if self.interior is not None and self.interior[i] is not None:
            return "spline"
        return "arc" if abs(self.angles[i]) > _EPS else "line"

    def edge_region(self, i: int) -> int:
        return int(self.region[i % self.n]) if self.region is not None else 0

    def split_edge(self, i: int, t: float) -> "Loop":
        """New loop with edge i split at arc-length fraction t (lines and arcs only)."""
        i %= self.n
        if self.kind(i) == "spline":
            raise NotImplementedError("splitting spline edges is not supported; cut at a control point")
        p = self.edge(i).point(float(t))
        pts = np.insert(self.points, i + 1, p, axis=0)
        ang = np.insert(self.angles, i + 1, self.angles[i] * (1 - t))
        ang[i] = self.angles[i] * t
        tags = None if self.tags is None else self.tags[:i + 1] + [self.tags[i]] + self.tags[i + 1:]
        interior = None if self.interior is None else self.interior[:i + 1] + [None] + self.interior[i + 1:]
        smooth = None if self.smooth is None else self.smooth[:i + 1] + [False] + self.smooth[i + 1:]
        region = None if self.region is None else np.insert(self.region, i + 1, self.region[i])
        return Loop(pts, ang, tags, interior, smooth, region)

    def kinds(self) -> list[str]:
        return [self.kind(i) for i in range(self.n)]

    def is_smooth(self, v: int) -> bool:
        return bool(self.smooth[v % self.n]) if self.smooth is not None else False

    def edges(self) -> Iterator[tuple[int, np.ndarray, np.ndarray, float]]:
        """(i, start, end, angle) per edge (angle 0 for straight and spline edges)."""
        n = self.n
        for i in range(n):
            yield i, self.points[i], self.points[(i + 1) % n], float(self.angles[i])

    def is_arc(self) -> np.ndarray:
        return np.abs(self.angles) > _EPS

    def chord_lengths(self) -> np.ndarray:
        d = np.roll(self.points, -1, axis=0) - self.points
        return np.hypot(d[:, 0], d[:, 1])

    # -- evaluators -----------------------------------------------------------
    def _vertex_tangent(self, v: int, base: list) -> np.ndarray:
        n = self.n
        ein, eout = (v - 1) % n, v % n
        if base[ein] is not None:
            return base[ein].tangent(1.0)
        if base[eout] is not None:
            return base[eout].tangent(0.0)
        return _curves.bessel_tangent(self.interior[ein][-1], self.points[v % n], self.interior[eout][0])

    def evaluators(self) -> list:
        """Cached Line / Arc / Spline evaluator per edge (see curves.py)."""
        if self._edges is None:
            n = self.n
            base: list = [None] * n
            for i in range(n):
                k = self.kind(i)
                p0, p1 = self.points[i], self.points[(i + 1) % n]
                if k == "line":
                    base[i] = _curves.Line(p0, p1)
                elif k == "arc":
                    base[i] = _curves.Arc(p0, p1, float(self.angles[i]))
            splines = {}
            for i in range(n):
                if self.kind(i) == "spline":
                    pts = np.vstack([self.points[i], self.interior[i], self.points[(i + 1) % n]])
                    t0 = self._vertex_tangent(i, base) if self.is_smooth(i) else None
                    t1 = self._vertex_tangent(i + 1, base) if self.is_smooth(i + 1) else None
                    splines[i] = _curves.Spline(pts, t0, t1)
            for i, e in splines.items():
                base[i] = e
            self._edges = base
        return self._edges

    def edge(self, i: int):
        return self.evaluators()[i % self.n]

    def edge_lengths(self) -> np.ndarray:
        return np.array([e.length for e in self.evaluators()])

    # -- geometry -----------------------------------------------------------
    def signed_area(self) -> float:
        """Signed area (exact for lines and arcs, quadrature on the spline tables)."""
        x, y = self.points[:, 0], self.points[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        return area + sum(e.area_correction() for e in self.evaluators())

    def is_ccw(self) -> bool:
        return self.signed_area() > 0

    def reversed(self) -> "Loop":
        """Same curve traversed the other way (angle signs flip)."""
        n = self.n
        pts = self.points[::-1].copy()
        ang = np.array([-self.angles[(n - 2 - j) % n] for j in range(n)])
        tags = None if self.tags is None else [self.tags[(n - 2 - j) % n] for j in range(n)]
        interior = None
        if self.interior is not None:
            interior = [None if self.interior[(n - 2 - j) % n] is None else self.interior[(n - 2 - j) % n][::-1].copy()
                        for j in range(n)]
        smooth = None if self.smooth is None else [self.smooth[(n - 1 - j) % n] for j in range(n)]
        region = None if self.region is None else np.array([self.region[(n - 2 - j) % n] for j in range(n)])
        return Loop(pts, ang, tags, interior, smooth, region)

    def rolled(self, k: int) -> "Loop":
        """Same loop with vertex ``k`` first (edges, tags, interior and smooth follow)."""
        n = self.n
        k %= n
        roll = lambda seq: None if seq is None else [seq[(j + k) % n] for j in range(n)]
        return Loop(np.roll(self.points, -k, axis=0), np.roll(self.angles, -k), roll(self.tags),
                    roll(self.interior), roll(self.smooth), None if self.region is None else np.roll(self.region, -k))

    def discretize_edges(self, max_step_deg: float = 5.0) -> list[np.ndarray]:
        """Per edge, a polyline (end points included) approximating the edge."""
        return [e.polyline(max_step_deg) for e in self.evaluators()]

    def discretize(self, max_step_deg: float = 5.0) -> np.ndarray:
        """Closed polyline (first point not repeated) approximating the loop."""
        return np.vstack([e[:-1] for e in self.discretize_edges(max_step_deg)])

    def bbox(self) -> tuple[float, float, float, float]:
        b = np.array([e.bbox() for e in self.evaluators()])
        return (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))

    def tangents(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit tangents at each vertex: incoming (end of edge i-1) and outgoing (start of edge i)."""
        ev = self.evaluators()
        starts = np.array([e.tangent(0.0) for e in ev])
        ends = np.array([e.tangent(1.0) for e in ev])
        return np.roll(ends, 1, axis=0), starts

    def turns(self) -> np.ndarray:
        """Signed turning angle (degrees) at each vertex, positive = left turn."""
        tin, tout = self.tangents()
        cross = tin[:, 0] * tout[:, 1] - tin[:, 1] * tout[:, 0]
        dot = (tin * tout).sum(axis=1)
        return np.degrees(np.arctan2(cross, dot))

    def interior_angles(self) -> np.ndarray:
        """Angle of the material (on the left) at each vertex in degrees, taking the
        edge tangents into account: a circle has 180 at all four control points, a
        rectangular hole 270 at its corners.  180 means a smooth joint."""
        return 180.0 - self.turns()

    vertex_angles = interior_angles

    # -- transforms ---------------------------------------------------------
    def _with(self, f) -> "Loop":
        interior = None if self.interior is None else [None if x is None else f(x) for x in self.interior]
        return Loop(f(self.points), self.angles.copy(), self.tags, interior, self.smooth, self.region)

    def translated(self, dx: float, dy: float) -> "Loop":
        return self._with(lambda P: P + np.array([dx, dy]))

    def scaled(self, s: float) -> "Loop":
        if s <= 0:
            raise ValueError("scale must be positive")
        return self._with(lambda P: P * s)

    # -- serialization ------------------------------------------------------
    def to_dict(self, vertex_angles: bool = True) -> dict:
        """Dict for JSON.  ``vertex_angles`` (derived, ignored on load) adds the
        interior angle of the domain at every vertex for the mesh generator."""
        d = {"points": [[_num(x), _num(y)] for x, y in self.points],
             "angles": [_num(a) for a in self.angles]}
        if self.tags is not None:
            d["tags"] = list(self.tags)
        if self.interior is not None:
            d["interior"] = [None if x is None else [[_num(a), _num(b)] for a, b in x] for x in self.interior]
        if self.smooth is not None:
            d["smooth"] = list(self.smooth)
        if self.region is not None:
            d["region"] = [int(r) for r in self.region]
        if vertex_angles:
            d["vertex_angles"] = [_num(round(float(a), 9)) for a in self.interior_angles()]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Loop":
        ang = d.get("angles")
        return cls(np.array(d["points"], float), None if ang is None else np.array(ang, float),
                   d.get("tags"), d.get("interior"), d.get("smooth"), d.get("region"))


# --------------------------------------------------------------------------

@dataclass(eq=False)
class Chain:
    """Open curve made of the same edge types as a Loop: an internal boundary between two
    regions.  ``regions = (left, right)`` are the region ids on either side (equal for a single
    cut through a ring); ``label_regions`` sets them.  Endpoints must coincide with vertices of
    loops or other chains (conformity), and chains may not cross anything."""
    points: np.ndarray
    angles: np.ndarray | None = None       # per edge (n - 1)
    tags: list | None = None
    interior: list | None = None           # per edge
    smooth: list | None = None             # per vertex (interior vertices matter)
    regions: tuple = (0, 0)
    _edges: list | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, float).reshape(-1, 2)
        n = len(self.points)
        if n < 2:
            raise ValueError("a chain needs at least 2 points")
        m = n - 1
        self.angles = np.zeros(m) if self.angles is None else np.asarray(self.angles, float).reshape(-1)
        if len(self.angles) != m:
            raise ValueError("angles must have one entry per edge")
        if np.any(np.abs(self.angles) > MAX_ARC_DEG + 1e-9):
            raise ValueError("arc angles are limited to +-90 degrees")
        if self.tags is not None:
            self.tags = list(self.tags)
            if len(self.tags) != m:
                raise ValueError("tags must have one entry per edge")
        if self.interior is not None:
            if len(self.interior) != m:
                raise ValueError("interior must have one entry per edge")
            cleaned = []
            for i, x in enumerate(self.interior):
                if x is None or len(x) == 0:
                    cleaned.append(None)
                    continue
                if abs(self.angles[i]) > _EPS:
                    raise ValueError(f"edge {i}: a spline edge must have angle 0")
                cleaned.append(np.asarray(x, float).reshape(-1, 2))
            self.interior = cleaned if any(c is not None for c in cleaned) else None
        if self.smooth is not None:
            self.smooth = [bool(x) for x in self.smooth]
            if len(self.smooth) != n:
                raise ValueError("smooth must have one entry per vertex")
        self.regions = (int(self.regions[0]), int(self.regions[1]))

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def n_edges(self) -> int:
        return len(self.points) - 1

    def kind(self, i: int) -> str:
        if self.interior is not None and self.interior[i] is not None:
            return "spline"
        return "arc" if abs(self.angles[i]) > _EPS else "line"

    def is_smooth(self, v: int) -> bool:
        return bool(self.smooth[v]) if self.smooth is not None and 0 < v < self.n - 1 else False

    def evaluators(self) -> list:
        if self._edges is None:
            m = self.n_edges
            base: list = [None] * m
            for i in range(m):
                k = self.kind(i)
                if k == "line":
                    base[i] = _curves.Line(self.points[i], self.points[i + 1])
                elif k == "arc":
                    base[i] = _curves.Arc(self.points[i], self.points[i + 1], float(self.angles[i]))
            splines = {}
            for i in range(m):
                if self.kind(i) == "spline":
                    pts = np.vstack([self.points[i], self.interior[i], self.points[i + 1]])
                    t0 = self._vertex_tangent(i, base) if self.is_smooth(i) else None
                    t1 = self._vertex_tangent(i + 1, base) if self.is_smooth(i + 1) else None
                    splines[i] = _curves.Spline(pts, t0, t1)
            for i, e in splines.items():
                base[i] = e
            self._edges = base
        return self._edges

    def _vertex_tangent(self, v: int, base: list) -> np.ndarray:
        ein, eout = v - 1, v
        if base[ein] is not None:
            return base[ein].tangent(1.0)
        if base[eout] is not None:
            return base[eout].tangent(0.0)
        return _curves.bessel_tangent(self.interior[ein][-1], self.points[v], self.interior[eout][0])

    def edge(self, i: int):
        return self.evaluators()[i]

    def discretize_edges(self, max_step_deg: float = 5.0) -> list[np.ndarray]:
        return [e.polyline(max_step_deg) for e in self.evaluators()]

    def discretize(self, max_step_deg: float = 5.0) -> np.ndarray:
        """Open polyline from the first to the last point."""
        parts = self.discretize_edges(max_step_deg)
        return np.vstack([q[:-1] for q in parts] + [parts[-1][-1:]])

    def bbox(self):
        b = np.array([e.bbox() for e in self.evaluators()])
        return (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))

    def reversed(self) -> "Chain":
        m = self.n_edges
        interior = None if self.interior is None else [None if x is None else x[::-1].copy() for x in self.interior[::-1]]
        return Chain(self.points[::-1].copy(), -self.angles[::-1], None if self.tags is None else self.tags[::-1],
                     interior, None if self.smooth is None else self.smooth[::-1], (self.regions[1], self.regions[0]))

    def split_edge(self, i: int, t: float) -> "Chain":
        if self.kind(i) == "spline":
            raise NotImplementedError("splitting spline edges is not supported; cut at a control point")
        p = self.edge(i).point(float(t))
        pts = np.insert(self.points, i + 1, p, axis=0)
        ang = np.insert(self.angles, i + 1, self.angles[i] * (1 - t))
        ang[i] = self.angles[i] * t
        tags = None if self.tags is None else self.tags[:i + 1] + [self.tags[i]] + self.tags[i + 1:]
        interior = None if self.interior is None else self.interior[:i + 1] + [None] + self.interior[i + 1:]
        smooth = None if self.smooth is None else self.smooth[:i + 1] + [False] + self.smooth[i + 1:]
        return Chain(pts, ang, tags, interior, smooth, self.regions)

    def _with(self, f) -> "Chain":
        interior = None if self.interior is None else [None if x is None else f(x) for x in self.interior]
        return Chain(f(self.points), self.angles.copy(), self.tags, interior, self.smooth, self.regions)

    def translated(self, dx: float, dy: float) -> "Chain":
        return self._with(lambda P: P + np.array([dx, dy]))

    def scaled(self, s: float) -> "Chain":
        return self._with(lambda P: P * s)

    def to_dict(self) -> dict:
        d = {"points": [[_num(x), _num(y)] for x, y in self.points], "angles": [_num(a) for a in self.angles],
             "regions": [int(self.regions[0]), int(self.regions[1])]}
        if self.tags is not None:
            d["tags"] = list(self.tags)
        if self.interior is not None:
            d["interior"] = [None if x is None else [[_num(a), _num(b)] for a, b in x] for x in self.interior]
        if self.smooth is not None:
            d["smooth"] = list(self.smooth)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        ang = d.get("angles")
        return cls(np.array(d["points"], float), None if ang is None else np.array(ang, float), d.get("tags"),
                   d.get("interior"), d.get("smooth"), tuple(d.get("regions", (0, 0))))


@dataclass
class Geometry:
    """One outer loop (counter-clockwise) followed by hole loops (clockwise), plus optional
    internal chains that split the domain into regions."""
    loops: list[Loop]
    meta: dict = field(default_factory=dict)
    chains: list = field(default_factory=list)

    @property
    def outer(self) -> Loop:
        return self.loops[0]

    @property
    def holes(self) -> list[Loop]:
        return self.loops[1:]

    def bbox(self) -> tuple[float, float, float, float]:
        b = np.array([lp.bbox() for lp in self.loops])
        return (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))

    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox()
        return x1 - x0, y1 - y0

    def area(self) -> float:
        return sum(lp.signed_area() for lp in self.loops)

    def discretize(self, max_step_deg: float = 5.0) -> list[np.ndarray]:
        return [lp.discretize(max_step_deg) for lp in self.loops]

    def vertex_angles(self) -> list[np.ndarray]:
        """Per loop, the interior angle of the domain at every vertex (degrees), arcs
        and splines taken into account; for holes this is the angle outside the hole."""
        return [lp.interior_angles() for lp in self.loops]

    # -- regions ---------------------------------------------------------------
    def regions(self) -> list:
        ids = {lp.edge_region(i) for lp in self.loops for i in range(lp.n)}
        ids |= {r for c in self.chains for r in c.regions}
        return sorted(ids)

    def label_regions(self) -> int:
        from .regions import label_regions
        return label_regions(self)

    def region_geometry(self, r: int) -> "Geometry":
        from .regions import region_geometry
        return region_geometry(self, r)

    def cut(self, p, q, angle: float = 0.0, tol: float | None = None) -> "Geometry":
        from .regions import cut
        return cut(self, p, q, angle, tol)

    def cut_line(self, axis: int, value: float, tol: float | None = None) -> "Geometry":
        from .regions import cut_line
        return cut_line(self, axis, value, tol)

    # -- curve evaluation (t in [0, 1] proportional to arc length) -----------
    def edge(self, li: int, ei: int):
        """Edge evaluator; ``li`` < 0 addresses chain ``-li - 1``."""
        return self.loops[li].edge(ei) if li >= 0 else self.chains[-li - 1].edge(ei)

    def point(self, li: int, ei: int, t):
        return self.edge(li, ei).point(t)

    def tangent(self, li: int, ei: int, t):
        return self.edge(li, ei).tangent(t)

    def normal(self, li: int, ei: int, t):
        """Unit normal pointing into the material (left of the edge)."""
        tg = self.tangent(li, ei, t)
        return np.stack([-tg[..., 1], tg[..., 0]], axis=-1)

    def curvature(self, li: int, ei: int, t):
        return self.edge(li, ei).curvature(t)

    def locate(self, p, tol: float | None = None):
        from .snap import locate
        return locate(self, p, tol)

    def project(self, p):
        from .snap import project
        return project(self, p)

    def split(self, A, B, s: float = 0.5, tol: float | None = None):
        from .snap import split
        return split(self, A, B, s, tol)

    def snap_chain(self, pts, tol: float | None = None):
        from .snap import snap_chain
        return snap_chain(self, pts, tol)

    def snap_mesh(self, nodes, elements, tol: float | None = None):
        from .snap import snap_mesh
        return snap_mesh(self, nodes, elements, tol)

    # -- transforms and I/O -------------------------------------------------
    def translated(self, dx: float, dy: float) -> "Geometry":
        return Geometry([lp.translated(dx, dy) for lp in self.loops], dict(self.meta), [c.translated(dx, dy) for c in self.chains])

    def scaled(self, s: float) -> "Geometry":
        return Geometry([lp.scaled(s) for lp in self.loops], dict(self.meta), [c.scaled(s) for c in self.chains])

    def normalized(self, size: float = 1.0) -> "Geometry":
        """Translate the lower-left bbox corner to the origin and scale the longest side to ``size``."""
        x0, y0, x1, y1 = self.bbox()
        s = size / max(x1 - x0, y1 - y0)
        g = self.translated(-x0, -y0).scaled(s)
        g.meta = dict(self.meta, normalized=True, scale=s)
        return g

    def to_dict(self, vertex_angles: bool = True) -> dict:
        d = {"loops": [lp.to_dict(vertex_angles) for lp in self.loops], "meta": self.meta}
        if self.chains:
            d["chains"] = [c.to_dict() for c in self.chains]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls([Loop.from_dict(l) for l in d["loops"]], d.get("meta", {}),
                   [Chain.from_dict(c) for c in d.get("chains", [])])

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "Geometry":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=1))

    @classmethod
    def load(cls, path: str) -> "Geometry":
        with open(path) as f:
            return cls.from_json(f.read())


def save_jsonl(geoms: Iterable[Geometry], path: str) -> None:
    with open(path, "w") as f:
        for g in geoms:
            f.write(g.to_json() + "\n")


def load_jsonl(path: str) -> list[Geometry]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Geometry.from_json(line))
    return out


# --------------------------------------------------------------------------
# Small constructors

def circle_loop(cx: float, cy: float, r: float, hole: bool = False, tag: str = "circle") -> Loop:
    """Circle as four 90-degree arcs; clockwise (a hole) if ``hole``."""
    pts = np.array([[cx + r, cy], [cx, cy + r], [cx - r, cy], [cx, cy - r]], float)
    lp = Loop(pts, np.full(4, 90.0), [tag] * 4)
    return lp.reversed() if hole else lp


def rect_loop(x0: float, y0: float, x1: float, y1: float, hole: bool = False, tag: str = "rect") -> Loop:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    lp = Loop(pts, np.zeros(4), [tag] * 4)
    return lp.reversed() if hole else lp


def slot_loop(cx: float, cy: float, r: float, half_len: float, axis: str = "x",
              hole: bool = False, tag: str = "slot") -> Loop:
    """Rounded rectangle of width 2r and straight length 2*half_len along ``axis``."""
    L = half_len
    pts = np.array([[cx - L, cy - r], [cx + L, cy - r], [cx + L + r, cy], [cx + L, cy + r],
                    [cx - L, cy + r], [cx - L - r, cy]], float)
    ang = np.array([0.0, 90.0, 90.0, 0.0, 90.0, 90.0])
    if axis == "y":
        pts = np.column_stack([cx + (pts[:, 1] - cy), cy + (pts[:, 0] - cx)])  # swap axes (reflection)
        lp = Loop(pts, -ang, [tag] * 6).reversed()          # reflection flips orientation; restore CCW
    else:
        lp = Loop(pts, ang, [tag] * 6)
    return lp.reversed() if hole else lp
