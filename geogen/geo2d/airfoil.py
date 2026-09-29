"""Smooth-curve constructors: NACA 4-digit airfoils, Selig point files, generic
spline loops and fluid domains (box minus bodies).

Airfoil point lists follow the Selig convention: start at the trailing edge,
run over the upper surface to the leading edge and back along the lower
surface; the trailing edge is not repeated.  ``airfoil_loop`` turns such a
list into a 4-edge spline loop whose control point 0 is the trailing edge,
followed (in the loop's direction) by mid surface, leading edge, mid surface;
the trailing edge is a corner, the other three joints are smooth.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Geometry, Loop, rect_loop


def naca4(code: str = "0012", n: int = 80, closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit profile with ``n`` cosine-spaced points per surface, unit chord,
    trailing edge exactly at (1, 0) when ``closed_te``.  Returns (2n - 1, 2) points."""
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n)))
    a4 = 0.1036 if closed_te else 0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - a4 * x ** 4)
    if m > 0 and 0 < p < 1:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.column_stack([xu, yu])[::-1]          # TE -> LE
    lower = np.column_stack([xl, yl])[1:-1]          # LE -> TE, without LE and TE
    pts = np.vstack([upper, lower])
    if closed_te:
        pts[0] = [1.0, float(yc[-1])]
    return pts


def read_selig(path: str) -> np.ndarray:
    """Points from a Selig-format airfoil file (first line is a name)."""
    pts = []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    P = np.array(pts, float)
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    return P


def transform_points(pts, chord: float = 1.0, alpha_deg: float = 0.0, position=(0.0, 0.0),
                     pivot=(0.0, 0.0)) -> np.ndarray:
    """Scale by ``chord``, rotate nose-up by ``alpha_deg`` about ``pivot`` (in scaled
    coordinates), then translate so that the pivot lands at ``position``."""
    P = np.asarray(pts, float) * chord
    piv = np.asarray(pivot, float) * chord
    a = -math.radians(alpha_deg)                       # nose up = clockwise rotation
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (P - piv) @ R.T + np.asarray(position, float)


def airfoil_loop(pts, chord: float = 1.0, alpha_deg: float = 0.0, position=(0.0, 0.0),
                 pivot=(0.0, 0.0), hole: bool = True, tag: str = "airfoil", x_mid: float = 0.3) -> Loop:
    """4-edge spline loop from Selig-ordered points (trailing edge first).  Clockwise
    (a body inside a fluid domain) when ``hole``.  The mid-surface control points sit
    at chord fraction ``x_mid`` (0.3 keeps each edge's turning near 90 degrees, since
    nearly all of it happens around the leading edge)."""
    P = np.asarray(pts, float)
    i_le = int(np.argmin(P[:, 0]))
    i_mu = int(np.argmin(np.abs(P[1:i_le, 0] - x_mid))) + 1
    i_ml = i_le + 1 + int(np.argmin(np.abs(P[i_le + 1:, 0] - x_mid)))
    Q = transform_points(P, chord, alpha_deg, position, pivot)
    ctrl = Q[[0, i_mu, i_le, i_ml]]
    interior = [Q[1:i_mu], Q[i_mu + 1:i_le], Q[i_le + 1:i_ml], Q[i_ml + 1:]]
    lp = Loop(ctrl, np.zeros(4), [tag] * 4, interior, [False, True, True, True])
    return lp.reversed().rolled(3) if hole else lp        # trailing edge stays control point 0


def spline_loop(pts, n_control: int = 4, hole: bool = False, tag: str = "spline") -> Loop:
    """Closed smooth spline through ``pts`` (counter-clockwise order expected) with
    ``n_control`` control points spread evenly along the chord length."""
    P = np.asarray(pts, float)
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    total = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg)])[:-1]          # arc position of every point
    idx = sorted({int(np.argmin(np.abs(cum - k * total / n_control))) for k in range(n_control)})
    if len(idx) < 3:
        raise ValueError("too few distinct control points")
    ctrl, interior = [], []
    for k in range(len(idx)):
        a, b = idx[k], idx[(k + 1) % len(idx)]
        ctrl.append(P[a])
        interior.append(P[a + 1:b] if b > a else np.vstack([P[a + 1:], P[:b]]))
    lp = Loop(np.array(ctrl), np.zeros(len(idx)), [tag] * len(idx), interior, [True] * len(idx))
    return lp.reversed() if hole else lp


def fluid_domain(x0: float, y0: float, x1: float, y1: float, bodies, tag: str = "box") -> Geometry:
    """Rectangle minus bodies; bodies are reversed to clockwise if needed."""
    holes = [b if not b.is_ccw() else b.reversed() for b in bodies]
    return Geometry([rect_loop(x0, y0, x1, y1, tag=tag)] + holes, {"family": "fluid"})
