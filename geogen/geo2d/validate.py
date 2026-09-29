"""Validity checks: orientation, self-intersection, hole containment, feature size."""
from __future__ import annotations

import numpy as np

from . import descriptors
from .geometry import MAX_ARC_DEG, Geometry


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _within(P, A, B, tol):
    lo = np.minimum(A, B) - tol
    hi = np.maximum(A, B) + tol
    return np.all((P >= lo) & (P <= hi), axis=-1)


def _seg_hits(p, q, A, B, tol):
    """Does segment p->q meet any of the segments A->B (proper crossing or touching)?"""
    p = p[None]
    q = q[None]
    d1 = _cross(A, B, p)
    d2 = _cross(A, B, q)
    d3 = _cross(p, q, A)
    d4 = _cross(p, q, B)
    proper = (d1 * d2 < -tol * tol) & (d3 * d4 < -tol * tol)
    touch = ((np.abs(d1) <= tol) & _within(p, A, B, tol)) | ((np.abs(d2) <= tol) & _within(q, A, B, tol)) \
        | ((np.abs(d3) <= tol) & _within(A, p, q, tol)) | ((np.abs(d4) <= tol) & _within(B, p, q, tol))
    return proper | touch


def _all_segments(geom: Geometry, max_step_deg: float):
    S, lid, idx, counts = [], [], [], []
    for li, lp in enumerate(geom.loops):
        poly = lp.discretize(max_step_deg)
        m = len(poly)
        S.append(np.stack([poly, np.roll(poly, -1, axis=0)], axis=1))
        lid += [li] * m
        idx += list(range(m))
        counts.append(m)
    return np.vstack(S), np.array(lid), np.array(idx), counts


def self_intersections(geom: Geometry, max_step_deg: float = 22.5, tol: float = 1e-9) -> int:
    """Number of pairs of non-adjacent boundary segments that touch or cross."""
    S, lid, idx, counts = _all_segments(geom, max_step_deg)
    N = len(S)
    count = 0
    for i in range(N - 1):
        A, B = S[i + 1:, 0], S[i + 1:, 1]
        hit = _seg_hits(S[i, 0], S[i, 1], A, B, tol)
        same = lid[i + 1:] == lid[i]
        dj = np.abs(idx[i + 1:] - idx[i])
        adjacent = same & ((dj == 1) | (dj == counts[lid[i]] - 1))
        count += int((hit & ~adjacent).sum())
    return count


def point_in_polygon(pt, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    X, Y = poly[:, 0], poly[:, 1]
    X2, Y2 = np.roll(X, -1), np.roll(Y, -1)
    cond = (Y > y) != (Y2 > y)
    dy = np.where(Y2 - Y == 0, 1e-300, Y2 - Y)
    xint = X + (y - Y) * (X2 - X) / dy
    return bool((cond & (x < xint)).sum() % 2 == 1)


def validate(geom: Geometry, min_feature: float | None = None, tol: float = 1e-7,
             max_step_deg: float = 22.5, strict: bool = False) -> list[str]:
    """Return a list of problems; an empty list means the geometry is valid.
    With ``strict`` the convention that a spline edge turns by at most about 90 degrees
    (limit 100) is enforced."""
    problems: list[str] = []
    if not geom.loops:
        return ["no loops"]
    for li, lp in enumerate(geom.loops):
        if np.any(lp.chord_lengths() < tol):
            problems.append(f"loop {li}: zero-length edge")
        if np.any(np.abs(lp.angles) > MAX_ARC_DEG + 1e-9):
            problems.append(f"loop {li}: arc larger than 90 degrees")
        for ei, e in enumerate(lp.evaluators()):
            if e.kind == "spline":
                turn = e.turning_deg()
                if turn > 180 + 1e-6 or (strict and turn > 100):
                    problems.append(f"loop {li} edge {ei}: spline turns by {turn:.0f} degrees")
    if geom.outer.signed_area() <= 0:
        problems.append("outer loop must be counter-clockwise")
    for li, hole in enumerate(geom.holes, start=1):
        if hole.signed_area() >= 0:
            problems.append(f"loop {li}: hole must be clockwise")
    if problems:
        return problems
    k = self_intersections(geom, max_step_deg)
    if k:
        problems.append(f"{k} pairs of boundary segments touch or cross")
    polys = geom.discretize(max_step_deg)
    for li in range(1, len(polys)):
        if not point_in_polygon(polys[li][0], polys[0]):
            problems.append(f"loop {li}: hole not inside the outer loop")
        for lj in range(1, len(polys)):
            if li != lj and point_in_polygon(polys[li][0], polys[lj]):
                problems.append(f"loop {li}: hole lies inside hole {lj}")
    if min_feature is not None:
        f = descriptors.feature_size(geom)
        if f < min_feature - tol:
            problems.append(f"smallest feature {f:.5g} is below {min_feature}")
    if geom.chains:
        from .regions import validate_regions
        problems += validate_regions(geom)
    return problems


def loops_intersect(polyA: np.ndarray, polyB: np.ndarray, tol: float = 1e-9) -> bool:
    """Do two closed polylines (different loops) touch or cross anywhere?"""
    A1, A2 = polyA, np.roll(polyA, -1, axis=0)
    B1, B2 = polyB, np.roll(polyB, -1, axis=0)
    for i in range(len(A1)):
        if _seg_hits(A1[i], A2[i], B1, B2, tol).any():
            return True
    return False
