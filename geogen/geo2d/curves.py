"""Edge evaluators: Line, Arc and Spline.

Every evaluator maps a parameter t in [0, 1], proportional to arc length along
the edge, to a point, a unit tangent and a curvature, and can locate the
nearest parameter to a given point.  Arcs and lines are exact; splines are
interpolating C2 cubics in a chord-length parameter with a cached arc-length
table (a few hundred samples per edge).
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-9


# --------------------------------------------------------------------------- arc helpers

def _unit(v: np.ndarray) -> np.ndarray:
    n = math.hypot(float(v[0]), float(v[1]))
    if n < _EPS:
        raise ValueError("zero-length edge")
    return np.array([v[0] / n, v[1] / n])


def _rot(v: np.ndarray, a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])


def arc_center(p0, p1, theta_deg: float) -> np.ndarray:
    """Centre of the arc from p0 to p1 with signed included angle theta:
    C = m + (c/2) cot(theta/2) nL  (m chord midpoint, nL left unit normal of the chord)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    c = math.hypot(float(d[0]), float(d[1]))
    if c < _EPS:
        raise ValueError("zero-length arc chord")
    m = 0.5 * (p0 + p1)
    n_left = np.array([-d[1], d[0]]) / c
    return m + 0.5 * c / math.tan(0.5 * math.radians(theta_deg)) * n_left


def arc_radius(p0, p1, theta_deg: float) -> float:
    c = math.dist(tuple(map(float, p0)), tuple(map(float, p1)))
    return 0.5 * c / math.sin(0.5 * math.radians(abs(theta_deg)))


def arc_bulge(theta_deg: float) -> float:
    """DXF bulge of an arc with included angle theta: tan(theta/4)."""
    return math.tan(math.radians(theta_deg) / 4.0)


def bulge_angle(bulge: float) -> float:
    return math.degrees(4.0 * math.atan(bulge))


def arc_points(p0, p1, theta_deg: float, n: int) -> np.ndarray:
    """n + 1 points from p0 to p1 along the arc, both end points included."""
    return Arc(p0, p1, theta_deg).point(np.linspace(0.0, 1.0, max(1, n) + 1))


def arc_tangents(p0, p1, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    d = _unit(np.asarray(p1, float) - np.asarray(p0, float))
    if abs(theta_deg) < _EPS:
        return d, d
    h = 0.5 * math.radians(theta_deg)
    return _rot(d, -h), _rot(d, h)


def arc_length(p0, p1, theta_deg: float) -> float:
    if abs(theta_deg) < _EPS:
        return math.dist(tuple(map(float, p0)), tuple(map(float, p1)))
    return arc_radius(p0, p1, theta_deg) * math.radians(abs(theta_deg))


def bessel_tangent(a, v, b) -> np.ndarray:
    """Unit tangent at v of the parabola through a, v, b (chord-length parameter)."""
    a, v, b = (np.asarray(x, float) for x in (a, v, b))
    h0, h1 = math.dist(a, v), math.dist(v, b)
    if h0 < _EPS or h1 < _EPS:
        return _unit(b - a)
    d = (h1 / (h0 + h1)) * (v - a) / h0 + (h0 / (h0 + h1)) * (b - v) / h1
    return _unit(d)


def _turning_deg(pts: np.ndarray) -> float:
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(ang)
    da = (da + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.abs(da).sum()))


# --------------------------------------------------------------------------- evaluators

class Line:
    kind = "line"

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        d = self.p1 - self.p0
        self.length = float(math.hypot(d[0], d[1]))
        if self.length < _EPS:
            raise ValueError("zero-length edge")
        self._d = d
        self._u = d / self.length

    def point(self, t):
        t = np.asarray(t, float)
        return self.p0 + t[..., None] * self._d if t.ndim else self.p0 + float(t) * self._d

    def tangent(self, t):
        t = np.asarray(t, float)
        return np.broadcast_to(self._u, t.shape + (2,)).copy() if t.ndim else self._u.copy()

    def curvature(self, t):
        return np.zeros_like(np.asarray(t, float))

    def polyline(self, max_step_deg: float = 5.0) -> np.ndarray:
        return np.vstack([self.p0, self.p1])

    def area_correction(self) -> float:
        return 0.0

    def bbox(self):
        return (min(self.p0[0], self.p1[0]), min(self.p0[1], self.p1[1]),
                max(self.p0[0], self.p1[0]), max(self.p0[1], self.p1[1]))

    def turning_deg(self) -> float:
        return 0.0

    def locate(self, p):
        """Parameter of the nearest point on the edge and the distance to it."""
        p = np.asarray(p, float)
        t = float(np.clip((p - self.p0) @ self._d / self.length ** 2, 0.0, 1.0))
        return t, float(math.dist(p, self.point(t)))


class Arc:
    kind = "arc"

    def __init__(self, p0, p1, theta_deg: float):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.theta = float(theta_deg)
        self.center = arc_center(self.p0, self.p1, self.theta)
        self.radius = arc_radius(self.p0, self.p1, self.theta)
        self._phi0 = math.atan2(self.p0[1] - self.center[1], self.p0[0] - self.center[0])
        self._dphi = math.radians(self.theta)
        self.length = self.radius * abs(self._dphi)

    def point(self, t):
        t = np.asarray(t, float)
        phi = self._phi0 + t * self._dphi
        out = np.stack([self.center[0] + self.radius * np.cos(phi), self.center[1] + self.radius * np.sin(phi)], axis=-1)
        if not t.ndim:
            if abs(float(t)) < _EPS:
                return self.p0.copy()
            if abs(float(t) - 1.0) < _EPS:
                return self.p1.copy()
        return out

    def tangent(self, t):
        t = np.asarray(t, float)
        phi = self._phi0 + t * self._dphi
        s = 1.0 if self._dphi > 0 else -1.0
        return np.stack([-s * np.sin(phi), s * np.cos(phi)], axis=-1)

    def curvature(self, t):
        return np.full(np.asarray(t, float).shape, (1.0 if self._dphi > 0 else -1.0) / self.radius)

    def polyline(self, max_step_deg: float = 5.0) -> np.ndarray:
        n = max(1, math.ceil(abs(self.theta) / max_step_deg - 1e-9))
        pts = self.point(np.linspace(0.0, 1.0, n + 1))
        pts[0], pts[-1] = self.p0, self.p1
        return pts

    def area_correction(self) -> float:
        return 0.5 * self.radius ** 2 * (self._dphi - math.sin(self._dphi))

    def bbox(self):
        pts = [self.p0, self.p1]
        lo, hi = (self._phi0, self._phi0 + self._dphi) if self._dphi > 0 else (self._phi0 + self._dphi, self._phi0)
        for a0 in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            k = math.ceil((lo - a0) / (2 * math.pi) - 1e-12)
            a = a0 + 2 * math.pi * k
            if a <= hi + 1e-12:
                pts.append(self.center + self.radius * np.array([math.cos(a), math.sin(a)]))
        P = np.vstack(pts)
        return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))

    def turning_deg(self) -> float:
        return abs(self.theta)

    def locate(self, p):
        p = np.asarray(p, float)
        phi = math.atan2(p[1] - self.center[1], p[0] - self.center[0])
        a = (phi - self._phi0) % (2 * math.pi) if self._dphi > 0 else (self._phi0 - phi) % (2 * math.pi)
        if a <= abs(self._dphi):
            t = a / abs(self._dphi)
        else:
            t = 0.0 if math.dist(p, self.p0) <= math.dist(p, self.p1) else 1.0
        return float(t), float(math.dist(p, self.point(t)))


def _second_derivatives(u: np.ndarray, y: np.ndarray, d0=None, d1=None) -> np.ndarray:
    """Second derivatives of the interpolating cubic spline (natural or clamped ends)."""
    m = len(u) - 1
    h = np.diff(u)
    A = np.zeros((m + 1, m + 1))
    rhs = np.zeros(m + 1)
    for j in range(1, m):
        A[j, j - 1], A[j, j], A[j, j + 1] = h[j - 1], 2 * (h[j - 1] + h[j]), h[j]
        rhs[j] = 6 * ((y[j + 1] - y[j]) / h[j] - (y[j] - y[j - 1]) / h[j - 1])
    if d0 is None:
        A[0, 0] = 1.0
    else:
        A[0, 0], A[0, 1] = 2 * h[0], h[0]
        rhs[0] = 6 * ((y[1] - y[0]) / h[0] - d0)
    if d1 is None:
        A[m, m] = 1.0
    else:
        A[m, m - 1], A[m, m] = h[m - 1], 2 * h[m - 1]
        rhs[m] = 6 * (d1 - (y[m] - y[m - 1]) / h[m - 1])
    return np.linalg.solve(A, rhs)


class Spline:
    """Interpolating cubic spline through ``pts`` (end points included), chord-length
    parameterized; optional unit tangents at the ends (clamped) give G1 joints."""
    kind = "spline"

    def __init__(self, pts, tan0=None, tan1=None, per_interval: int = 16):
        P = np.asarray(pts, float).reshape(-1, 2)
        if len(P) < 3:
            raise ValueError("a spline edge needs at least one interior point")
        seg = np.hypot(*np.diff(P, axis=0).T)
        if np.any(seg < _EPS):
            raise ValueError("repeated points in spline edge")
        u = np.concatenate([[0.0], np.cumsum(seg)])
        self.P, self.u = P, u
        self.p0, self.p1 = P[0].copy(), P[-1].copy()
        self.M = np.column_stack([
            _second_derivatives(u, P[:, 0], None if tan0 is None else float(tan0[0]), None if tan1 is None else float(tan1[0])),
            _second_derivatives(u, P[:, 1], None if tan0 is None else float(tan0[1]), None if tan1 is None else float(tan1[1]))])
        k = len(u) - 1
        uu = np.concatenate([np.linspace(u[j], u[j + 1], per_interval, endpoint=False) for j in range(k)] + [u[-1:]])
        Q = self._eval(uu)
        S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
        self._uu, self._S, self._Q = uu, S, Q
        self.length = float(S[-1])

    # -- evaluation in the spline parameter u
    def _eval(self, u, deriv: int = 0):
        u = np.asarray(u, float)
        scalar = not u.ndim
        u = np.atleast_1d(u)
        j = np.clip(np.searchsorted(self.u, u, side="right") - 1, 0, len(self.u) - 2)
        h = (self.u[j + 1] - self.u[j])[:, None]
        A = ((self.u[j + 1] - u) / h[:, 0])[:, None]
        B = ((u - self.u[j]) / h[:, 0])[:, None]
        Pj, Pj1, Mj, Mj1 = self.P[j], self.P[j + 1], self.M[j], self.M[j + 1]
        if deriv == 0:
            out = A * Pj + B * Pj1 + ((A ** 3 - A) * Mj + (B ** 3 - B) * Mj1) * h ** 2 / 6.0
        elif deriv == 1:
            out = (Pj1 - Pj) / h - (3 * A ** 2 - 1) / 6.0 * h * Mj + (3 * B ** 2 - 1) / 6.0 * h * Mj1
        else:
            out = A * Mj + B * Mj1
        return out[0] if scalar else out

    def _u_of_t(self, t):
        return np.interp(np.asarray(t, float) * self.length, self._S, self._uu)

    def _t_of_u(self, u):
        return float(np.interp(u, self._uu, self._S) / self.length)

    # -- public API in t
    def point(self, t):
        return self._eval(self._u_of_t(t))

    def tangent(self, t):
        d = self._eval(self._u_of_t(t), 1)
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / n

    def curvature(self, t):
        u = self._u_of_t(t)
        d1, d2 = self._eval(u, 1), self._eval(u, 2)
        num = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        return num / np.linalg.norm(d1, axis=-1) ** 3

    def turning_deg(self) -> float:
        return _turning_deg(self._Q)

    def polyline(self, max_step_deg: float = 5.0) -> np.ndarray:
        n = max(8, math.ceil(self.turning_deg() / max_step_deg), 2 * (len(self.P) - 1))
        pts = self.point(np.linspace(0.0, 1.0, n + 1))
        pts[0], pts[-1] = self.p0, self.p1
        return pts

    def area_correction(self) -> float:
        x, y = self._Q[:, 0], self._Q[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    def bbox(self):
        Q = self._Q
        return (float(Q[:, 0].min()), float(Q[:, 1].min()), float(Q[:, 0].max()), float(Q[:, 1].max()))

    def locate(self, p):
        p = np.asarray(p, float)
        i = int(np.argmin(((self._Q - p) ** 2).sum(axis=1)))
        u = float(self._uu[i])
        for _ in range(12):                      # Newton on (P(u) - p) . P'(u) = 0
            d0, d1, d2 = self._eval(u), self._eval(u, 1), self._eval(u, 2)
            f = float((d0 - p) @ d1)
            fp = float(d1 @ d1 + (d0 - p) @ d2)
            if abs(fp) < 1e-300:
                break
            step = f / fp
            u = min(max(u - step, 0.0), float(self.u[-1]))
            if abs(step) < 1e-14 * max(1.0, self.u[-1]):
                break
        return self._t_of_u(u), float(math.dist(p, self._eval(u)))
