import math
import shutil
import subprocess

import numpy as np
import pytest

from geogen.geo2d import (Geometry, Loop, airfoil_loop, circle_loop, describe, fluid_domain, naca4, rect_loop,
                   snap_mesh, spline_loop, to_geo, validate)
from geogen.geo2d.curves import Arc, Line, Spline


def test_arc_and_line_evaluators():
    a = Arc((1, 0), (0, 1), 90)
    assert abs(a.length - math.pi / 2) < 1e-12
    np.testing.assert_allclose(a.point(0.5), (math.sqrt(0.5), math.sqrt(0.5)))
    np.testing.assert_allclose(a.tangent(0.0), (0, 1), atol=1e-12)
    assert abs(a.curvature(0.3) - 1.0) < 1e-12
    t, d = a.locate((math.sqrt(0.5), math.sqrt(0.5)))
    assert abs(t - 0.5) < 1e-12 and d < 1e-12
    cw = Arc((1, 0), (0, -1), -90)
    assert abs(cw.curvature(0.0) + 1.0) < 1e-12
    ln = Line((0, 0), (4, 0))
    np.testing.assert_allclose(ln.point(0.25), (1, 0))
    t, d = ln.locate((3, 2))
    assert abs(t - 0.75) < 1e-12 and abs(d - 2) < 1e-12


def test_split_on_arc_and_line():
    g = Geometry([rect_loop(0, 0, 10, 6), circle_loop(5, 3, 2, hole=True)])
    A, B = g.loops[1].points[0], g.loops[1].points[1]          # (5,1) -> (3,3): clockwise quarter circle
    np.testing.assert_allclose(A, (5, 1))
    M = g.split(A, B)
    assert abs(math.hypot(M[0] - 5, M[1] - 3) - 2) < 1e-12
    np.testing.assert_allclose(M, (5 - 2 * math.sqrt(0.5), 3 - 2 * math.sqrt(0.5)))
    Q = g.split(A, M, 0.5)                                     # recursive split stays on the circle, equal arcs
    ang = math.degrees(math.atan2(Q[1] - 3, Q[0] - 5))
    assert abs(ang + 112.5) < 1e-9
    np.testing.assert_allclose(g.split((0, 0), (10, 0), 0.3), (3, 0))   # straight edge: linear
    with pytest.raises(ValueError):
        g.split((0, 0), (10, 6))                               # not on a common edge


def test_snap_chain_fractions():
    g = Geometry([circle_loop(0, 0, 1)])
    A, B = g.outer.points[0], g.outer.points[1]                # (1,0) -> (0,1)
    chord = np.array([A, A + 0.25 * (B - A), A + 0.75 * (B - A), B])
    out = g.snap_chain(chord)
    r = np.hypot(out[:, 0], out[:, 1])
    assert np.allclose(r, 1)
    angs = np.degrees(np.arctan2(out[:, 1], out[:, 0]))
    np.testing.assert_allclose(angs, [0, 22.5, 67.5, 90], atol=1e-9)


def _fan_mesh_of_circle(k):
    """Disc: control points plus k-1 extra chord nodes per edge, centre node, triangle fan."""
    g = Geometry([circle_loop(0, 0, 1)])
    nodes = [np.array([0.0, 0.0])]
    ring = []
    for i, p0, p1, _ in g.outer.edges():
        for j in range(k):
            nodes.append(p0 + (j / k) * (p1 - p0))
            ring.append(len(nodes) - 1)
    elements = [[0, ring[i], ring[(i + 1) % len(ring)]] for i in range(len(ring))]
    return g, np.array(nodes), elements


def test_snap_mesh_disc():
    g, nodes, elements = _fan_mesh_of_circle(4)
    new, chains = snap_mesh(g, nodes, elements)
    r = np.hypot(new[1:, 0], new[1:, 1])
    assert np.allclose(r, 1)
    assert len(chains) == 4 and all(len(c[2]) == 5 for c in chains)
    np.testing.assert_allclose(new[0], (0, 0))                 # interior node untouched
    # snapping a snapped mesh is a no-op
    again, _ = snap_mesh(g, new, elements)
    np.testing.assert_allclose(again, new)


def test_spline_circle_accuracy_and_smoothness():
    th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    pts = np.column_stack([np.cos(th), np.sin(th)])
    lp = spline_loop(pts, n_control=4)
    assert lp.kinds() == ["spline"] * 4 and lp.smooth == [True] * 4
    q = lp.discretize(2.0)
    assert np.abs(np.hypot(q[:, 0], q[:, 1]) - 1).max() < 2e-4
    assert abs(lp.signed_area() - math.pi) < 1e-3
    assert np.allclose(lp.interior_angles(), 180, atol=1e-9)   # shared tangents make the joints G1
    e = lp.edge(0)
    assert abs(e.length - math.pi / 2) < 1e-3
    assert np.allclose(e.curvature(np.linspace(0, 1, 7)), 1, atol=2e-2)
    g = Geometry([lp])
    t, d = e.locate(e.point(0.37))
    assert abs(t - 0.37) < 1e-9 and d < 1e-12
    M = g.split(lp.points[0], lp.points[1])
    assert abs(math.hypot(*M) - 1) < 2e-4
    assert validate(g, strict=True) == []
    rev = lp.reversed()
    assert abs(rev.signed_area() + math.pi) < 1e-3 and np.allclose(rev.interior_angles(), 180, atol=1e-9)


def test_naca0012_domain():
    pts = naca4("0012", n=80)
    assert pts.shape == (158, 2)                             # 2n - 2: TE and LE not repeated
    np.testing.assert_allclose(pts[0], (1, 0))                 # sharp trailing edge exactly at (1, 0)
    assert abs(pts[:, 1]).max() < 0.0605 and abs(pts[:, 1]).max() > 0.059
    lp = airfoil_loop(pts, hole=True)
    assert not lp.is_ccw() and lp.kinds() == ["spline"] * 4
    np.testing.assert_allclose(lp.points[0], (1, 0))          # trailing edge stays control point 0
    va = lp.interior_angles()
    assert 340 < va[0] < 345                                  # reflex trailing-edge corner seen from the fluid (360 - 16.5)
    assert np.allclose(va[1:], 180, atol=1e-9)
    solid = airfoil_loop(pts, hole=False)
    assert 15 < solid.interior_angles()[0] < 20 and solid.is_ccw()
    g = fluid_domain(-1, -1.5, 3, 1.5, [lp])
    assert validate(g, strict=True) == []
    d = describe(g)
    assert d["n_splines"] == 4 and d["n_holes"] == 1
    j = Geometry.from_json(g.to_json())
    assert j.to_dict() == g.to_dict()
    # two-element configuration with a rotated flap
    flap = airfoil_loop(naca4("2412", 60), chord=0.4, alpha_deg=25, position=(1.05, -0.08))
    g2 = fluid_domain(-1, -1.5, 3, 1.5, [lp, flap])
    assert validate(g2, strict=True) == []


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh binary not installed")
def test_gmsh_meshes_airfoil(tmp_path):
    g = fluid_domain(-1, -1.5, 3, 1.5, [airfoil_loop(naca4("0012", 80))])
    geo = tmp_path / "af.geo"
    geo.write_text(to_geo(g, lc=0.15))
    out = tmp_path / "af.msh"
    res = subprocess.run(["gmsh", "-2", str(geo), "-o", str(out), "-v", "2"], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert out.exists()
