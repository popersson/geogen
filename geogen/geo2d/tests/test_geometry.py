import math

import numpy as np

from geogen.geo2d.geometry import (Geometry, Loop, arc_bulge, arc_center, arc_radius, bulge_angle,
                            circle_loop, rect_loop, slot_loop)


def test_circle_area_and_center():
    c = circle_loop(0, 0, 1.0)
    assert abs(c.signed_area() - math.pi) < 1e-12
    assert c.is_ccw()
    h = circle_loop(0, 0, 1.0, hole=True)
    assert abs(h.signed_area() + math.pi) < 1e-12
    np.testing.assert_allclose(arc_center((1, 0), (0, 1), 90), (0, 0), atol=1e-12)
    assert abs(arc_radius((1, 0), (0, 1), 90) - 1) < 1e-12
    # a clockwise quarter circle has its centre on the right of the chord
    np.testing.assert_allclose(arc_center((1, 0), (0, -1), -90), (0, 0), atol=1e-12)


def test_bulge_roundtrip():
    assert abs(bulge_angle(arc_bulge(90)) - 90) < 1e-12
    assert abs(arc_bulge(90) - (math.sqrt(2) - 1)) < 1e-12
    assert abs(arc_bulge(-90) + (math.sqrt(2) - 1)) < 1e-12


def test_reversed_roundtrip():
    loop = slot_loop(0, 0, 1, 2)
    r = loop.reversed()
    assert abs(r.signed_area() + loop.signed_area()) < 1e-12
    rr = r.reversed()
    np.testing.assert_allclose(rr.points, loop.points)
    np.testing.assert_allclose(rr.angles, loop.angles)
    assert rr.tags == loop.tags


def test_slot_area_and_axes():
    sx = slot_loop(0, 0, 1, 2, "x")
    sy = slot_loop(0, 0, 1, 2, "y")
    expected = 4 * 2 + math.pi          # rectangle 4x2 plus two half discs of radius 1
    assert abs(sx.signed_area() - expected) < 1e-12
    assert abs(sy.signed_area() - expected) < 1e-12
    assert sx.is_ccw() and sy.is_ccw()
    np.testing.assert_allclose(sy.bbox(), (-1, -3, 1, 3), atol=1e-12)


def test_discretize_on_circle():
    pts = circle_loop(2, 3, 1.5).discretize(5.0)
    assert len(pts) == 4 * 18
    assert np.allclose(np.hypot(pts[:, 0] - 2, pts[:, 1] - 3), 1.5)


def test_interior_angles():
    assert np.allclose(rect_loop(0, 0, 2, 1).interior_angles(), 90)
    assert np.allclose(circle_loop(0, 0, 1).interior_angles(), 180)
    assert np.allclose(rect_loop(0, 0, 1, 1, hole=True).interior_angles(), 270)
    assert np.allclose(circle_loop(0, 0, 1, hole=True).interior_angles(), 180)


def test_bbox_with_arcs():
    assert np.allclose(circle_loop(1, 1, 1).bbox(), (0, 0, 2, 2))
    # single 90-degree arc from (1,0) to (0,1): bbox is the chord's bbox (no axis extreme inside)
    lp = Loop([[0, 0], [1, 0], [0, 1]], [0, 90, 0])
    assert np.allclose(lp.bbox(), (0, 0, 1, 1))


def test_json_roundtrip(tmp_path):
    g = Geometry([rect_loop(0, 0, 4, 3), circle_loop(2, 1.5, 1, hole=True)], {"seed": 1})
    s = g.to_json()
    assert '"points": [[0, 0]' in s          # integers stay integers
    g2 = Geometry.from_json(s)
    assert g2.to_dict() == g.to_dict()
    p = tmp_path / "g.json"
    g.save(str(p))
    assert Geometry.load(str(p)).to_dict() == g.to_dict()


def test_normalized():
    g = Geometry([rect_loop(2, 2, 12, 6)])
    n = g.normalized()
    assert np.allclose(n.bbox(), (0, 0, 1, 0.4))
