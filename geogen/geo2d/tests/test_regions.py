import shutil

import numpy as np
import pytest

from geogen import geo2d
from geogen.geo2d import Geometry, circle_loop, rect_loop, validate


def ring():
    return Geometry([rect_loop(0, 0, 10, 6), rect_loop(3, 2, 5, 4, hole=True)])


def test_single_cut_is_one_region():
    g = ring()
    one = g.cut((4, 0), (4, 2))
    assert len(g.chains) == 0                               # input untouched
    assert one.regions() == [0] and one.chains[0].regions == (0, 0)
    assert validate(one) == []
    sub = one.region_geometry(0)
    assert len(sub.loops) == 2 and len(sub.chains) == 1 and abs(sub.area() - 56) < 1e-9
    assert validate(sub) == []
    assert one.outer.n == 5                                 # the bottom edge was split at (4, 0)
    d = geo2d.describe(one)
    assert d["n_chains"] == 1 and d["n_regions"] == 1


def test_two_cuts_two_regions_and_json():
    two = ring().cut((4, 0), (4, 2)).cut((4, 4), (4, 6))
    assert two.regions() == [0, 1] and all(c.regions == (0, 1) for c in two.chains)
    areas = [two.region_geometry(r).area() for r in two.regions()]
    assert np.allclose(sorted(areas), [22, 34])
    for r in two.regions():
        assert validate(two.region_geometry(r)) == []
    j = Geometry.from_json(two.to_json())
    assert len(j.chains) == 2 and [c.regions for c in j.chains] == [(0, 1), (0, 1)] and validate(j) == []
    assert j.loops[0].region is not None and set(j.loops[0].region.tolist()) == {0, 1}


def test_island_dangling_and_curved_cut():
    g = ring()
    isl = g.cut((1, 1), (2, 1)).cut((2, 1), (2, 2)).cut((2, 2), (1, 2)).cut((1, 2), (1, 1))
    assert isl.regions() == [0, 1] and np.allclose(sorted(isl.region_geometry(r).area() for r in isl.regions()), [1, 55])
    assert validate(isl) == []
    dang = g.cut((1, 1), (2, 1))
    assert dang.regions() == [0] and dang.chains[0].regions == (0, 0) and validate(dang) == []
    assert len(dang.region_geometry(0).chains) == 1
    disc = Geometry([circle_loop(0, 0, 4)])
    arc = disc.cut((-4, 0), (4, 0), angle=90)
    assert arc.regions() == [0, 1] and validate(arc) == []
    areas = [arc.region_geometry(r).area() for r in arc.regions()]
    assert abs(sum(areas) - disc.area()) < 1e-9 and abs(min(areas) - 16.0) < 1e-9   # quarter-circle chord segment


def test_bad_cuts_are_rejected():
    g = ring()
    with pytest.raises(ValueError):
        g.cut((4, 0), (4, 3))                               # ends inside the hole
    with pytest.raises(ValueError):
        g.cut((0, 3), (10, 3))                              # crosses the hole
    with pytest.raises(ValueError):
        g.cut((4, 0), (12, 3))                              # ends outside
    bad = ring()
    bad.chains.append(geo2d.Chain([[4, 0], [4, 2]]))       # not attached: the bottom edge is not split
    assert any("middle of an edge" in p for p in validate(bad))


def test_cut_line_and_split_on_chain():
    s = geo2d.generate(5, preset="rounded")
    cl = s.cut_line(0, 4)
    assert len(cl.chains) >= 1 and validate(cl, min_feature=1.0) == []
    assert abs(sum(cl.region_geometry(r).area() for r in cl.regions()) - s.area()) < 1e-9
    arc = Geometry([circle_loop(0, 0, 4)]).cut((-4, 0), (4, 0), angle=90)
    A, B = arc.chains[0].points
    M = arc.split(A, B)                                     # midpoint of the arc chain, exactly on it
    e = arc.chains[0].edge(0)
    assert abs(np.linalg.norm(M - e.center) - e.radius) < 1e-12
    hits = geo2d.locate(arc, M)
    assert any(li < 0 for li, _, _ in hits)


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh binary not installed")
def test_gmsh_regions_conform():
    two = ring().cut((4, 0), (4, 2)).cut((4, 4), (4, 6))
    m = geo2d.gmsh_mesh(two, lc=0.5, exact=True, snap=False)
    assert set(np.unique(m.data["region"])) == {0, 1}
    assert (np.abs(m.nodes[:, 0] - 4) < 1e-9).sum() >= 8   # nodes on the shared internal curves
    one = ring().cut((4, 0), (4, 2))
    m1 = geo2d.gmsh_mesh(one, lc=0.5, exact=True, snap=False)
    assert set(np.unique(m1.data["region"])) == {0}
    assert ((np.abs(m1.nodes[:, 0] - 4) < 1e-9) & (m1.nodes[:, 1] <= 2 + 1e-9)).sum() >= 4   # embedded cut
