import shutil

import numpy as np
import pytest

from geogen import geo2d
from geogen.geo2d import Mesh, generate, gmsh_mesh, laplacian, optimize, smart_laplacian, snap_mesh


def grid(nx, ny):
    X, Y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = lambda i, j: i * (ny + 1) + j
    quads = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(nx) for j in range(ny)]
    return Mesh(nodes, quads), idx


def test_mesh_basics_and_quality():
    m, idx = grid(5, 4)
    assert m.n_elements == 20 and set(m.by_size()) == {4}
    assert np.allclose(m.quality(), 1) and len(m.inverted()) == 0
    assert np.allclose(m.signed_areas(), 1)
    assert m.degrees()[idx(2, 2)] == 4 and m.degrees()[idx(2, 0)] == 3 and m.degrees()[idx(0, 0)] == 2
    assert np.allclose(np.degrees(m.node_angles())[[idx(2, 2), idx(2, 0), idx(0, 0)]], [360, 180, 90])
    assert np.allclose(m.desired_degrees()[[idx(2, 2), idx(2, 0), idx(0, 0)]], [4, 3, 2])
    assert len(m.irregular_nodes()[0]) == 0
    assert len(m.boundary_edges()) == 2 * (5 + 4) and len(m.boundary_nodes()) == 2 * (5 + 4)
    # clockwise input is re-oriented
    cw = Mesh(m.nodes, [list(e[::-1]) for e in m.elements()])
    assert np.allclose(cw.signed_areas(), 1)
    # mixed mesh with a polygon
    mixed = Mesh([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [3, 0.5], [0.5, 2]],
                 [[0, 1, 2, 3], [1, 4, 5, 2], [4, 6, 5], [3, 2, 7]])
    assert sorted(mixed.sizes()) == [3, 3, 4, 4]
    q = mixed.quality()
    assert np.allclose(q[:2], 1) and 0 < q[2] <= 1.0001 and 0 < q[3] <= 1.0001
    hexagon = Mesh([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)], [[0, 1, 2, 3, 4, 5]])
    assert abs(hexagon.quality()[0] - 1) < 1e-12
    concave = Mesh([[0, 0], [2, 0], [1, 0.5], [2, 2], [0, 2]], [[0, 1, 2, 3, 4]])
    assert list(concave.inverted()) == [0]


def test_irregular_nodes_detected():
    m, idx = grid(4, 4)
    # merge: remove the centre node by turning its 4 quads into one polygon -> neighbours lose degree
    c = idx(2, 2)
    ring = [idx(1, 1), idx(2, 1), idx(3, 1), idx(3, 2), idx(3, 3), idx(2, 3), idx(1, 3), idx(1, 2)]
    keep = [e for e in m.elements() if c not in e]
    poly = Mesh(m.nodes, keep + [ring])
    ids, actual, desired = poly.irregular_nodes(alpha=90)
    assert set(ids) - {c} == {idx(2, 1), idx(3, 2), idx(2, 3), idx(1, 2)} and np.all(actual < desired)   # c is orphaned (degree 0)


def test_io_roundtrips(tmp_path):
    m, _ = grid(3, 2)
    m.data["tag"] = np.arange(m.n_elements)
    m.save(str(tmp_path / "m.json"))
    j = Mesh.load(str(tmp_path / "m.json"))
    assert np.allclose(j.nodes, m.nodes) and j.elements()[3].tolist() == m.elements()[3].tolist()
    m.write_msh(str(tmp_path / "m.msh"))
    r = geo2d.read_msh(str(tmp_path / "m.msh"))
    assert r.n_elements == m.n_elements and np.allclose(r.nodes, m.nodes)
    v = m.to_vtk()
    assert "CELL_TYPES" in v and v.count("\n9\n") >= 1


def test_laplacian_and_smart_recover_grid():
    m, idx = grid(6, 6)
    rng = np.random.default_rng(0)
    pert = m.copy()
    free = ~m.is_boundary()
    pert.nodes[free] += rng.uniform(-0.3, 0.3, size=(free.sum(), 2))
    assert pert.quality().min() < 0.9
    lap = laplacian(pert, iters=200)
    assert np.allclose(lap.nodes, m.nodes, atol=1e-6)
    sm = smart_laplacian(pert, iters=30)
    assert sm.quality().min() > 0.99
    # smart mode never lowers the minimum quality
    assert sm.quality().min() >= pert.quality().min() - 1e-12
    assert np.allclose(sm.nodes[~free], m.nodes[~free])        # boundary frozen


def test_optimize_untangles():
    m, idx = grid(5, 4)
    t = m.copy()
    t.nodes[idx(2, 2)] = [3.7, 2.9]
    assert len(t.inverted()) == 2
    o = optimize(t, iters=8)
    assert len(o.inverted()) == 0 and o.quality().min() > 0.98
    assert np.allclose(o.nodes[idx(2, 2)], [2, 2], atol=2e-3)
    assert np.allclose(o.nodes[m.is_boundary()], m.nodes[m.is_boundary()])


def test_snap_mesh_with_mesh_object():
    g = geo2d.Geometry([geo2d.circle_loop(0, 0, 1)])
    nodes, ring = [np.array([0.0, 0.0])], []
    for i, p0, p1, _ in g.outer.edges():
        for j in range(3):
            nodes.append(p0 + (j / 3) * (p1 - p0))
            ring.append(len(nodes) - 1)
    m = Mesh(np.array(nodes), [[0, ring[i], ring[(i + 1) % 12]] for i in range(12)])
    s = snap_mesh(g, m)
    assert isinstance(s, Mesh) and np.allclose(np.hypot(*s.nodes[1:].T), 1)
    assert (s.data["bnd_loop"] == 0).sum() == 12 and s.data["bnd_loop"][0] == -1
    assert np.isnan(s.data["bnd_t"][0]) and len(s.meta["chains"]) == 4
    t = s.data["bnd_t"][ring]
    assert np.allclose(np.sort(t), np.tile([0, 1 / 3, 2 / 3], 4)[np.argsort(np.tile([0, 1 / 3, 2 / 3], 4))])


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh binary not installed")
def test_gmsh_snap_then_untangle():
    g = generate(5, preset="rounded")
    mesh = gmsh_mesh(g, lc=0.7, exact=False)
    assert mesh.n_elements > 50 and (mesh.data["bnd_loop"] >= 0).sum() == len(mesh.boundary_nodes())
    inv0 = len(mesh.inverted())
    o = optimize(mesh, iters=5)
    assert len(o.inverted()) == 0 and o.quality().min() > 0.3
    assert o.quality().min() >= mesh.quality().min()
    exact = gmsh_mesh(g, lc=0.7, exact=True)
    assert len(exact.inverted()) == 0
    ids, actual, desired = o.irregular_nodes()          # quad-dominant: alpha deduced as 90
    assert len(ids) > 0
    assert inv0 >= 0
