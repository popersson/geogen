import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from geogen.geo2d import Geometry, generate, plot_gallery, plot_geometry, to_geo  # noqa: E402
from geogen.geo2d.geometry import circle_loop, rect_loop, slot_loop  # noqa: E402


def test_vertex_angles_with_arcs():
    g = Geometry([rect_loop(0, 0, 10, 6), circle_loop(3, 3, 1, hole=True), slot_loop(7, 3, 1, 1, hole=True)])
    va = g.vertex_angles()
    assert np.allclose(va[0], 90)            # rectangle corners
    assert np.allclose(va[1], 180)           # circle: smooth at all four control points
    assert np.allclose(va[2], 180)           # slot: arcs are tangent to the straight edges, smooth everywhere
    d = json.loads(g.to_json())
    assert d["loops"][1]["vertex_angles"] == [180, 180, 180, 180]
    assert d["loops"][0]["vertex_angles"] == [90, 90, 90, 90]
    assert Geometry.from_dict(d).to_dict() == d          # derived field is ignored on load and recomputed


def test_geo_export_has_angles():
    s = to_geo(generate(2, preset="rounded"))
    assert "// angle" in s and "// loop 1" in s or "// loop 0" in s


def test_straight_presets_have_no_arcs():
    for seed in range(8):
        for preset in ("straight", "polycube"):
            d = generate(seed, preset=preset).meta["descriptors"]
            assert d["n_arcs"] == 0


def test_plot_reuses_current_figure():
    fig = plt.figure()
    g = generate(0)
    ax = plot_geometry(g, show_angles=True, show_points=True)
    assert ax.figure is fig and plt.gcf() is fig
    n_before = len(plt.get_fignums())
    plot_geometry(generate(1))
    assert len(plt.get_fignums()) == n_before        # no new window
    f2 = plot_gallery([g, generate(1)], ncols=2)
    assert f2 is fig
    plt.close("all")
