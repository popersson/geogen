import pytest

from geogen.geo2d import PRESETS, Params, generate, generate_many, validate


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_valid(preset):
    for s in range(12):
        g = generate(s, preset=preset)
        assert validate(g, min_feature=1.0, tol=1e-3) == []
        d = g.meta["descriptors"]
        assert d["ratio_bbox"] <= g.meta["params"]["ratio"] + 1e-6
        assert d["f_min"] >= 1.0 - 1e-3


def test_deterministic():
    assert generate(7).to_json() == generate(7).to_json()
    assert generate(7).to_json() != generate(8).to_json()


def test_symmetry_forced():
    for s in range(6):
        assert generate(s, symmetry="x").meta["descriptors"]["sym_x"]
        assert generate(s, symmetry="y").meta["descriptors"]["sym_y"]
        d = generate(s, symmetry="xy").meta["descriptors"]
        assert d["sym_x"] and d["sym_y"]


def test_angle_and_arc_switches():
    for s in range(10):
        d = generate(s, angles=(90,), arcs="none").meta["descriptors"]
        assert d["n_arcs"] == 0
        assert set(d["interior_angles"]) <= {"90", "270"}


def test_generate_many_and_params():
    gs = generate_many(0, 10, Params(ratio=12, n_holes=(1, 2)))
    assert len(gs) == 10
    # n_holes counts hole *features*; a pattern or a mirrored pair adds several loops,
    # and placement can fail when the shape is too small, so this is a soft check
    assert sum(1 for g in gs if g.meta["descriptors"]["n_holes"] >= 1) >= 8
    assert all(len(g.meta["program"]) >= 2 for g in gs)
    assert all(g.meta["descriptors"]["ratio_bbox"] <= 12 + 1e-6 for g in gs)
