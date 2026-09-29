import numpy as np

from geogen.geo2d import describe, validate
from geogen.geo2d.geometry import Geometry, Loop, circle_loop, rect_loop


def test_feature_size_rect_with_hole():
    g = Geometry([rect_loop(0, 0, 10, 6), circle_loop(3, 3, 1, hole=True)])
    d = describe(g)
    assert abs(d["min_ligament"] - 2.0) < 1e-6
    assert abs(d["f_min"] - 1.0) < 1e-9
    assert d["n_holes"] == 1
    assert d["sym_y"] is True and d["sym_x"] is False
    assert abs(d["ratio_bbox"] - 10.0) < 1e-9
    assert d["interior_angles"] == {"180": 4, "90": 4}


def test_ligament_of_notch():
    # a notch of width 1 between x=4 and x=5, depth 3
    g = Geometry([Loop([[0, 0], [10, 0], [10, 6], [5, 6], [5, 3], [4, 3], [4, 6], [0, 6]])])
    d = describe(g)
    assert abs(d["min_ligament"] - 1.0) < 1e-9
    assert d["n_concave"] == 2


def test_validate_good_and_bad():
    good = Geometry([rect_loop(0, 0, 10, 6), circle_loop(3, 3, 1, hole=True)])
    assert validate(good, min_feature=1.0) == []
    bowtie = Geometry([Loop([[0, 0], [2, 2], [2, 0], [0, 2]])])
    assert validate(bowtie) != []
    outside = Geometry([rect_loop(0, 0, 2, 2), circle_loop(5, 5, 0.5, hole=True)])
    assert any("inside" in p for p in validate(outside))
    wrong = Geometry([rect_loop(0, 0, 2, 2, hole=True)])
    assert any("counter-clockwise" in p for p in validate(wrong))
    touching = Geometry([rect_loop(0, 0, 4, 4), rect_loop(0, 1, 2, 3, hole=True)])
    assert any("touch" in p for p in validate(touching))
    thin = Geometry([rect_loop(0, 0, 10, 6), circle_loop(1.5, 3, 1, hole=True)])
    assert any("feature" in p for p in validate(thin, min_feature=1.0))
