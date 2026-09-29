"""A corner-modifier combination that consumes the outline must be rejected, not raised.

Seed 11919 of the "straight" preset used to escape the generator with
"a loop needs at least 3 points": two unit chamfers on opposite corners of a
unit-wide rectangle left fewer than three outline points inside `_outer_ok`,
whose job is exactly to reject such candidates. Found by an RL env worker that
died on the exception after about 70,000 draws.
"""
from geogen import geo2d


def test_seed_11919_straight_generates():
    g = geo2d.generate(11919, preset="straight", n_holes=0)
    assert g.outer.n >= 3
    assert geo2d.validate(g) == []


def test_straight_seeds_never_raise():
    for seed in range(11900, 11950):
        g = geo2d.generate(seed, preset="straight", n_holes=0)
        assert g.outer.n >= 3
