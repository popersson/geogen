"""`_optimize_node` evaluates its objective through a compiled kernel.

The tangent bookkeeping is hoisted out of the objective and the corner terms
are computed once per evaluation; with numba the evaluation is compiled. The
numpy path is the old arithmetic in the old order, so it must match
`_corner_terms` + `_objective` bit for bit; the numba path sums in another
order, so it must match to rounding and reach the same minimum.
"""

import math

import numpy as np

from geogen.geo2d.smooth import (_corner_terms, _node_objective_numpy, _node_tangent_terms,
                          _objective, _optimize_node)
import importlib
_smooth = importlib.import_module("geogen.geo2d.smooth")  # `geo2d.smooth` the attribute is the dispatcher function


def _patch(rng, curved):
    """Corners around one node of a small quad patch, some sides curved."""
    P = rng.normal(size=(8, 3, 2))
    P[:, 1] = 0.0  # every corner centred on the node at the origin
    dirs = np.full((8, 2, 2), np.nan)
    if curved:
        for k in (1, 4):
            dirs[k, 0] = rng.normal(size=2)
        dirs[6, 1] = rng.normal(size=2)
    sn = np.full(8, math.sin(math.pi / 2))
    return P, sn, (dirs if curved else None)


def _old_objective(P, sn, dirs):
    J = _corner_terms(P, dirs)[0]
    jmin = float(J.min())
    delta = 0.0 if jmin > 0 else math.sqrt(0.1 * float(abs(J).mean()) * (0.1 * float(abs(J).mean()) - jmin))
    return _objective(P, sn, delta, dirs)


def test_numpy_objective_is_the_old_one_bit_for_bit():
    rng = np.random.default_rng(0)
    for curved in (False, True):
        for _ in range(50):
            P, sn, dirs = _patch(rng, curved)
            terms = _node_tangent_terms(len(P), dirs)
            assert _node_objective_numpy(P, sn, *terms) == _old_objective(P, sn, dirs)


def test_compiled_objective_matches_to_rounding():
    rng = np.random.default_rng(1)
    for curved in (False, True):
        for _ in range(50):
            P, sn, dirs = _patch(rng, curved)
            terms = _node_tangent_terms(len(P), dirs)
            a = _node_objective_numpy(P, sn, *terms)
            b = _smooth._node_objective(P, sn, *terms)
            assert (math.isinf(a) and math.isinf(b)) or abs(a - b) <= 1e-12 * max(1.0, abs(a))


def _run_both_paths(P, mask, sn, x0, dirs):
    xs = {}
    for name in ("numpy", "compiled"):
        saved = _smooth._node_objective
        _smooth._node_objective = (_node_objective_numpy if name == "numpy" else saved)
        try:
            xs[name] = _optimize_node(P, mask, sn, x0, 6, dirs)
        finally:
            _smooth._node_objective = saved
    return xs


def test_optimize_node_improves_on_every_path():
    rng = np.random.default_rng(2)
    for curved in (False, True):
        for _ in range(20):
            P, sn, dirs = _patch(rng, curved)
            mask = np.zeros((8, 3), bool)
            mask[:, 1] = True
            x0 = rng.normal(size=2) * 0.3
            Pn = P.copy()
            Pn[mask] = x0
            f0 = _old_objective(Pn, sn, dirs)
            for name, x in _run_both_paths(P, mask, sn, x0, dirs).items():
                Pn[mask] = x
                assert _old_objective(Pn, sn, dirs) <= f0 + 1e-12, name


def test_paths_agree_on_a_well_posed_patch():
    """Four unit quads round one node, the node pushed off centre. The minimum
    is unique and well conditioned, so both paths must find it. (On a random
    tangled patch the finite-difference Newton path can diverge between
    summation orders and land in another local minimum; that is the
    optimiser's fragility, not the kernel's, and is not asserted here.)"""
    ring = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], float)
    # corners centred on the node: (prev, node, next) = (ring[k+2], node, ring[k]) for each quad k,
    # plus the corners at the ring nodes that see the centre node as prev or next
    P, mask = [], []
    for k in range(0, 8, 2):
        nxt, prv = ring[k], ring[(k + 2) % 8]
        P.append([prv, [0, 0], nxt]); mask.append([False, True, False])
        P.append([[0, 0], nxt, ring[(k + 1) % 8]]); mask.append([True, False, False])
        P.append([ring[(k + 1) % 8], prv, [0, 0]]); mask.append([False, False, True])
    P, mask = np.array(P, float), np.array(mask)
    sn = np.full(len(P), math.sin(math.pi / 2))
    x0 = np.array([0.35, -0.2])
    xs = _run_both_paths(P, mask, sn, x0, None)
    assert np.allclose(xs["numpy"], [0.0, 0.0], atol=1e-6)
    assert np.allclose(xs["compiled"], [0.0, 0.0], atol=1e-6)
