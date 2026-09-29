"""The smoother can be told that a side is curved.

Corner quality is built from the two sides meeting at a corner. On a curved
edge the element's side follows the arc, so the direction it leaves the corner
is the TANGENT; the chord describes an element nobody has. Lengths stay on the
chord, so aspect ratio is measured on the straight-sided polygonal patch and
only the Jacobian moves.

`tangents` maps (node, neighbour) -> unit direction, for curved sides only.
"""

import numpy as np
import pytest

from geogen import geo2d
from geogen.geo2d.smooth import _corner_terms, optimize, smart_laplacian


def _unit_square_mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                      [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    elements = [[0, 1, 4, 3], [1, 2, 5, 4]]
    return nodes, elements


def test_no_tangents_is_the_old_behaviour_exactly():
    nodes, elements = _unit_square_mesh()
    nodes[4] += [0.15, -0.2]
    mesh = geo2d.Mesh(nodes, elements)
    fixed = mesh.is_boundary()
    assert np.array_equal(np.asarray(optimize(mesh, iters=10, fixed=fixed).nodes),
                          np.asarray(optimize(mesh, iters=10, fixed=fixed,
                                              tangents=None).nodes))
    assert np.array_equal(np.asarray(smart_laplacian(mesh, iters=5, fixed=fixed).nodes),
                          np.asarray(smart_laplacian(mesh, iters=5, fixed=fixed,
                                                     tangents=None).nodes))


def test_an_empty_table_is_also_a_no_op():
    nodes, elements = _unit_square_mesh()
    mesh = geo2d.Mesh(nodes, elements)
    fixed = mesh.is_boundary()
    assert np.array_equal(np.asarray(optimize(mesh, iters=8, fixed=fixed).nodes),
                          np.asarray(optimize(mesh, iters=8, fixed=fixed,
                                              tangents={}).nodes))


def test_direction_is_retargeted_and_length_is_kept():
    """Only J may move; |a|^2 + |b|^2 describes the straight-sided patch."""
    P = np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]])      # prev, node, next
    J_chord, L2_chord = _corner_terms(P)
    turned = np.array([[[np.cos(0.4), np.sin(0.4)], [np.nan, np.nan]]])
    J_tan, L2_tan = _corner_terms(P, turned)
    assert L2_tan == pytest.approx(L2_chord)
    assert not np.allclose(J_tan, J_chord)
    # the retargeted side keeps its length: |a| = 1 either way
    assert J_tan[0] == pytest.approx(-np.sin(0.4 - np.pi / 2) * 1.0 * 1.0, abs=1e-12)


def test_a_curved_side_moves_where_the_smoother_puts_a_free_node():
    """A quarter-circle side: the tangent is 45 degrees off its chord."""
    angles = np.linspace(0.0, np.pi / 2, 3)
    arc = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    outer = 2.0 * arc
    nodes = np.vstack([arc, outer])
    elements = [[0, 1, 4, 3], [1, 2, 5, 4]]
    free = 4                                    # the middle node of the outer ring
    nodes[free] += [0.25, -0.2]
    mesh = geo2d.Mesh(nodes, elements)
    fixed = np.ones(len(nodes), bool); fixed[free] = False
    tangents = {}
    for ring, base in ((arc, 0), (outer, 3)):
        for j in range(3):
            n = base + j
            t = np.array([-nodes[n][1], nodes[n][0]]); t /= np.linalg.norm(t)
            for jj in (j - 1, j + 1):
                if 0 <= jj < 3:
                    tangents[(n, base + jj)] = (1.0 if jj > j else -1.0) * t
    chord = np.asarray(optimize(mesh, iters=40, fixed=fixed).nodes)[free]
    tangent = np.asarray(optimize(mesh, iters=40, fixed=fixed,
                                  tangents=tangents).nodes)[free]
    assert np.linalg.norm(chord - tangent) > 1e-3


def test_the_acceptance_predicate_sees_tangents():
    """`smart_laplacian` accepts on min corner quality, so that must use them.

    Its TARGET is the Laplacian mean and does not depend on tangents, so the
    smoothed result only differs when the verdict flips. What must be true
    unconditionally is that the predicate itself reads them, which is this.
    """
    from geogen.geo2d.smooth import _min_quality
    P = np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]])
    sn = np.array([1.0])
    plain = _min_quality(P, sn)
    turned = np.array([[[np.cos(0.5), np.sin(0.5)], [np.nan, np.nan]]])
    assert _min_quality(P, sn, turned) != pytest.approx(plain)


def test_a_curved_boundary_changes_what_gets_accepted():
    """A verdict that flips: refused on chords, accepted on tangents.

    `smart_laplacian` accepts a move when the minimum corner quality does not
    drop, so a metric that misreads the corner accepts and refuses different
    moves. Found by searching a two-parameter family rather than asserted by
    construction -- whether a given pair flips depends on the whole corner set,
    and guessing it is how a test passes for the wrong reason.
    """
    from geogen.geo2d.smooth import _min_quality
    angles = np.linspace(0.0, np.pi * 0.9, 3)
    arc = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    nodes = np.vstack([arc, 2.0 * arc])
    tri = np.array([[0, 1, 4]])                 # prev, node (on the arc), next
    sn = np.array([1.0])
    t = np.array([-nodes[1][1], nodes[1][0]]); t /= np.linalg.norm(t)
    dirs = np.array([[[np.nan, np.nan], [-t[0], -t[1]]]])

    def verdicts(dx_from, dx_to):
        before, after = nodes.copy(), nodes.copy()
        before[4] += [dx_from, 0.0]
        after[4] += [dx_to, 0.0]
        chord = _min_quality(after[tri], sn) >= _min_quality(before[tri], sn)
        tangent = (_min_quality(after[tri], sn, dirs)
                   >= _min_quality(before[tri], sn, dirs))
        return chord, tangent

    grid = np.linspace(-0.8, 0.8, 41)
    assert any(verdicts(a, b)[0] != verdicts(a, b)[1] for a in grid for b in grid), \
        "no move was judged differently, so the acceptance test is not using them"
    # the specific pair the search finds, pinned so a regression is legible
    chord, tangent = verdicts(-0.6, 0.8)
    assert chord is False and tangent is True
