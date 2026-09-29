"""Matplotlib plotting with smooth arcs.

MATLAB style: unless an ``ax`` (or ``fig``) is given, the plot functions reuse
the current figure and clear it first, instead of opening a new window each
time.  Pass ``clear=False`` to draw into the current figure without clearing.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from .geometry import Geometry

TAG_COLORS = {
    "outer": "#1b2430", "chamfer": "#b5322a", "fillet": "#2d6a9f", "notch": "#2d6a9f",
    "bulge": "#2d6a9f", "hole": "#6b7a8a", "slot": "#6b7a8a", "rect": "#6b7a8a", "diamond": "#6b7a8a",
}


def geometry_path(geom: Geometry, step_deg: float = 5.0) -> Path:
    """Compound path: outer loop plus holes (holes are oppositely oriented, so any fill rule works)."""
    verts, codes = [], []
    for lp in geom.loops:
        poly = lp.discretize(step_deg)
        verts.extend(poly.tolist())
        verts.append(poly[0].tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY])
    return Path(np.array(verts), codes)


def _current_axes(clear: bool = True):
    fig = plt.gcf()
    if clear:
        fig.clf()
    return fig.gca()


def plot_geometry(geom: Geometry, ax=None, fill: bool = True, facecolor: str = "#d5dde6",
                  edgecolor: str = "#1b2430", lw: float = 1.2, show_points: bool = False,
                  show_chords: bool = False, show_tags: bool = False, show_angles: bool = False,
                  show_interior: bool = False, color_regions: bool = False, chain_color: str = "#b5322a",
                  step_deg: float = 5.0, margin: float = 0.05, axis_off: bool = True, clear: bool = True):
    """Draw a geometry into ``ax`` (default: the current figure, cleared first).

    Arcs are sampled every ``step_deg`` degrees so they look smooth.  Options:
    ``show_points`` (control points), ``show_chords`` (the chord polygon),
    ``show_tags`` (colour edges by feature tag), ``show_angles`` (label every
    vertex with its interior angle in degrees, 180 = smooth joint),
    ``show_interior`` (the interior points of spline edges), ``color_regions`` (fill every region
    in its own colour; internal chains are drawn in ``chain_color``).
    """
    if ax is None:
        ax = _current_axes(clear)
    path = geometry_path(geom, step_deg)
    if fill and color_regions and geom.chains:
        palette = ["#d5dde6", "#f1d9c9", "#d7e6d0", "#e8dcef", "#f4ead0", "#cfe3ea", "#e6d2d2", "#dde7c9"]
        for r in geom.regions():
            sub = geom.region_geometry(r)
            ax.add_patch(PathPatch(geometry_path(sub, step_deg), facecolor=palette[r % len(palette)],
                                   edgecolor="none", lw=0, zorder=1))
    elif fill:
        ax.add_patch(PathPatch(path, facecolor=facecolor, edgecolor="none", lw=0, zorder=1))
    for ch in geom.chains:
        q = ch.discretize(step_deg)
        ax.plot(q[:, 0], q[:, 1], color=chain_color, lw=lw, zorder=3)
        if show_points:
            ax.plot(ch.points[:, 0], ch.points[:, 1], "o", ms=3, color=chain_color, zorder=4)
    if show_tags:
        for lp in geom.loops:
            for i, pts in enumerate(lp.discretize_edges(step_deg)):
                tag = lp.tags[i] if lp.tags else "outer"
                ax.plot(pts[:, 0], pts[:, 1], color=TAG_COLORS.get(tag, edgecolor), lw=lw,
                        solid_capstyle="round", zorder=3)
    else:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=edgecolor, lw=lw, zorder=3))
    if show_chords:
        for lp in geom.loops:
            pts = np.vstack([lp.points, lp.points[:1]])
            ax.plot(pts[:, 0], pts[:, 1], ls="--", lw=0.6, color="#8a949e", zorder=2)
    if show_points:
        for lp in geom.loops:
            ax.plot(lp.points[:, 0], lp.points[:, 1], "o", ms=3, color="#b5322a", zorder=4)
    if show_interior:
        for lp in geom.loops:
            if lp.interior is not None:
                for x in lp.interior:
                    if x is not None:
                        ax.plot(x[:, 0], x[:, 1], ".", ms=2.5, color="#6b7a8a", zorder=4)
    if show_angles:
        for lp in geom.loops:
            for p, a in zip(lp.points, lp.interior_angles()):
                ax.annotate(f"{a:.0f}", p, fontsize=6, color="#b5322a", ha="center", va="center",
                            xytext=(0, 5), textcoords="offset points", zorder=5)
    x0, y0, x1, y1 = geom.bbox()
    m = margin * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    ax.figure.canvas.draw_idle()
    return ax


def plot_gallery(geoms, ncols: int = 6, cell: float = 2.2, titles=None, fig=None, clear: bool = True, **kw):
    """Grid of thumbnails in ``fig`` (default: the current figure, cleared first); returns the figure.
    Extra keyword arguments go to :func:`plot_geometry`."""
    n = len(geoms)
    nrows = max(1, math.ceil(n / ncols))
    if fig is None:
        fig = plt.gcf()
    if clear:
        fig.clf()
    fig.set_size_inches(cell * ncols, cell * nrows, forward=True)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    kw.pop("clear", None)
    for k, ax in enumerate(axes.flat):
        if k < n:
            plot_geometry(geoms[k], ax=ax, **kw)
            if titles is not None:
                ax.set_title(str(titles[k]), fontsize=7, pad=2)
        else:
            ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.canvas.draw_idle()
    return fig


def plot_mesh(mesh, ax=None, geom=None, color="quality", cmap="viridis", vmin: float = 0.0, vmax: float = 1.0,
              colorbar: bool = False, show_inverted: bool = True, irregular: bool = False, alpha=None,
              node_ids: bool = False, elem_ids: bool = False, lw: float = 0.5, edgecolor: str = "#1b2430",
              facecolor: str = "#d5dde6", margin: float = 0.03, clear: bool = True):
    """Draw a Mesh into ``ax`` (default: the current figure, cleared first).

    ``color``: "quality" (per-element minimum corner quality, 0..1), None (plain), or an
    array with one value per element.  Inverted / non-convex elements are filled red when
    ``show_inverted``.  ``irregular=True`` marks nodes whose degree differs from the
    heuristic desired degree (``mesh.irregular_nodes``): red dot above, blue dot below;
    ``alpha`` is the target element angle (deduced from a single-type mesh if None).
    ``geom`` overlays the exact geometry.
    """
    if ax is None:
        ax = _current_axes(clear)
    verts = [mesh.nodes[el] for el in mesh.elements()]
    if color is None:
        pc = PolyCollection(verts, facecolors=facecolor, edgecolors=edgecolor, linewidths=lw)
    else:
        vals = mesh.quality() if isinstance(color, str) and color == "quality" else np.asarray(color, float)
        pc = PolyCollection(verts, array=vals, cmap=cmap, edgecolors=edgecolor, linewidths=lw)
        pc.set_clim(vmin, vmax)
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    ax.add_collection(pc)
    if show_inverted:
        inv = mesh.inverted()
        if len(inv):
            ax.add_collection(PolyCollection([verts[i] for i in inv], facecolors="#e0574d", edgecolors="#8a1f18",
                                             linewidths=lw, alpha=0.9, zorder=3))
    if geom is not None:
        plot_geometry(geom, ax=ax, fill=False, clear=False, lw=1.0, edgecolor="#2d6a9f")
    if irregular:
        ids, actual, desired = mesh.irregular_nodes(alpha)
        hi, lo = ids[actual > desired], ids[actual < desired]
        ax.plot(mesh.nodes[hi, 0], mesh.nodes[hi, 1], "o", ms=4, color="#d62728", zorder=5, label="degree above target")
        ax.plot(mesh.nodes[lo, 0], mesh.nodes[lo, 1], "o", ms=4, color="#1f77b4", zorder=5, label="degree below target")
    if node_ids:
        for i, (x, y) in enumerate(mesh.nodes):
            ax.annotate(str(i), (x, y), fontsize=6, color="#2d6a9f", ha="center", va="center", zorder=6)
    if elem_ids:
        for i, el in enumerate(mesh.elements()):
            c = mesh.nodes[el].mean(axis=0)
            ax.annotate(str(i), c, fontsize=6, color="#5f6b78", ha="center", va="center", zorder=6)
    x0, y0 = mesh.nodes.min(axis=0)
    x1, y1 = mesh.nodes.max(axis=0)
    m = margin * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.figure.canvas.draw_idle()
    return ax
