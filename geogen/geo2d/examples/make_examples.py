"""Generate the figures referenced in geo2d/README.md.  Run:  python geogen/geo2d/examples/make_examples.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geogen import geo2d  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def row(name, geoms):
    fig = geo2d.plot_gallery(geoms, ncols=len(geoms), cell=2.0)
    fig.savefig(os.path.join(OUT, name), dpi=100)


# 1. complexity ladder
for preset in ("simple", "medium", "complex"):
    row(f"ex_{preset}.png", geo2d.generate_many(1, 6, preset=preset))

# 2. straight lines only, with and without 45 degrees
row("ex_straight.png", geo2d.generate_many(2, 6, preset="straight"))
row("ex_polycube.png", geo2d.generate_many(2, 6, preset="polycube"))

# 3. individual parameters
row("ex_symmetry_x.png", geo2d.generate_many(3, 6, symmetry="x"))
row("ex_arcs_all.png", geo2d.generate_many(4, 6, arcs="all", n_mods=(3, 6)))
row("ex_holes.png", geo2d.generate_many(5, 6, n_holes=(2, 4), n_ops=(1, 3), hole_types={"circle": 0.5, "slot": 0.5}))
row("ex_ratio20.png", geo2d.generate_many(6, 6, ratio=20, aspect=(1.5, 2.5), n_ops=(4, 8), n_mods=(3, 6), n_holes=(1, 4)))
row("ex_star.png", geo2d.generate_many(7, 6, preset="star"))
row("ex_comb.png", geo2d.generate_many(8, 6, preset="comb"))

# 4. plotting options on one shape
g = geo2d.generate(11, preset="medium")
fig = plt.figure(figsize=(12, 3.2))
axes = fig.subplots(1, 4)
geo2d.plot_geometry(g, ax=axes[0])
geo2d.plot_geometry(g, ax=axes[1], show_points=True, show_chords=True)
geo2d.plot_geometry(g, ax=axes[2], show_tags=True, fill=False)
geo2d.plot_geometry(g, ax=axes[3], show_angles=True, lw=0.8)
for ax, t in zip(axes, ["default", "show_points + show_chords", "show_tags, fill=False", "show_angles"]):
    ax.set_title(t, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex_plot_options.png"), dpi=100)

# 5. the same seed under different parameters
fig = plt.figure(figsize=(12, 2.6))
axes = fig.subplots(1, 5)
variants = [dict(), dict(angles=(90,)), dict(arcs="none"), dict(symmetry="xy"), dict(n_holes=(3, 3))]
for ax, kw in zip(axes, variants):
    geo2d.plot_geometry(geo2d.generate(21, **kw), ax=ax)
    ax.set_title(", ".join(f"{k}={v}" for k, v in kw.items()) or "default", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex_same_seed.png"), dpi=100)
print("wrote figures to", OUT)

# 6. airfoils and fluid domains
main = geo2d.airfoil_loop(geo2d.naca4("0012", 80))
flap = geo2d.airfoil_loop(geo2d.naca4("2412", 60), chord=0.4, alpha_deg=25, position=(1.05, -0.08))
slat = geo2d.airfoil_loop(geo2d.naca4("4412", 50), chord=0.25, alpha_deg=-15, position=(-0.12, 0.02), pivot=(1.0, 0.0))
fig = plt.figure(figsize=(12, 3.4))
axes = fig.subplots(1, 3)
geo2d.plot_geometry(geo2d.fluid_domain(-1, -1.5, 3, 1.5, [main]), ax=axes[0], show_points=True)
geo2d.plot_geometry(geo2d.fluid_domain(-1, -1.5, 3, 1.5, [main, flap, slat]), ax=axes[1])
geo2d.plot_geometry(geo2d.Geometry([geo2d.airfoil_loop(geo2d.naca4("0012", 80), hole=False)]), ax=axes[2],
                    show_points=True, show_interior=True, show_angles=True)
for ax, t in zip(axes, ["NACA 0012 in a box (4 control points)", "three-element configuration", "control and interior points, vertex angles"]):
    ax.set_title(t, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex_airfoil.png"), dpi=100)

# 7. meshing, snapping, untangling and smoothing (needs the gmsh binary)
import shutil  # noqa: E402

import numpy as np  # noqa: E402

g = geo2d.generate(5, preset="rounded")
if shutil.which("gmsh"):
    chord = geo2d.gmsh_mesh(g, lc=0.7, exact=False, snap=False)      # Gmsh on the chord polygon
    snapped = geo2d.snap_mesh(g, chord)                              # boundary nodes onto the arcs
    smoothed = geo2d.optimize(snapped, iters=5)                      # untangle + smooth, boundary frozen
    fig = plt.figure(figsize=(13, 4.4))
    axes = fig.subplots(1, 3)
    titles = [f"Gmsh mesh of the chord polygon (min q {chord.quality().min():.2f})",
              f"after snap_mesh: {len(snapped.inverted())} inverted (min q {snapped.quality().min():.2f})",
              f"after optimize: {len(smoothed.inverted())} inverted (min q {smoothed.quality().min():.2f})"]
    for ax, m, t in zip(axes, [chord, snapped, smoothed], titles):
        geo2d.plot_mesh(m, ax=ax, geom=g, irregular=True)
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ex_snap.png"), dpi=100)
    # exact Gmsh mesh with quality colours and irregular nodes (red: degree above target, blue: below)
    exact = geo2d.gmsh_mesh(g, lc=0.5, exact=True)
    fig = plt.figure(figsize=(6.5, 5))
    ax = geo2d.plot_mesh(exact, colorbar=True, irregular=True)
    ax.set_title(f"exact Gmsh quad-dominant mesh, colour = corner quality, {len(exact.irregular_nodes()[0])} irregular nodes", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ex_mesh.png"), dpi=100)
    print(f"mesh demo: snapped {len(snapped.inverted())} inverted -> optimized {len(smoothed.inverted())}, min q {smoothed.quality().min():.3f}")
else:
    print("gmsh not found: skipping the mesh demo")


# 8. internal boundaries: cuts split the domain into regions
g = geo2d.Geometry([geo2d.rect_loop(0, 0, 10, 6), geo2d.rect_loop(3, 2, 5, 4, hole=True)])
one = g.cut((4, 0), (4, 2))                                   # one cut: the ring stays one region, chain (0, 0)
two = one.cut((4, 4), (4, 6))                                 # a second cut: two regions
isl = g.cut((1, 1), (2, 1)).cut((2, 1), (2, 2)).cut((2, 2), (1, 2)).cut((1, 2), (1, 1))   # an island region
arc = geo2d.Geometry([geo2d.circle_loop(0, 0, 4)]).cut((-4, 0), (4, 0), angle=90)         # a curved cut
cl = geo2d.generate(5, preset="rounded").cut_line(0, 4)                                    # a lattice line cut
fig = plt.figure(figsize=(14, 3.4))
for k, (gg, t) in enumerate([(one, "one cut: chain (0, 0)"), (two, "two cuts: two regions"), (isl, "island region"),
                             (arc, "curved cut of a disc"), (cl, "cut_line(0, 4) on a generated shape")]):
    ax = geo2d.plot_geometry(gg, ax=fig.add_subplot(1, 5, k + 1), color_regions=True, show_points=True)
    ax.set_title(t, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ex_regions.png"), dpi=100)
print("wrote regions figure")
