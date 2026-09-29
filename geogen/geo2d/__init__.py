"""geo2d: seeded generator of 2D "generic mechanical part" domains for quad meshing.

Geometry = one counter-clockwise outer loop plus clockwise hole loops; each
edge is straight, a circular arc of at most 90 degrees, or a cubic spline
through interior points (see geometry.py).  snap.py places mesh nodes on the
curved edges; airfoil.py builds airfoils and fluid domains; mesh.py is a
minimal mixed-element mesh container and smooth.py smooths and untangles it.
"""
from .geometry import (Geometry, Loop, Chain, arc_bulge, arc_center, arc_points, arc_radius,
                       bulge_angle, circle_loop, load_jsonl, rect_loop, save_jsonl, slot_loop)
from .descriptors import compute as describe
from .validate import validate
from .generator import PRESETS, Params, generate, generate_many
from .plot import plot_gallery, plot_geometry, plot_mesh
from .mesh import Mesh, read_msh, gmsh_mesh
from .smooth import laplacian, smart_laplacian, optimize, smooth
from .export import to_dxf, to_geo, save_dxf, save_geo
from .snap import locate, project, split, snap_chain, snap_mesh, boundary_edges
from .airfoil import naca4, read_selig, airfoil_loop, spline_loop, fluid_domain, transform_points
from .regions import label_regions, region_geometry, cut, cut_line, validate_regions, n_regions

__all__ = ["Geometry", "Loop", "Chain", "label_regions", "region_geometry", "cut", "cut_line",
           "validate_regions", "n_regions", "arc_bulge", "arc_center", "arc_points", "arc_radius",
           "bulge_angle", "circle_loop", "rect_loop", "slot_loop", "load_jsonl", "save_jsonl",
           "describe", "validate", "PRESETS", "Params", "generate", "generate_many",
           "plot_gallery", "plot_geometry", "plot_mesh", "Mesh", "read_msh", "gmsh_mesh",
           "laplacian", "smart_laplacian", "optimize", "smooth", "to_dxf", "to_geo", "save_dxf", "save_geo",
           "locate", "project", "split", "snap_chain", "snap_mesh", "boundary_edges",
           "naca4", "read_selig", "airfoil_loop", "spline_loop", "fluid_domain", "transform_points"]
