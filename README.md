# geogen

Basic geometry formats and random geometry generators, in pure Python
(numpy and matplotlib).

- **[`geo2d`](geogen/geo2d/README.md)** — planar domains bounded by straight edges,
  circular arcs and cubic splines: a seeded generator of "generic mechanical
  part" shapes, airfoils and fluid domains, with validity checks, a bounded
  feature-size ratio, Gmsh export and meshing, smoothing and boundary snapping.

```
pip install git+https://github.com/popersson/geogen
```

```python
from geogen import geo2d

g = geo2d.generate(seed=7)                  # a planar Geometry
geo2d.plot_geometry(g)
```

`numba`, if installed (`pip install "geogen[fast] @ git+https://github.com/popersson/geogen"`),
compiles the mesh optimizer's inner loop; without it the same code runs in numpy.
Gmsh export works without Gmsh; meshing through it needs the `gmsh` binary on the path.

Tests: `python -m pytest geogen`. The [`geo2d` README](geogen/geo2d/README.md) has
the full API and example figures.

MIT licensed; see [LICENSE](LICENSE).
