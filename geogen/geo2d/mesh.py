"""Minimal 2D mesh container for mixed elements, including arbitrary polygons.

``Mesh(nodes, elements)`` stores the connectivity as one flat index array plus
offsets (the VTK layout), so triangles, quads and polygons live in one array
pair.  Topological queries (edges, boundary, adjacency) are cached; geometric
quantities (areas, corner Jacobians, quality) are recomputed from the current
node positions on every call, so smoothers can move nodes freely.

Quality is corner based so that one definition covers every element type: at a
corner with edge vectors a (to the next node) and b (to the previous node) the
signed Jacobian is J = a x b and the corner quality is 2J / (|a|^2 + |b|^2)
divided by sin of the regular n-gon corner angle, so a regular element of any
size scores 1, a degenerate corner 0 and an inverted corner is negative.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np


class Mesh:
    def __init__(self, nodes, elements, data=None, meta=None, orient: bool = True):
        self.nodes = np.array(nodes, float).reshape(-1, 2)
        if isinstance(elements, tuple) and len(elements) == 2:
            self.conn = np.asarray(elements[0], np.int64).ravel().copy()
            self.offsets = np.asarray(elements[1], np.int64).ravel().copy()
        else:
            els = [np.asarray(e, np.int64).ravel() for e in elements]
            sizes = np.array([len(e) for e in els], np.int64)
            self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            self.conn = np.concatenate(els).astype(np.int64) if els else np.zeros(0, np.int64)
        if len(self.conn) and (self.conn.min() < 0 or self.conn.max() >= len(self.nodes)):
            raise ValueError("element refers to a node index out of range")
        if np.any(np.diff(self.offsets) < 3):
            raise ValueError("every element needs at least 3 nodes")
        self.data = {k: np.asarray(v) for k, v in (data or {}).items()}
        self.meta = dict(meta or {})
        self._cache: dict = {}
        if orient:
            self.orient()

    # ------------------------------------------------------------------ basics
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.offsets) - 1

    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def element(self, i: int) -> np.ndarray:
        return self.conn[self.offsets[i]:self.offsets[i + 1]]

    def elements(self) -> list[np.ndarray]:
        return [self.element(i) for i in range(self.n_elements)]

    def copy(self) -> "Mesh":
        m = Mesh(self.nodes.copy(), (self.conn.copy(), self.offsets.copy()),
                 {k: v.copy() for k, v in self.data.items()}, dict(self.meta), orient=False)
        m._cache = dict(self._cache)
        return m

    def by_size(self) -> dict:
        """{k: (element ids, (M_k, k) connectivity)} for vectorised per-type work."""
        if "by_size" not in self._cache:
            sizes = self.sizes()
            out = {}
            for k in np.unique(sizes):
                idx = np.nonzero(sizes == k)[0]
                out[int(k)] = (idx, self.conn[self.offsets[idx][:, None] + np.arange(k)[None, :]])
            self._cache["by_size"] = out
        return self._cache["by_size"]

    # ------------------------------------------------------------------ geometry per element
    def signed_areas(self) -> np.ndarray:
        A = np.zeros(self.n_elements)
        for k, (idx, conn) in self.by_size().items():
            P = self.nodes[conn]
            x, y = P[..., 0], P[..., 1]
            A[idx] = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
        return A

    def orient(self) -> int:
        """Reverse clockwise elements so that all are counter-clockwise; returns how many."""
        flip = np.nonzero(self.signed_areas() < 0)[0]
        for i in flip:
            self.conn[self.offsets[i]:self.offsets[i + 1]] = self.conn[self.offsets[i]:self.offsets[i + 1]][::-1]
        if len(flip):
            self._cache = {}
        return len(flip)

    def corners(self) -> dict:
        """Flat corner arrays (one entry per element corner): elem, node, prev, next, n (element size)."""
        if "corners" not in self._cache:
            elem, node, prev, nxt, n = [], [], [], [], []
            for k, (idx, conn) in self.by_size().items():
                elem.append(np.repeat(idx, k))
                node.append(conn.ravel())
                prev.append(np.roll(conn, 1, axis=1).ravel())
                nxt.append(np.roll(conn, -1, axis=1).ravel())
                n.append(np.full(conn.size, k))
            self._cache["corners"] = {key: np.concatenate(v) if v else np.zeros(0, np.int64)
                                      for key, v in zip(("elem", "node", "prev", "next", "n"), (elem, node, prev, nxt, n))}
        return self._cache["corners"]

    def corner_geometry(self):
        """(J, |a|^2, |b|^2, a.b) for every corner, a = next - node, b = prev - node."""
        c = self.corners()
        X = self.nodes
        a = X[c["next"]] - X[c["node"]]
        b = X[c["prev"]] - X[c["node"]]
        J = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return J, (a ** 2).sum(axis=1), (b ** 2).sum(axis=1), (a * b).sum(axis=1)

    def corner_jacobians(self) -> np.ndarray:
        return self.corner_geometry()[0]

    def corner_angles(self) -> np.ndarray:
        """Interior angle of the element at each corner, in (0, 2pi); reflex for non-convex corners."""
        J, _, _, dot = self.corner_geometry()
        return np.arctan2(J, dot) % (2 * math.pi)

    def corner_quality(self) -> np.ndarray:
        J, la2, lb2, _ = self.corner_geometry()
        n = self.corners()["n"]
        ideal = np.sin(math.pi * (n - 2) / n)
        return 2 * J / (la2 + lb2) / ideal

    def quality(self) -> np.ndarray:
        """Per element: minimum corner quality (1 regular, 0 degenerate, negative inverted)."""
        q = np.full(self.n_elements, np.inf)
        np.minimum.at(q, self.corners()["elem"], self.corner_quality())
        return q

    def inverted(self) -> np.ndarray:
        """Ids of elements with a non-positive corner Jacobian (inverted or non-convex)."""
        jmin = np.full(self.n_elements, np.inf)
        np.minimum.at(jmin, self.corners()["elem"], self.corner_jacobians())
        return np.nonzero(jmin <= 0)[0]

    def angles_stats(self) -> dict:
        ang = np.degrees(self.corner_angles())
        return {"min_angle": float(ang.min()), "max_angle": float(ang.max())}

    # ------------------------------------------------------------------ topology
    def _edge_tables(self):
        if "edges" not in self._cache:
            c = self.corners()
            directed = np.column_stack([c["node"], c["next"]])
            und = np.sort(directed, axis=1)
            uniq, inv, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
            self._cache["edges"] = (uniq, counts, directed, inv.ravel())
        return self._cache["edges"]

    def edges(self) -> np.ndarray:
        """Unique undirected edges (E, 2), sorted node pairs."""
        return self._edge_tables()[0]

    def edge_counts(self) -> np.ndarray:
        return self._edge_tables()[1]

    def boundary_edges(self) -> np.ndarray:
        """Directed boundary edges (B, 2) as they appear in their element: material on the left."""
        uniq, counts, directed, inv = self._edge_tables()
        return directed[counts[inv] == 1]

    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.boundary_edges())

    def is_boundary(self) -> np.ndarray:
        m = np.zeros(self.n_nodes, bool)
        m[self.boundary_nodes()] = True
        return m

    def degrees(self) -> np.ndarray:
        """Number of edges at each node."""
        e = self.edges()
        return np.bincount(e.ravel(), minlength=self.n_nodes)

    def node_elements(self):
        """CSR (offsets, element ids): elements adjacent to each node."""
        if "node_elements" not in self._cache:
            c = self.corners()
            order = np.argsort(c["node"], kind="stable")
            counts = np.bincount(c["node"], minlength=self.n_nodes)
            self._cache["node_elements"] = (np.concatenate([[0], np.cumsum(counts)]), c["elem"][order])
        return self._cache["node_elements"]

    def elements_of_node(self, i: int) -> np.ndarray:
        off, el = self.node_elements()
        return el[off[i]:off[i + 1]]

    def node_neighbors(self):
        """CSR (offsets, node ids): edge neighbours of each node."""
        if "node_neighbors" not in self._cache:
            e = self.edges()
            both = np.concatenate([e, e[:, ::-1]])
            order = np.argsort(both[:, 0], kind="stable")
            counts = np.bincount(both[:, 0], minlength=self.n_nodes)
            self._cache["node_neighbors"] = (np.concatenate([[0], np.cumsum(counts)]), both[order, 1])
        return self._cache["node_neighbors"]

    def neighbors_of_node(self, i: int) -> np.ndarray:
        off, nb = self.node_neighbors()
        return nb[off[i]:off[i + 1]]

    # ------------------------------------------------------------------ node angles and irregular nodes
    def node_angles(self) -> np.ndarray:
        """Sum of the element angles at each node: 2pi inside, the domain angle on the boundary."""
        out = np.zeros(self.n_nodes)
        np.add.at(out, self.corners()["node"], self.corner_angles())
        return out

    def element_angle(self) -> float:
        """Target corner angle in degrees of the dominant element type (90 for quads, 60 for
        triangles); for a mixed mesh the most common size decides, so pass ``alpha`` explicitly
        when that is not what you mean."""
        ks, counts = np.unique(self.sizes(), return_counts=True)
        k = int(ks[np.argmax(counts)])
        return 180.0 * (k - 2) / k

    def desired_degrees(self, alpha: float | None = None, round_boundary: bool = True) -> np.ndarray:
        """Heuristic desired node degree from the target element angle ``alpha``: 360/alpha
        inside (4 for quads), and theta/alpha + 1 on the boundary where theta is the domain
        angle at the node (3 on a straight quad-mesh boundary, 2 at a convex corner)."""
        alpha = self.element_angle() if alpha is None else float(alpha)
        d0 = np.full(self.n_nodes, 360.0 / alpha)
        bdry = self.boundary_nodes()
        if len(bdry):
            ratio = np.degrees(self.node_angles()[bdry]) / alpha
            deg = np.floor(ratio + 0.5) + 1.0 if round_boundary else ratio + 1.0
            d0[bdry] = np.maximum(deg, 2.0)
        return d0

    def irregular_nodes(self, alpha: float | None = None, round_boundary: bool = True):
        """(ids, actual degree, desired degree) of nodes whose degree differs from the heuristic target."""
        actual = self.degrees().astype(float)
        desired = self.desired_degrees(alpha, round_boundary)
        bad = np.abs(actual - desired) > (0.5 if not round_boundary else 1e-9)
        ids = np.nonzero(bad)[0]
        return ids, actual[ids], desired[ids]

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict:
        return {"nodes": self.nodes.tolist(), "elements": [e.tolist() for e in self.elements()],
                "data": {k: v.tolist() for k, v in self.data.items()}, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Mesh":
        return cls(d["nodes"], d["elements"], d.get("data"), d.get("meta"), orient=False)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "Mesh":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_msh(self) -> str:
        """Gmsh MSH version 2 (triangles and quads only)."""
        sizes = self.sizes()
        if np.any((sizes != 3) & (sizes != 4)):
            raise ValueError("MSH2 has no polygon element type; use VTK for polygonal meshes")
        tags = self.data.get("tag", np.ones(self.n_elements, int))
        out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(self.n_nodes)]
        out += [f"{i + 1} {x:.16g} {y:.16g} 0" for i, (x, y) in enumerate(self.nodes)]
        out += ["$EndNodes", "$Elements", str(self.n_elements)]
        for i, el in enumerate(self.elements()):
            typ = 2 if len(el) == 3 else 3
            out.append(f"{i + 1} {typ} 2 {int(tags[i])} {int(tags[i])} " + " ".join(str(v + 1) for v in el))
        out += ["$EndElements"]
        return "\n".join(out) + "\n"

    def write_msh(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_msh())

    def to_vtk(self, cell_data: dict | None = None) -> str:
        """Legacy ASCII VTK unstructured grid (triangles 5, quads 9, polygons 7)."""
        out = ["# vtk DataFile Version 3.0", "geo2d mesh", "ASCII", "DATASET UNSTRUCTURED_GRID",
               f"POINTS {self.n_nodes} double"]
        out += [f"{x:.16g} {y:.16g} 0" for x, y in self.nodes]
        els = self.elements()
        out.append(f"CELLS {len(els)} {sum(len(e) + 1 for e in els)}")
        out += [f"{len(e)} " + " ".join(map(str, e)) for e in els]
        out.append(f"CELL_TYPES {len(els)}")
        out += [str({3: 5, 4: 9}.get(len(e), 7)) for e in els]
        cell_data = dict(cell_data or {})
        cell_data.setdefault("quality", self.quality())
        out.append(f"CELL_DATA {len(els)}")
        for name, vals in cell_data.items():
            out += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"]
            out += [f"{float(v):.10g}" for v in vals]
        return "\n".join(out) + "\n"

    def write_vtk(self, path: str, **kw) -> None:
        with open(path, "w") as f:
            f.write(self.to_vtk(**kw))


def read_msh(path: str) -> Mesh:
    """Read a Gmsh MSH version 2 file (2D linear triangles and quads; other elements skipped)."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    ids, xy = [], []
    for k in range(n):
        a = lines[i + 2 + k].split()
        ids.append(int(a[0]))
        xy.append((float(a[1]), float(a[2])))
    remap = {nid: k for k, nid in enumerate(ids)}
    i = lines.index("$Elements")
    m = int(lines[i + 1])
    elements, tags = [], []
    for k in range(m):
        a = list(map(int, lines[i + 2 + k].split()))
        if a[1] in (2, 3):
            elements.append([remap[v] for v in a[3 + a[2]:]])
            tags.append(a[3] if a[2] > 0 else 1)
    nodes = np.array(xy)
    tags = np.array(tags, int)
    mesh = Mesh(nodes, elements, {"tag": tags, "region": tags - 1} if elements else None)
    used = np.zeros(len(nodes), bool)
    used[mesh.conn] = True
    if not used.all():                           # drop nodes that belong to no 2D element
        newid = np.cumsum(used) - 1
        mesh = Mesh(nodes[used], [newid[e] for e in mesh.elements()], mesh.data, orient=False)
    return mesh


def gmsh_mesh(geom, lc: float = 1.0, quads: bool = True, exact: bool = True, snap: bool = True,
              gmsh: str = "gmsh", extra=()) -> Mesh:
    """Mesh a Geometry with the Gmsh binary and return a Mesh.

    ``exact=False`` meshes the chord polygon (straight edges) instead of the true
    curves; with ``snap`` the boundary nodes are then moved onto the curves by
    :func:`geo2d.snap.snap_mesh`, which also records the boundary-to-geometry
    link in ``mesh.data`` (``bnd_loop``, ``bnd_edge``, ``bnd_t``).
    """
    from .export import save_geo
    from .geometry import Geometry, Loop
    from .snap import snap_mesh
    exe = shutil.which(gmsh)
    if exe is None:
        raise FileNotFoundError("gmsh binary not found on PATH")
    g = geom if exact else Geometry([Loop(lp.points, np.zeros(lp.n), lp.tags) for lp in geom.loops])
    with tempfile.TemporaryDirectory() as tmp:
        geo = os.path.join(tmp, "domain.geo")
        msh = os.path.join(tmp, "domain.msh")
        save_geo(g, geo, lc=lc, recombine=quads)
        res = subprocess.run([exe, "-2", geo, "-format", "msh2", "-o", msh, "-v", "0", *extra],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gmsh failed: {res.stderr}")
        mesh = read_msh(msh)
    if snap:
        mesh = snap_mesh(geom, mesh)
    mesh.meta.update({"lc": lc, "exact": exact, "quads": quads})
    return mesh
