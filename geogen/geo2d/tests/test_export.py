import shutil
import subprocess

import pytest

from geogen.geo2d import generate, to_dxf, to_geo


def test_geo_and_dxf_strings():
    g = generate(1, preset="rounded")
    s = to_geo(g, recombine=True)
    assert "Plane Surface(1)" in s and "Circle(" in s and "Mesh.RecombineAll = 1;" in s
    d = to_dxf(g)
    assert d.count("LWPOLYLINE") == len(g.loops)
    assert "\n42\n" in d          # at least one bulge


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh binary not installed")
def test_gmsh_meshes_export(tmp_path):
    g = generate(3, preset="rounded")
    geo = tmp_path / "shape.geo"
    geo.write_text(to_geo(g, lc=1.0, recombine=True))
    out = tmp_path / "shape.msh"
    res = subprocess.run(["gmsh", "-2", str(geo), "-o", str(out), "-v", "2"], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert out.exists() and out.stat().st_size > 0
