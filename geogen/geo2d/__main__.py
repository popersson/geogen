"""Command line: python -m geogen.geo2d {generate,gallery,export,info} ..."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

from . import PRESETS, Params, descriptors, generate_many, load_jsonl, plot_gallery, save_dxf, save_geo, save_jsonl  # noqa: E402


def _parse_overrides(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"bad override {it!r}; use key=value")
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geogen.geo2d", description="Seeded 2D mechanical-part domain generator.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate shapes into a JSON-lines file")
    g.add_argument("--n", type=int, default=24)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--preset", default="default", choices=sorted(PRESETS))
    g.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="parameter overrides, JSON values, e.g. ratio=12 n_holes=[1,3] symmetry='\"x\"'")
    g.add_argument("--out", required=True)
    g.add_argument("--gallery", help="also write a PNG gallery of the generated shapes")

    gal = sub.add_parser("gallery", help="render a PNG gallery from a JSON-lines file")
    gal.add_argument("jsonl")
    gal.add_argument("--out", required=True)
    gal.add_argument("--ncols", type=int, default=6)
    gal.add_argument("--max", type=int, default=48)
    gal.add_argument("--tags", action="store_true", help="colour edges by feature tag")
    gal.add_argument("--points", action="store_true", help="show control points")
    gal.add_argument("--chords", action="store_true", help="show the chord polygon")

    ex = sub.add_parser("export", help="export each shape to Gmsh .geo, DXF or JSON")
    ex.add_argument("jsonl")
    ex.add_argument("--fmt", choices=["geo", "dxf", "json"], default="geo")
    ex.add_argument("--outdir", required=True)
    ex.add_argument("--lc", type=float, default=1.0, help="Gmsh mesh size in lattice units")
    ex.add_argument("--recombine", action="store_true", help="ask Gmsh for an all-quad mesh")

    info = sub.add_parser("info", help="print descriptor statistics of a JSON-lines file")
    info.add_argument("jsonl")

    a = ap.parse_args(argv)
    if a.cmd == "generate":
        P = Params().replace(**PRESETS[a.preset]).replace(**_parse_overrides(a.set))
        geoms = generate_many(a.seed, a.n, P)
        save_jsonl(geoms, a.out)
        print(f"wrote {len(geoms)} shapes to {a.out}")
        if a.gallery:
            fig = plot_gallery(geoms[:48], titles=[g.meta["seed"] for g in geoms[:48]])
            fig.savefig(a.gallery, dpi=130)
            print(f"wrote {a.gallery}")
    elif a.cmd == "gallery":
        geoms = load_jsonl(a.jsonl)[: a.max]
        fig = plot_gallery(geoms, ncols=a.ncols, show_tags=a.tags, show_points=a.points,
                           show_chords=a.chords, titles=[g.meta.get("seed", i) for i, g in enumerate(geoms)])
        fig.savefig(a.out, dpi=130)
        print(f"wrote {a.out}")
    elif a.cmd == "export":
        os.makedirs(a.outdir, exist_ok=True)
        for i, g in enumerate(load_jsonl(a.jsonl)):
            stem = os.path.join(a.outdir, f"shape_{g.meta.get('seed', i)}")
            if a.fmt == "geo":
                save_geo(g, stem + ".geo", lc=a.lc, recombine=a.recombine)
            elif a.fmt == "dxf":
                save_dxf(g, stem + ".dxf")
            else:
                g.save(stem + ".json")
        print(f"exported to {a.outdir}")
    elif a.cmd == "info":
        geoms = load_jsonl(a.jsonl)
        keys = ["n_holes", "n_corners", "n_arcs", "n_concave", "n_acute", "f_min", "ratio_bbox", "ratio_edge",
                "area_fraction"]
        rows = [g.meta.get("descriptors") or descriptors.compute(g) for g in geoms]
        print(f"{len(geoms)} shapes")
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                print(f"  {k:14s} min {min(vals):8.3g}  mean {sum(vals) / len(vals):8.3g}  max {max(vals):8.3g}")
        sx = sum(r["sym_x"] for r in rows)
        sy = sum(r["sym_y"] for r in rows)
        print(f"  symmetric about x-axis: {sx}, about y-axis: {sy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
