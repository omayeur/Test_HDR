"""Command-line interface: ``stl-deviation A.stl B.stl``."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from .align import centroid_alignment, icp_rigid
from .deviation import compute_deviation
from .io import read_stl, write_ply, write_vtk
from .metrics import summarize
from .report import (auto_range, diverging_colors, format_summary, plot_histograms,
                     sequential_colors, write_csv, write_json)
from .voxel import overlap_metrics


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "mesh"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stl-deviation",
        description="Compare two STL surfaces: signed distances, ASSD/RMS/Hausdorff/HD95, "
                    "tolerance fractions, volumes, optional Dice, colour-mapped exports.",
    )
    p.add_argument("mesh_a", help="first STL (e.g. the manual segmentation)")
    p.add_argument("mesh_b", help="second STL (e.g. the automatic segmentation)")
    p.add_argument("-o", "--out", default="deviation_output", help="output folder (default: deviation_output)")
    p.add_argument("--name-a", default=None, help="label of A (default: file name)")
    p.add_argument("--name-b", default=None, help="label of B (default: file name)")
    p.add_argument("--units", default="mm", help="length unit of the STL files, for labels only (default: mm)")
    p.add_argument("--tolerances", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="acceptance thresholds, in the STL units (default: 0.5 1 2)")
    p.add_argument("--samples", type=int, default=200_000, help="random surface samples per direction (default: 200000)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--align", choices=["none", "centroid", "icp"], default="none",
                   help="move A onto B first (default: none; do not align two segmentations of the same image)")
    p.add_argument("--overlap", action="store_true", help="also compute voxel Dice and Jaccard (closed meshes only)")
    p.add_argument("--voxel-size", type=float, default=None, help="voxel size for --overlap (default: auto, ~1/150 of the size)")
    p.add_argument("--range", type=float, default=None, dest="color_range",
                   help="colour-map half-range in the STL units (default: automatic)")
    p.add_argument("--no-plots", action="store_true", help="skip the PNG figure")
    p.add_argument("--no-meshes", action="store_true", help="skip the coloured VTK/PLY exports")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path_a, path_b = Path(args.mesh_a), Path(args.mesh_b)
    for path in (path_a, path_b):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2

    mesh_a, mesh_b = read_stl(path_a), read_stl(path_b)
    name_a = args.name_a or path_a.stem
    name_b = args.name_b or path_b.stem

    alignment = {"mode": args.align}
    if args.align == "centroid":
        matrix = centroid_alignment(mesh_a, mesh_b)
        mesh_a = mesh_a.transformed(matrix)
        alignment["matrix"] = matrix.tolist()
    elif args.align == "icp":
        matrix, history = icp_rigid(mesh_a, mesh_b, seed=args.seed)
        mesh_a = mesh_a.transformed(matrix)
        alignment.update({"matrix": matrix.tolist(), "rms_history": history})

    result = compute_deviation(mesh_a, mesh_b, n_samples=args.samples, seed=args.seed)
    metrics = summarize(result, tolerances=args.tolerances, units=args.units)
    metrics["inputs"] = {"mesh_a": str(path_a), "mesh_b": str(path_b), "name_a": name_a, "name_b": name_b}
    metrics["alignment"] = alignment
    if args.overlap:
        try:
            metrics["overlap"] = overlap_metrics(result.a, result.b, voxel_size=args.voxel_size)
        except ValueError as exc:
            metrics.setdefault("warnings", []).append(f"Overlap skipped: {exc}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_csv(out / "metrics.csv", metrics, label=f"{name_a}_vs_{name_b}")
    if not args.no_plots:
        if not plot_histograms(result, out / "deviation_histogram.png", name_a, name_b,
                               tolerances=args.tolerances, units=args.units):
            metrics.setdefault("warnings", []).append("matplotlib not installed: figure skipped (pip install matplotlib)")
    if not args.no_meshes:
        vrange = args.color_range or auto_range(result)
        for mesh, dev, name, other in ((result.a, result.a_to_b, name_a, name_b),
                                       (result.b, result.b_to_a, name_b, name_a)):
            d = dev.vertex_distances
            colors = diverging_colors(d, vrange) if dev.signed else sequential_colors(d, vrange)
            base = out / f"{_slug(name)}_vs_{_slug(other)}"
            write_vtk(f"{base}.vtk", mesh, {"deviation": d, "abs_deviation": np.abs(d)})
            write_ply(f"{base}.ply", mesh, colors, quality=d)
        metrics["color_range"] = vrange

    print(format_summary(metrics, name_a, name_b))
    print(f"\nResults written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
