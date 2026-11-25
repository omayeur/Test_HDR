"""Self-contained demo: no patient data needed.

Creates two synthetic STL files that mimic an automatic and a manual segmentation of the
same structure, then runs the full comparison. Run from the repository root:

    python examples/demo.py

Results are written to examples/_demo_output/ (ignored by git).
"""
from pathlib import Path

from stl_deviation.cli import main
from stl_deviation.io import write_stl
from stl_deviation.synthetic import demo_pair

out = Path(__file__).parent / "_demo_output"
out.mkdir(exist_ok=True)

manual, automatic = demo_pair(subdivisions=5)
write_stl(out / "manual.stl", manual)
write_stl(out / "automatic.stl", automatic)

raise SystemExit(main([
    str(out / "manual.stl"), str(out / "automatic.stl"),
    "--name-a", "manual", "--name-b", "automatic",
    "--out", str(out / "results"),
    "--tolerances", "0.5", "1", "2",
    "--overlap", "--voxel-size", "0.2",
]))
