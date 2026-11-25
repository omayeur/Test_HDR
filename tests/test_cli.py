import json

from stl_deviation.cli import main
from stl_deviation.io import write_stl
from stl_deviation.synthetic import demo_pair


def test_cli_end_to_end(tmp_path, capsys):
    manual, auto = demo_pair(subdivisions=3)
    a, b = tmp_path / "manual.stl", tmp_path / "auto.stl"
    write_stl(a, manual)
    write_stl(b, auto)
    out = tmp_path / "res"
    code = main([str(a), str(b), "-o", str(out), "--samples", "5000", "--overlap",
                 "--voxel-size", "0.5", "--name-a", "manual", "--name-b", "ai"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["symmetric"]["hd95"] > 0
    assert 0 < metrics["overlap"]["dice"] <= 1
    for name in ("metrics.csv", "deviation_histogram.png", "manual_vs_ai.vtk", "manual_vs_ai.ply",
                 "ai_vs_manual.vtk", "ai_vs_manual.ply"):
        assert (out / name).exists(), name
    assert "ASSD" in capsys.readouterr().out


def test_cli_missing_file(tmp_path):
    assert main([str(tmp_path / "nope.stl"), str(tmp_path / "nope2.stl")]) == 2
