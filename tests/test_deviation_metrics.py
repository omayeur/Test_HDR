import numpy as np
import pytest

from stl_deviation.align import icp_rigid
from stl_deviation.deviation import compute_deviation
from stl_deviation.metrics import summarize
from stl_deviation.synthetic import demo_pair, ellipsoid, icosphere
from stl_deviation.voxel import overlap_metrics


def test_concentric_spheres():
    a, b = icosphere(10.0, 4), icosphere(11.0, 4)
    m = summarize(compute_deviation(a, b, n_samples=20000), tolerances=(0.5, 2.0))
    assert m["a_to_b"]["mean_signed"] == pytest.approx(-1.0, abs=0.03)   # A is inside B
    assert m["b_to_a"]["mean_signed"] == pytest.approx(+1.0, abs=0.03)
    assert m["symmetric"]["assd"] == pytest.approx(1.0, abs=0.03)
    assert m["symmetric"]["hd"] == pytest.approx(1.0, abs=0.05)
    assert m["symmetric"]["within_tolerance"]["2"] == 1.0
    assert m["symmetric"]["within_tolerance"]["0.5"] == 0.0
    assert m["volume"]["relative_difference_pct"] == pytest.approx(100 * (1000 / 1331 - 1), abs=1.0)


def test_translated_sphere():
    a, b = icosphere(10.0, 4), icosphere(10.0, 4, center=(1.0, 0, 0))
    m = summarize(compute_deviation(a, b, n_samples=40000))
    assert m["symmetric"]["assd"] == pytest.approx(0.5, abs=0.05)   # mean |cos| * shift
    assert m["symmetric"]["hd"] == pytest.approx(1.0, abs=0.05)
    assert m["symmetric"]["hd95"] < 1.0


def test_open_surface_gives_unsigned_distances():
    sphere = icosphere(10.0, 3)
    hemi = sphere.select_faces(sphere.triangles.mean(axis=1)[:, 2] > 0)
    res = compute_deviation(hemi, sphere, n_samples=5000)
    assert res.a_to_b.signed            # reference = closed sphere
    assert not res.b_to_a.signed        # reference = open hemisphere
    assert (res.b_to_a.sample_distances >= 0).all()
    m = summarize(res)
    assert "mean_signed" in m["a_to_b"] and "mean_signed" not in m["b_to_a"]
    assert m["warnings"]


def test_overlap_two_spheres_matches_analytic_lens():
    r, d = 10.0, 2.0
    a, b = icosphere(r, 4), icosphere(r, 4, center=(d, 0, 0))
    o = overlap_metrics(a, b, voxel_size=0.25)
    lens = np.pi * (4 * r + d) * (2 * r - d) ** 2 / 12
    vol = 4 / 3 * np.pi * r ** 3
    assert o["dice"] == pytest.approx(lens / vol, abs=0.01)
    assert o["volume_a_voxels"] == pytest.approx(a.signed_volume(), rel=0.01)
    same = overlap_metrics(a, a, voxel_size=0.25)
    assert same["dice"] == pytest.approx(1.0)


def test_icp_recovers_rigid_motion():
    target = ellipsoid((12, 8, 5), 4)
    angle = np.deg2rad(6)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0, 1.0],
                    [np.sin(angle), np.cos(angle), 0, -0.5],
                    [0, 0, 1, 0.8],
                    [0, 0, 0, 1]])
    moved = target.transformed(rot)
    matrix, history = icp_rigid(moved, target, n_samples=8000)
    aligned = moved.transformed(matrix)
    m = summarize(compute_deviation(aligned, target, n_samples=20000))
    assert m["symmetric"]["assd"] < 0.05
    assert history[-1] <= history[0]


def test_demo_pair_shows_expected_bias():
    manual, auto = demo_pair(subdivisions=4)
    m = summarize(compute_deviation(manual, auto, n_samples=20000))
    assert m["symmetric"]["hd"] > 1.0                  # the bump
    assert -0.3 < m["a_to_b"]["mean_signed"] < 0.3     # globally small bias
