import numpy as np
import pytest

from stl_deviation.geometry import SurfaceIndex, closest_point_on_triangle, sample_surface
from stl_deviation.io import read_stl, write_ply, write_stl, write_vtk
from stl_deviation.mesh import Mesh
from stl_deviation.synthetic import ellipsoid, icosphere


def test_closest_point_all_regions():
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    cases = [
        ((0.25, 0.25, 1.0), (0.25, 0.25, 0.0)),   # interior
        ((-1.0, -1.0, 0.0), (0.0, 0.0, 0.0)),     # vertex A
        ((2.0, -1.0, 0.0), (1.0, 0.0, 0.0)),      # vertex B
        ((-1.0, 3.0, 0.0), (0.0, 1.0, 0.0)),      # vertex C
        ((0.5, -1.0, 0.0), (0.5, 0.0, 0.0)),      # edge AB
        ((-1.0, 0.5, 0.0), (0.0, 0.5, 0.0)),      # edge AC
        ((1.0, 1.0, 0.0), (0.5, 0.5, 0.0)),       # edge BC
    ]
    p = np.array([c_[0] for c_ in cases])
    expected = np.array([c_[1] for c_ in cases])
    n = len(cases)
    got = closest_point_on_triangle(p, np.tile(a, (n, 1)), np.tile(b, (n, 1)), np.tile(c, (n, 1)))
    assert np.allclose(got, expected)


def test_index_matches_brute_force():
    mesh = ellipsoid((12, 6, 4), subdivisions=2)
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(300, 3)) * 8
    index = SurfaceIndex(mesh, k=32)
    d, _, _ = index.query(pts)
    tri = mesh.outward().triangles
    brute = np.empty(len(pts))
    for i, p in enumerate(pts):
        q = closest_point_on_triangle(np.tile(p, (len(tri), 1)), tri[:, 0], tri[:, 1], tri[:, 2])
        brute[i] = np.linalg.norm(q - p, axis=1).min()
    assert np.allclose(np.abs(d), brute, atol=1e-9)


def test_sphere_topology_and_volume():
    s = icosphere(10.0, 4)
    assert s.is_closed()
    assert s.signed_volume() > 0
    assert s.signed_volume() == pytest.approx(4 / 3 * np.pi * 1000, rel=0.01)
    assert s.area() == pytest.approx(4 * np.pi * 100, rel=0.01)
    flipped = Mesh(s.vertices, s.faces[:, ::-1])
    assert flipped.signed_volume() < 0
    assert flipped.outward().signed_volume() > 0


def test_samples_lie_on_surface_and_sign():
    s = icosphere(10.0, 3)
    pts, _ = sample_surface(s, 2000, np.random.default_rng(0))
    d, _, _ = SurfaceIndex(s).query(pts)
    assert np.abs(d).max() < 1e-9
    d_in, _, _ = SurfaceIndex(s).query(np.array([[0.0, 0, 0], [0, 0, 20.0]]))
    assert d_in[0] < 0 < d_in[1]
    assert d_in[0] == pytest.approx(-10.0, abs=0.2)
    assert d_in[1] == pytest.approx(10.0, abs=0.01)


@pytest.mark.parametrize("binary", [True, False])
def test_stl_round_trip(tmp_path, binary):
    s = icosphere(5.0, 2)
    path = tmp_path / "s.stl"
    write_stl(path, s, binary=binary)
    back = read_stl(path)
    assert back.n_faces == s.n_faces and back.n_vertices == s.n_vertices
    assert back.is_closed()
    assert back.signed_volume() == pytest.approx(s.signed_volume(), rel=1e-5)


def test_degenerate_triangles_are_removed(tmp_path):
    s = icosphere(5.0, 1)
    v = np.vstack([s.vertices, s.vertices[:1]])
    f = np.vstack([s.faces, [[0, 0, len(v) - 1]]])
    path = tmp_path / "d.stl"
    write_stl(path, Mesh(v, f), binary=True)
    assert read_stl(path).n_faces == s.n_faces


def test_vtk_and_ply_writers(tmp_path):
    s = icosphere(5.0, 1)
    vals = np.linspace(-1, 1, s.n_vertices)
    write_vtk(tmp_path / "a.vtk", s, {"deviation": vals})
    text = (tmp_path / "a.vtk").read_text()
    assert "POINT_DATA" in text and "SCALARS deviation" in text
    write_ply(tmp_path / "a.ply", s, np.zeros((s.n_vertices, 3), dtype=np.uint8), quality=vals)
    assert (tmp_path / "a.ply").read_text().startswith("ply")
