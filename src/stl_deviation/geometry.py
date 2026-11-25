"""Exact point-to-triangle geometry, surface sampling and signed distance to a mesh."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", u, v)


def closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on each triangle (a, b, c) to the matching point p.

    Vectorised version of the region-based algorithm of C. Ericson, *Real-Time Collision
    Detection* (2004), section 5.1.5. All inputs have shape (n, 3).
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    out = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def assign(mask: np.ndarray, value: np.ndarray) -> None:
        m = mask & ~done
        out[m] = value[m]
        np.logical_or(done, m, out=done)

    with np.errstate(divide="ignore", invalid="ignore"):
        assign((d1 <= 0) & (d2 <= 0), a)                                    # vertex A
        assign((d3 >= 0) & (d4 <= d3), b)                                   # vertex B
        assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)   # edge AB
        assign((d6 >= 0) & (d5 <= d6), c)                                   # vertex C
        assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)   # edge AC
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w[:, None] * (c - b))  # edge BC
        denom = va + vb + vc
        interior = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
    rest = ~done
    out[rest] = interior[rest]
    bad = ~np.isfinite(out).all(axis=1)  # only possible for numerically degenerate triangles
    out[bad] = a[bad]
    return out


def sample_surface(mesh: Mesh, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` points uniformly distributed over the surface (area-weighted).

    Returns the points (n, 3) and the index of the triangle each one lies on.
    """
    areas = mesh.face_areas()
    prob = areas / areas.sum()
    tri_idx = rng.choice(mesh.n_faces, size=n, p=prob)
    r1, r2 = rng.random(n), rng.random(n)
    u = np.sqrt(r1)
    w = np.stack([1.0 - u, u * (1.0 - r2), u * r2], axis=1)
    corners = mesh.triangles[tri_idx]
    points = np.einsum("ij,ijk->ik", w, corners)
    return points, tri_idx


class SurfaceIndex:
    """Fast signed distance from arbitrary points to a triangulated surface.

    Candidate triangles are found with a KD-tree on triangle centroids (the ``k`` nearest),
    then the exact point-to-triangle distance is evaluated on each candidate. For the
    regular triangulations produced by marching cubes this matches a brute-force search
    (checked in the test suite); increase ``k`` for strongly anisotropic meshes.

    The sign is positive outside the surface and negative inside, which needs a closed,
    consistently oriented mesh (``self.signed``). For an open surface the distances are
    unsigned (always >= 0), because "inside" is then undefined.
    """

    def __init__(self, mesh: Mesh, k: int = 32):
        mesh = mesh.compact().outward()
        self.mesh = mesh
        self.signed = mesh.is_closed()
        self.k = int(min(k, mesh.n_faces))
        self._tri = mesh.triangles
        self._normals = mesh.face_normals()
        self._tree = cKDTree(self._tri.mean(axis=1))

    def query(self, points: np.ndarray, chunk: int = 10000):
        """Return (signed distance, closest point, triangle index) for each query point."""
        points = np.ascontiguousarray(points, dtype=np.float64)
        n = len(points)
        dist = np.empty(n)
        closest = np.empty((n, 3))
        tri = np.empty(n, dtype=np.int64)
        for start in range(0, n, chunk):
            p = points[start:start + chunk]
            m = len(p)
            _, idx = self._tree.query(p, k=self.k, workers=-1)
            idx = np.asarray(idx).reshape(m, -1)
            corners = self._tri[idx]                       # (m, k, 3, 3)
            pk = np.broadcast_to(p[:, None, :], (m, idx.shape[1], 3))
            q = closest_point_on_triangle(
                pk.reshape(-1, 3),
                corners[:, :, 0].reshape(-1, 3),
                corners[:, :, 1].reshape(-1, 3),
                corners[:, :, 2].reshape(-1, 3),
            ).reshape(m, idx.shape[1], 3)
            d2 = ((q - pk) ** 2).sum(axis=2)
            j = d2.argmin(axis=1)
            rows = np.arange(m)
            best = idx[rows, j]
            qb = q[rows, j]
            d = np.sqrt(d2[rows, j])
            if self.signed:
                side = np.einsum("ij,ij->i", p - qb, self._normals[best])
                d = np.where(side < 0, -d, d)
            dist[start:start + m] = d
            closest[start:start + m] = qb
            tri[start:start + m] = best
        return dist, closest, tri
