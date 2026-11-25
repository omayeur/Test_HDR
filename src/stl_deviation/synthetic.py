"""Synthetic meshes for tests, demos and teaching (no patient data needed)."""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


def icosphere(radius: float = 1.0, subdivisions: int = 3, center=(0.0, 0.0, 0.0), name: str = "sphere") -> Mesh:
    """Geodesic sphere made by subdividing an icosahedron (closed, outward-oriented)."""
    phi = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        edges = np.sort(np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        mid = v[unique].mean(axis=1)
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = len(v) + inverse.reshape(-1, 3)
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
        v = np.vstack([v, mid])
        f = np.vstack([
            np.stack([a, m01, m20], axis=1),
            np.stack([m01, b, m12], axis=1),
            np.stack([m20, m12, c], axis=1),
            np.stack([m01, m12, m20], axis=1),
        ])
    mesh = Mesh(v * radius + np.asarray(center, dtype=np.float64), f, name=name)
    return mesh.outward()


def ellipsoid(radii=(12.0, 8.0, 5.0), subdivisions: int = 4, name: str = "ellipsoid") -> Mesh:
    """Triaxial ellipsoid (no rotational symmetry, so it is a good test shape for alignment)."""
    base = icosphere(1.0, subdivisions)
    return Mesh(base.vertices * np.asarray(radii, dtype=np.float64), base.faces, name=name).outward()


def demo_pair(subdivisions: int = 5):
    """Two ellipsoids mimicking an automatic and a manual segmentation of the same structure.

    The "manual" one is slightly under-segmented overall (-0.3) with one over-segmented bump
    (+1.5, smooth) and a little noise, so every metric and colour map has something to show.
    Returns (manual, automatic).
    """
    auto = ellipsoid(subdivisions=subdivisions, name="automatic")
    rng = np.random.default_rng(42)
    v = auto.vertices
    direction = np.array([0.0, 1.0, 0.4])
    direction /= np.linalg.norm(direction)
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    angle = np.arccos(np.clip(unit @ direction, -1.0, 1.0))
    shift = -0.3 + 1.5 * np.exp(-(angle ** 2) / (2 * 0.35 ** 2)) + rng.normal(0.0, 0.05, len(v))
    manual = Mesh(v + unit * shift[:, None], auto.faces, name="manual").outward()
    return manual, auto
