"""Two-way signed deviation between two surfaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SurfaceIndex, sample_surface
from .mesh import Mesh


@dataclass
class DirectionalDeviation:
    """Distances from the points of one surface to the other surface.

    Positive = the measured surface lies *outside* the reference surface, negative = inside.
    When ``signed`` is False (open reference surface) all values are >= 0.
    """

    sample_distances: np.ndarray   # area-weighted random samples, used for statistics
    vertex_distances: np.ndarray   # one value per vertex, used for colour maps and extremes
    signed: bool


@dataclass
class DeviationResult:
    a: Mesh
    b: Mesh
    a_to_b: DirectionalDeviation   # points of A measured against surface B
    b_to_a: DirectionalDeviation   # points of B measured against surface A
    n_samples: int
    seed: int


def compute_deviation(a: Mesh, b: Mesh, n_samples: int = 200_000, seed: int = 0, k: int = 32) -> DeviationResult:
    """Compute exact signed distances A -> B and B -> A.

    Statistics should use ``sample_distances`` (uniform over the surface, independent of how
    the triangles happen to be distributed); ``vertex_distances`` are for visualisation.
    """
    a = a.compact()
    b = b.compact()
    rng = np.random.default_rng(seed)
    index_a = SurfaceIndex(a, k=k)
    index_b = SurfaceIndex(b, k=k)

    def one_way(source: Mesh, target_index: SurfaceIndex) -> DirectionalDeviation:
        samples, _ = sample_surface(source, n_samples, rng)
        d_samples, _, _ = target_index.query(samples)
        d_vertices, _, _ = target_index.query(source.vertices)
        return DirectionalDeviation(d_samples, d_vertices, target_index.signed)

    return DeviationResult(
        a=a, b=b,
        a_to_b=one_way(a, index_b),
        b_to_a=one_way(b, index_a),
        n_samples=n_samples, seed=seed,
    )
