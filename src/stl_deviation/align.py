"""Optional alignment of surface A onto surface B before comparing them.

Do NOT align when both surfaces come from the same image (e.g. manual vs. automatic
segmentation of one DICOM series): they already share a coordinate system and aligning
would hide a real, systematic offset. Alignment is meant for comparing a morphed or
generic model with a patient-specific one, or surfaces from different acquisitions.
"""
from __future__ import annotations

import numpy as np

from .geometry import SurfaceIndex, sample_surface
from .mesh import Mesh


def centroid_alignment(source: Mesh, target: Mesh) -> np.ndarray:
    """4x4 translation that superimposes the area-weighted surface centroids."""

    def centroid(m: Mesh) -> np.ndarray:
        t = m.triangles
        areas = m.face_areas()
        return (t.mean(axis=1) * areas[:, None]).sum(axis=0) / areas.sum()

    matrix = np.eye(4)
    matrix[:3, 3] = centroid(target) - centroid(source)
    return matrix


def _kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = qc - rot @ pc
    return matrix


def icp_rigid(source: Mesh, target: Mesh, max_iter: int = 60, tol: float = 1e-7,
              n_samples: int = 20000, trim: float = 0.9, seed: int = 0):
    """Trimmed rigid ICP (point-to-surface) moving ``source`` onto ``target``.

    ``trim`` keeps only that fraction of the closest pairs at each iteration, which makes the
    fit robust to local differences. Returns (4x4 matrix, list of RMS residuals per iteration).
    """
    rng = np.random.default_rng(seed)
    points, _ = sample_surface(source, n_samples, rng)
    index = SurfaceIndex(target)
    total = centroid_alignment(source, target)
    points = points @ total[:3, :3].T + total[:3, 3]
    history = []
    for _ in range(max_iter):
        dist, closest, _ = index.query(points)
        keep = np.argsort(np.abs(dist))[: max(int(trim * len(points)), 10)]
        step = _kabsch(points[keep], closest[keep])
        points = points @ step[:3, :3].T + step[:3, 3]
        total = step @ total
        history.append(float(np.sqrt(np.mean(dist[keep] ** 2))))
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break
    return total, history
