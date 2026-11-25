"""Minimal triangle-mesh container used throughout the package."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A triangulated surface: ``vertices`` (n, 3) floats and ``faces`` (m, 3) vertex indices."""

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if len(self.faces) == 0:
            raise ValueError("the mesh has no faces")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("face indices are out of range")

    # ------------------------------------------------------------------ basic
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def triangles(self) -> np.ndarray:
        """Array (m, 3, 3) of triangle corner coordinates."""
        return self.vertices[self.faces]

    @property
    def bounds(self) -> np.ndarray:
        """Array [[xmin, ymin, zmin], [xmax, ymax, zmax]] over referenced vertices."""
        v = self.vertices[np.unique(self.faces)]
        return np.vstack([v.min(axis=0), v.max(axis=0)])

    def face_normals(self) -> np.ndarray:
        t = self.triangles
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norm > 0, norm, 1.0)

    def face_areas(self) -> np.ndarray:
        t = self.triangles
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    def area(self) -> float:
        return float(self.face_areas().sum())

    def signed_volume(self) -> float:
        """Volume enclosed by the surface (positive if faces point outwards).

        Only meaningful for a closed, consistently oriented mesh (see :meth:`is_closed`).
        """
        t = self.triangles
        t = t - t.reshape(-1, 3).mean(axis=0)  # centre for numerical stability
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    # --------------------------------------------------------------- topology
    def is_closed(self) -> bool:
        """True for a closed, consistently oriented manifold surface.

        Every directed edge must appear exactly once and be matched by its reverse.
        """
        n = len(self.vertices)
        f = self.faces
        directed = np.concatenate([f[:, 0] * n + f[:, 1], f[:, 1] * n + f[:, 2], f[:, 2] * n + f[:, 0]])
        reverse = np.concatenate([f[:, 1] * n + f[:, 0], f[:, 2] * n + f[:, 1], f[:, 0] * n + f[:, 2]])
        directed.sort()
        reverse.sort()
        return bool(np.array_equal(directed, reverse)) and bool(np.all(directed[1:] != directed[:-1]))

    def outward(self) -> "Mesh":
        """Return a mesh whose faces point outwards (flips them if the volume is negative)."""
        if self.is_closed() and self.signed_volume() < 0:
            return Mesh(self.vertices.copy(), self.faces[:, ::-1].copy(), self.name)
        return self

    # ------------------------------------------------------------- operations
    def compact(self) -> "Mesh":
        """Drop vertices that no face uses."""
        used, inverse = np.unique(self.faces, return_inverse=True)
        if len(used) == len(self.vertices):
            return self
        return Mesh(self.vertices[used], inverse.reshape(-1, 3), self.name)

    def select_faces(self, mask: np.ndarray) -> "Mesh":
        """Sub-mesh made of the faces where ``mask`` is True (vertices are compacted)."""
        return Mesh(self.vertices, self.faces[np.asarray(mask, dtype=bool)], self.name).compact()

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        """Apply a 4x4 homogeneous transform (face order is flipped if it mirrors the mesh)."""
        m = np.asarray(matrix, dtype=np.float64)
        rot, trans = m[:3, :3], m[:3, 3]
        faces = self.faces if np.linalg.det(rot) > 0 else self.faces[:, ::-1]
        return Mesh(self.vertices @ rot.T + trans, faces.copy(), self.name)
