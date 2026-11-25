"""Reading and writing meshes (STL in; STL, VTK and PLY out)."""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .mesh import Mesh

_BINARY_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_VERTEX_RE = re.compile(
    r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", re.IGNORECASE
)


# ---------------------------------------------------------------------- reading
def read_stl(path) -> Mesh:
    """Read a binary or ASCII STL file.

    Duplicate vertices are merged (exact equality at float32 precision, which is what
    STL stores) and degenerate triangles are removed, so the result is an indexed mesh.
    """
    path = Path(path)
    data = path.read_bytes()
    if _looks_binary(data):
        count = struct.unpack("<I", data[80:84])[0]
        tris = np.frombuffer(data, dtype=_BINARY_DTYPE, count=count, offset=84)["v"]
        tris = tris.astype(np.float32)
    else:
        found = _VERTEX_RE.findall(data.decode("utf-8", errors="replace"))
        if not found or len(found) % 3 != 0:
            raise ValueError(f"{path} is not a valid STL file (no triangles found)")
        tris = np.asarray(found, dtype=np.float64).astype(np.float32).reshape(-1, 3, 3)
    return _triangles_to_mesh(tris, name=path.stem)


def _looks_binary(data: bytes) -> bool:
    if len(data) < 84:
        return False
    count = struct.unpack("<I", data[80:84])[0]
    return 84 + 50 * count == len(data)


def _triangles_to_mesh(tris: np.ndarray, name: str) -> Mesh:
    flat = np.ascontiguousarray(tris.reshape(-1, 3), dtype=np.float32)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    vertices = unique.astype(np.float64)

    # drop faces that reuse a vertex or have zero area
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    t = vertices[faces]
    area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    faces = faces[area2 > 0]
    if len(faces) == 0:
        raise ValueError("the STL contains only degenerate triangles")
    return Mesh(vertices, faces, name=name).compact()


# ---------------------------------------------------------------------- writing
def write_stl(path, mesh: Mesh, binary: bool = True) -> None:
    """Write a mesh as STL (binary by default, ASCII if ``binary=False``)."""
    path = Path(path)
    tris = mesh.triangles
    normals = mesh.face_normals()
    if binary:
        arr = np.zeros(len(tris), dtype=_BINARY_DTYPE)
        arr["normal"] = normals
        arr["v"] = tris
        header = b"stl-deviation".ljust(80, b" ")
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(struct.pack("<I", len(tris)))
            fh.write(arr.tobytes())
    else:
        lines = [f"solid {mesh.name}"]
        for n, t in zip(normals, tris):
            lines.append(f"  facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}")
            lines.append("    outer loop")
            for p in t:
                lines.append(f"      vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {mesh.name}")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_vtk(path, mesh: Mesh, point_data: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write a legacy ASCII VTK PolyData file with optional per-vertex scalars (ParaView, 3D Slicer)."""
    path = Path(path)
    n, m = mesh.n_vertices, mesh.n_faces
    out = ["# vtk DataFile Version 3.0", f"stl-deviation {mesh.name}", "ASCII", "DATASET POLYDATA"]
    out.append(f"POINTS {n} double")
    out.extend(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in mesh.vertices)
    out.append(f"POLYGONS {m} {4 * m}")
    out.extend(f"3 {a} {b} {c}" for a, b, c in mesh.faces)
    if point_data:
        out.append(f"POINT_DATA {n}")
        for key, values in point_data.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (n,):
                raise ValueError(f"point data '{key}' must have shape ({n},)")
            out.append(f"SCALARS {key} double 1")
            out.append("LOOKUP_TABLE default")
            out.extend(f"{v:.9g}" for v in values)
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_ply(path, mesh: Mesh, colors: np.ndarray, quality: Optional[np.ndarray] = None) -> None:
    """Write an ASCII PLY with per-vertex RGB colours (MeshLab, Blender).

    ``quality`` is stored as a ``quality`` vertex property, which MeshLab imports as vertex quality.
    """
    path = Path(path)
    n, m = mesh.n_vertices, mesh.n_faces
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != (n, 3):
        raise ValueError(f"colors must have shape ({n}, 3)")
    head = ["ply", "format ascii 1.0", f"comment stl-deviation {mesh.name}", f"element vertex {n}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue"]
    if quality is not None:
        head.append("property float quality")
    head += [f"element face {m}", "property list uchar int vertex_indices", "end_header"]
    out = list(head)
    for i, (x, y, z) in enumerate(mesh.vertices):
        r, g, b = colors[i]
        row = f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}"
        if quality is not None:
            row += f" {float(quality[i]):.9g}"
        out.append(row)
    out.extend(f"3 {a} {b} {c}" for a, b, c in mesh.faces)
    path.write_text("\n".join(out) + "\n", encoding="ascii")
