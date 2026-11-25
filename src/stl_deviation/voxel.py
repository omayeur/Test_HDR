"""Voxel overlap (Dice, Jaccard) between two closed surfaces, by scan-line voxelisation along z.

For thin structures (an orbital floor, a membrane) Dice depends strongly on the voxel size;
prefer the surface-distance metrics there and always report the voxel size with the Dice.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .mesh import Mesh

_JITTER_X = 0.0123456   # rays are shifted by a tiny fraction of a voxel so they never hit
_JITTER_Y = 0.0271828   # a vertex or an edge exactly (which would corrupt the parity count)


def common_grid(meshes, voxel_size: Optional[float] = None, max_voxels: int = 15_000_000, margin: int = 2):
    """Grid (origin, voxel size, shape) enclosing all meshes."""
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    extent = hi - lo
    size = float(voxel_size) if voxel_size else float(extent.max() / 150.0)
    if size <= 0:
        raise ValueError("voxel size must be positive")
    while True:
        shape = np.ceil(extent / size).astype(int) + 2 * margin
        if int(np.prod(shape)) <= max_voxels:
            break
        size *= 1.1
    origin = lo - margin * size
    return origin, size, tuple(int(s) for s in shape)


def voxelize(mesh: Mesh, origin, size: float, shape: Tuple[int, int, int]):
    """Boolean occupancy grid of a closed mesh. Returns (grid, number of ambiguous columns dropped)."""
    nx, ny, nz = shape
    ox, oy, oz = (float(v) for v in origin)
    tri = mesh.triangles
    x = (tri[:, :, 0] - ox) / size - 0.5 - _JITTER_X
    y = (tri[:, :, 1] - oy) / size - 0.5 - _JITTER_Y
    ix0 = np.maximum(np.ceil(x.min(axis=1)).astype(np.int64), 0)
    ix1 = np.minimum(np.floor(x.max(axis=1)).astype(np.int64), nx - 1)
    iy0 = np.maximum(np.ceil(y.min(axis=1)).astype(np.int64), 0)
    iy1 = np.minimum(np.floor(y.max(axis=1)).astype(np.int64), ny - 1)
    cx = np.maximum(ix1 - ix0 + 1, 0)
    cy = np.maximum(iy1 - iy0 + 1, 0)
    cells = cx * cy

    cols, zs = [], []
    order = np.flatnonzero(cells > 0)
    cum = np.cumsum(cells[order])
    budget = 4_000_000
    start = 0
    while start < len(order):
        base = cum[start - 1] if start else 0
        stop = int(np.searchsorted(cum, base + budget, side="right"))
        stop = max(stop, start + 1)
        sel = order[start:stop]
        start = stop

        reps = cells[sel]
        t_idx = np.repeat(sel, reps)
        offs = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        local_cx = cx[t_idx]
        ix = ix0[t_idx] + offs % local_cx
        iy = iy0[t_idx] + offs // local_cx

        px = ox + (ix + 0.5 + _JITTER_X) * size
        py = oy + (iy + 0.5 + _JITTER_Y) * size
        t = tri[t_idx]
        x0, y0, z0 = t[:, 0, 0], t[:, 0, 1], t[:, 0, 2]
        x1, y1, z1 = t[:, 1, 0], t[:, 1, 1], t[:, 1, 2]
        x2, y2, z2 = t[:, 2, 0], t[:, 2, 1], t[:, 2, 2]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        ok = np.abs(area2) > 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area2
            w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area2
            w0 = np.where(ok, w0, -1.0)   # vertical triangles never intersect a z-ray
            w1 = np.where(ok, w1, -1.0)
            w2 = 1.0 - w0 - w1
        inside = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        cols.append((ix * ny + iy)[inside])
        zs.append((w0 * z0 + w1 * z1 + w2 * z2)[inside])

    grid = np.zeros(shape, dtype=bool)
    if not cols:
        return grid, 0
    cols = np.concatenate(cols)
    zs = np.concatenate(zs)
    srt = np.lexsort((zs, cols))
    cols, zs = cols[srt], zs[srt]
    counts = np.bincount(cols, minlength=nx * ny)
    first = np.cumsum(counts) - counts
    rank = np.arange(len(cols)) - first[cols]
    even = counts[cols] % 2 == 0
    dropped = int(np.count_nonzero((counts % 2) == 1))
    enter = np.flatnonzero((rank % 2 == 0) & even)
    z_in, z_out, col = zs[enter], zs[enter + 1], cols[enter]
    k0 = np.maximum(np.ceil((z_in - oz) / size - 0.5).astype(np.int64), 0)
    k1 = np.minimum(np.floor((z_out - oz) / size - 0.5).astype(np.int64), nz - 1)
    keep = k1 >= k0
    col, k0, k1 = col[keep], k0[keep], k1[keep]
    width = nz + 1
    diff = np.bincount(col * width + k0, minlength=nx * ny * width).astype(np.int32)
    diff -= np.bincount(col * width + k1 + 1, minlength=nx * ny * width).astype(np.int32)
    filled = np.cumsum(diff.reshape(nx * ny, width), axis=1)[:, :nz] > 0
    return filled.reshape(shape), dropped


def overlap_metrics(a: Mesh, b: Mesh, voxel_size: Optional[float] = None) -> dict:
    """Dice and Jaccard between the volumes enclosed by two closed meshes."""
    if not (a.is_closed() and b.is_closed()):
        raise ValueError("Dice/Jaccard need two closed, consistently oriented meshes")
    origin, size, shape = common_grid([a, b], voxel_size)
    ga, drop_a = voxelize(a, origin, size, shape)
    gb, drop_b = voxelize(b, origin, size, shape)
    na, nb = int(ga.sum()), int(gb.sum())
    inter = int((ga & gb).sum())
    union = na + nb - inter
    return {
        "dice": 2.0 * inter / (na + nb) if (na + nb) else float("nan"),
        "jaccard": inter / union if union else float("nan"),
        "voxel_size": size,
        "grid_shape": list(shape),
        "volume_a_voxels": na * size ** 3,
        "volume_b_voxels": nb * size ** 3,
        "ambiguous_columns_dropped": drop_a + drop_b,
    }
