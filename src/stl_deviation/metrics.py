"""Segmentation-style surface metrics computed from a :class:`DeviationResult`."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .deviation import DeviationResult, DirectionalDeviation
from .mesh import Mesh


def _directional(dev: DirectionalDeviation) -> Dict[str, float]:
    s = dev.sample_distances
    absd = np.abs(s)
    stats = {
        "mean_abs": float(absd.mean()),
        "median_abs": float(np.median(absd)),
        "rms": float(np.sqrt((s ** 2).mean())),
        "p95_abs": float(np.percentile(absd, 95)),
        # the maximum also looks at vertices, so isolated spikes are not missed by sampling
        "max_abs": float(max(absd.max(), np.abs(dev.vertex_distances).max())),
    }
    if dev.signed:
        stats.update({
            "mean_signed": float(s.mean()),
            "std_signed": float(s.std()),
            "p05_signed": float(np.percentile(s, 5)),
            "p95_signed": float(np.percentile(s, 95)),
        })
    return stats


def _mesh_info(mesh: Mesh) -> Dict[str, object]:
    closed = mesh.is_closed()
    info: Dict[str, object] = {
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "area": mesh.area(),
        "closed": closed,
    }
    info["volume"] = abs(mesh.signed_volume()) if closed else None
    return info


def summarize(result: DeviationResult, tolerances: Sequence[float] = (0.5, 1.0, 2.0), units: str = "mm") -> dict:
    """Collect the metrics of a deviation analysis in a plain, JSON-friendly dictionary.

    Definitions (d = distance from a sampled point of one surface to the other surface):
      * ASSD  - average symmetric surface distance: mean of the two directional mean |d|.
      * RMSD  - root-mean-square symmetric distance.
      * HD    - Hausdorff distance: largest |d| over both directions (samples and vertices).
      * HD95  - the larger of the two directional 95th percentiles of |d|.
      * within_tolerance - share of samples (both directions pooled) with |d| <= tolerance.
      * mean_signed (A->B) > 0 means A is on average outside B, i.e. A is "larger" than B.
    """
    a_to_b = _directional(result.a_to_b)
    b_to_a = _directional(result.b_to_a)
    ab = np.abs(result.a_to_b.sample_distances)
    ba = np.abs(result.b_to_a.sample_distances)
    pooled = np.concatenate([ab, ba])

    info_a, info_b = _mesh_info(result.a), _mesh_info(result.b)
    metrics: dict = {
        "units": units,
        "n_samples_per_direction": result.n_samples,
        "seed": result.seed,
        "mesh_a": info_a,
        "mesh_b": info_b,
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "symmetric": {
            "assd": 0.5 * (a_to_b["mean_abs"] + b_to_a["mean_abs"]),
            "rmsd": float(np.sqrt(0.5 * (np.mean(ab ** 2) + np.mean(ba ** 2)))),
            "hd": max(a_to_b["max_abs"], b_to_a["max_abs"]),
            "hd95": max(a_to_b["p95_abs"], b_to_a["p95_abs"]),
            "within_tolerance": {f"{t:g}": float((pooled <= t).mean()) for t in tolerances},
        },
    }
    if info_a["volume"] is not None and info_b["volume"] is not None:
        va, vb = float(info_a["volume"]), float(info_b["volume"])
        metrics["volume"] = {
            "a": va,
            "b": vb,
            "difference_a_minus_b": va - vb,
            "relative_difference_pct": 100.0 * (va - vb) / vb if vb else None,
        }
    warnings = []
    if not result.a_to_b.signed:
        warnings.append("Mesh B is not a closed oriented surface: A->B distances are unsigned.")
    if not result.b_to_a.signed:
        warnings.append("Mesh A is not a closed oriented surface: B->A distances are unsigned.")
    if warnings:
        metrics["warnings"] = warnings
    return metrics
