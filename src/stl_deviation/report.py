"""Colour maps, JSON/CSV reports, text summary and the histogram figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .deviation import DeviationResult

# Diverging blue <-> red with a neutral light-grey midpoint (blue = inside, red = outside).
_NEG = np.array([37, 125, 224], dtype=float)
_MID = np.array([239, 239, 239], dtype=float)
_POS = np.array([236, 70, 71], dtype=float)


def diverging_colors(values: np.ndarray, vrange: float) -> np.ndarray:
    """Map values in [-vrange, +vrange] to uint8 RGB (values outside are clipped)."""
    t = np.clip(np.asarray(values, dtype=float) / vrange, -1.0, 1.0)[:, None]
    rgb = np.where(t < 0, _MID + (-t) * (_NEG - _MID), _MID + t * (_POS - _MID))
    return np.rint(rgb).astype(np.uint8)


def sequential_colors(values: np.ndarray, vmax: float) -> np.ndarray:
    """Light-grey to red ramp for unsigned distances (open surfaces)."""
    t = np.clip(np.asarray(values, dtype=float) / vmax, 0.0, 1.0)[:, None]
    return np.rint(_MID + t * (_POS - _MID)).astype(np.uint8)


def auto_range(result: DeviationResult) -> float:
    """Default colour range: 95th percentile of |vertex distance| over both surfaces, rounded up."""
    both = np.concatenate([np.abs(result.a_to_b.vertex_distances), np.abs(result.b_to_a.vertex_distances)])
    p95 = float(np.percentile(both, 95))
    if p95 <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(p95))
    return float(np.ceil(p95 / magnitude * 2) / 2 * magnitude)


# --------------------------------------------------------------------- reports
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path, metrics: dict) -> None:
    Path(path).write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten(metrics: dict, prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=name + "."))
        elif isinstance(value, (list, tuple)):
            flat[name] = ";".join(str(v) for v in value)
        else:
            flat[name] = value
    return flat


def write_csv(path, metrics: dict, label: Optional[str] = None) -> None:
    """One-row CSV (header + values); handy to stack many comparisons in a spreadsheet."""
    flat = flatten(_jsonable(metrics))
    if label:
        flat = {"label": label, **flat}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(flat.keys()))
        writer.writerow(list(flat.values()))


def format_summary(metrics: dict, name_a: str, name_b: str) -> str:
    u = metrics["units"]
    s = metrics["symmetric"]
    ab, ba = metrics["a_to_b"], metrics["b_to_a"]
    lines = [
        f"Surface deviation: {name_a} (A) vs {name_b} (B)   [units: {u}]",
        "",
        f"  ASSD  (mean symmetric distance) : {s['assd']:.3f} {u}",
        f"  RMSD                            : {s['rmsd']:.3f} {u}",
        f"  HD95                            : {s['hd95']:.3f} {u}",
        f"  Hausdorff (max)                 : {s['hd']:.3f} {u}",
    ]
    for tol, frac in s["within_tolerance"].items():
        lines.append(f"  within {tol:>5} {u:<2}                : {100 * frac:5.1f} % of the surface")
    if "mean_signed" in ab:
        lines.append(f"  bias A->B (mean signed)         : {ab['mean_signed']:+.3f} {u}  "
                     f"({'A larger than B' if ab['mean_signed'] > 0 else 'A smaller than B'} on average)")
    if "volume" in metrics:
        v = metrics["volume"]
        lines.append(f"  volume A / B                    : {v['a']:.1f} / {v['b']:.1f} {u}^3 "
                     f"({v['relative_difference_pct']:+.1f} %)")
    if "overlap" in metrics:
        o = metrics["overlap"]
        lines.append(f"  Dice / Jaccard                  : {o['dice']:.3f} / {o['jaccard']:.3f} "
                     f"(voxel {o['voxel_size']:.3g} {u})")
    for w in metrics.get("warnings", []):
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- figure
def plot_histograms(result: DeviationResult, path, name_a: str, name_b: str,
                    tolerances=(), units: str = "mm") -> bool:
    """Save a two-panel figure (signed-distance histogram, cumulative share). Returns False if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ink, ink2, grid = "#191919", "#666666", "#e7e7e7"
    c_ab, c_ba = "#257de0", "#f47230"
    series = [
        (result.a_to_b, f"{name_a} → {name_b}", c_ab),
        (result.b_to_a, f"{name_b} → {name_a}", c_ba),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor("white")

    allv = np.concatenate([d.sample_distances for d, _, _ in series])
    lo, hi = np.percentile(allv, [0.5, 99.5])
    pad = 0.05 * max(hi - lo, 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 80)
    for dev, label, color in series:
        # values beyond the 0.5-99.5 % window are left out of the drawing (they stay in the metrics)
        ax1.hist(dev.sample_distances, bins=bins, density=True,
                 histtype="stepfilled", alpha=0.35, color=color, linewidth=0)
        ax1.hist(dev.sample_distances, bins=bins, density=True,
                 histtype="step", color=color, linewidth=1.8, label=label)
    if all(d.signed for d, _, _ in series):
        ax1.axvline(0, color=ink2, linewidth=1)
        ax1.set_xlabel(f"signed distance ({units}); negative = inside the other surface")
    else:
        ax1.set_xlabel(f"distance ({units}); open surface, unsigned")
    ax1.set_ylabel("share of the surface (density)")
    ax1.set_title("Distribution of the deviation", loc="left", color=ink, fontsize=11)

    xmax = float(np.percentile(np.abs(allv), 99.9))
    xs = np.linspace(0, xmax, 300)
    for dev, label, color in series:
        a = np.sort(np.abs(dev.sample_distances))
        ax2.plot(xs, np.searchsorted(a, xs, side="right") / len(a) * 100, color=color, linewidth=2, label=label)
    for t in tolerances:
        if t <= xmax:
            ax2.axvline(t, color=ink2, linewidth=0.8, linestyle=":")
            ax2.text(t, 2, f" {t:g}", color=ink2, fontsize=8, va="bottom")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel(f"distance threshold ({units})")
    ax2.set_ylabel("% of the surface within the threshold")
    ax2.set_title("Cumulative agreement", loc="left", color=ink, fontsize=11)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c2c2c2")
        ax.tick_params(colors=ink2, labelsize=9)
        ax.xaxis.label.set_color(ink2)
        ax.yaxis.label.set_color(ink2)
    ax1.legend(frameon=False, fontsize=9, labelcolor=ink2, loc="upper right")
    ax2.legend(frameon=False, fontsize=9, labelcolor=ink2, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return True
