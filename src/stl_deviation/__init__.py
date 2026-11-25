"""stl-deviation: quantify and visualise the geometric deviation between two STL surfaces."""
from .align import centroid_alignment, icp_rigid
from .deviation import DeviationResult, compute_deviation
from .io import read_stl, write_ply, write_stl, write_vtk
from .mesh import Mesh
from .metrics import summarize
from .voxel import overlap_metrics

__version__ = "0.1.0"

__all__ = [
    "Mesh", "read_stl", "write_stl", "write_vtk", "write_ply",
    "compute_deviation", "DeviationResult", "summarize", "overlap_metrics",
    "centroid_alignment", "icp_rigid", "__version__",
]
