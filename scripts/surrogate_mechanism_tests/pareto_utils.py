from __future__ import annotations

import numpy as np


def pareto_dominated(points: np.ndarray) -> np.ndarray:
    """Return dominated flags for minimization points."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated[i] = True
                break
    return dominated


def dominated_hypervolume_grid(points: np.ndarray, reference: np.ndarray | None = None, grid_size: int = 80) -> float:
    """Approximate dominated hypervolume for 2D/3D minimization points on a fixed grid."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in {2, 3}:
        raise ValueError("Hypervolume grid approximation supports 2D or 3D points")
    if reference is None:
        reference = np.full(pts.shape[1], 1.05, dtype=float)
    ref = np.asarray(reference, dtype=float)
    mins = np.zeros(pts.shape[1], dtype=float)
    axes = [np.linspace(mins[d], ref[d], grid_size) for d in range(pts.shape[1])]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, pts.shape[1])
    dominated = np.zeros(mesh.shape[0], dtype=bool)
    for p in pts:
        dominated |= np.all(mesh >= p, axis=1)
    return float(dominated.mean() * np.prod(ref - mins))


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    denom = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / denom
