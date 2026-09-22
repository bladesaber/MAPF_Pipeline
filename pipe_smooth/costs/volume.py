"""Sphere-union volume via voxel occupancy (overlap counted once)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree


def tube_volume(xyz, radius, voxel: float = 0.25) -> float:
    """Solid volume ≈ voxel union of balls centered at ``xyz`` with ``radius``.

    Voxel centers on the global AABB; ``KDTree.query_radius`` marks spheres
    (per-point ``r``). Overlap counted once via unique indices.
    """
    if voxel <= 0:
        raise ValueError(f"voxel must be > 0, got {voxel}")
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    radius = np.asarray(radius, float)
    if radius.ndim == 0:
        radius = np.full(xyz.shape[0], float(radius))
    if xyz.shape[0] == 0:
        return 0.0
    if radius.shape[0] != xyz.shape[0]:
        raise ValueError(f"radius length {radius.shape[0]} != xyz N {xyz.shape[0]}")

    rmax = float(radius.max())
    mn = xyz.min(axis=0) - rmax
    mx = xyz.max(axis=0) + rmax
    nx, ny, nz = np.maximum(1, np.ceil((mx - mn) / voxel).astype(int))
    xs = mn[0] + (np.arange(int(nx)) + 0.5) * voxel
    ys = mn[1] + (np.arange(int(ny)) + 0.5) * voxel
    zs = mn[2] + (np.arange(int(nz)) + 0.5) * voxel
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    hits = KDTree(centers).query_radius(xyz, r=radius)
    parts = [h for h in hits if h.size]
    if not parts:
        return 0.0
    return float(np.unique(np.concatenate(parts)).size) * (voxel**3)
