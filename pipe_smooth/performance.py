"""Stage-1 performance reports (§8.1): clearance / path / solve."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from pipe_smooth.collision import (
    AabbObs,
    CapsuleObs,
    SphereObs,
    find_broadphase_pairs,
    find_narrowphase_pairs,
)
from pipe_smooth.costs.path import kappa_menger
from pipe_smooth.costs.volume import tube_volume
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.recorder import Recorder

try:
    from pipe_smooth.collision import find_hull_narrowphase_pairs
except ImportError:  # coal optional
    find_hull_narrowphase_pairs = None  # type: ignore[assignment]

Obstacle = SphereObs | CapsuleObs | AabbObs
ClearKind = Literal["capsule", "hull", "ignore"]


def refresh_capsule(graphs, obstacles, margin: float):
    broad_obs, broad_grp = find_broadphase_pairs(graphs, obstacles, margin=margin)
    return find_narrowphase_pairs(
        graphs, obstacles, broad_obs, broad_grp, margin=margin
    )


def refresh_hull(graphs, obstacles, margin: float):
    if find_hull_narrowphase_pairs is None:
        raise ImportError("hull narrowphase needs coal (pipe_smooth.collision.gjk)")
    broad_obs, broad_grp = find_broadphase_pairs(graphs, obstacles, margin=margin)
    return find_hull_narrowphase_pairs(
        graphs, obstacles, broad_obs, broad_grp, margin=margin
    )


def clearance_report(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    *,
    kind: ClearKind = "hull",
    margin: float = 0.0,
) -> dict[str, Any]:
    """Hard clearance report (§8.1.1). Same proxy as the loop; strip soft ``margin``.

    Soft loop uses ``d_req = radii + margin``. Feasibility uses
    ``d_req_hard = d_req - margin``; ``n_hit`` / ``ok`` count only hard violations
    (``d_req_hard - d_geom > 0``). ``max_shortfall`` is that max (0 if none).
    ``hard_obs`` / ``hard_grp`` are JSON-friendly ``{pair, shortfall}`` via ``.export()``.
    ``kind='ignore'`` skips detection (debug).
    """
    if kind == "ignore":
        return {
            "kind": kind,
            "margin": float(margin),
            "n_hit": 0,
            "n_hit_obs": 0,
            "n_hit_grp": 0,
            "max_shortfall": 0.0,
            "ok": True,
            "hard_obs": [],
            "hard_grp": [],
        }

    if kind == "hull":
        narrow_obs, narrow_grp = refresh_hull(graphs, obstacles, margin)
    else:
        narrow_obs, narrow_grp = refresh_capsule(graphs, obstacles, margin)

    def hard_pairs(pairs):
        out = []
        for p in pairs:
            d_req = getattr(p, "d_req", None)
            if d_req is None:
                d_req = p.hits[0].d_req
            s = (float(d_req) - float(margin)) - float(p.d_geom)
            if s > 0.0:
                out.append({"pair": p.export(), "shortfall": float(s)})
        return out

    hard_obs, hard_grp = hard_pairs(narrow_obs), hard_pairs(narrow_grp)
    hard = hard_obs + hard_grp
    max_sf = max((h["shortfall"] for h in hard), default=0.0)
    return {
        "kind": kind,
        "margin": float(margin),
        "n_hit": len(hard),
        "n_hit_obs": len(hard_obs),
        "n_hit_grp": len(hard_grp),
        "max_shortfall": max_sf,
        "ok": len(hard) == 0,
        "hard_obs": hard_obs,
        "hard_grp": hard_grp,
    }


def path_report(
    graphs: Sequence[OptimizeGraph],
    *,
    voxel: float = 0.25,
) -> dict[str, Any]:
    """Usage-path metrics (§8.1.2): per-usage → group → aggregate.

    ``L`` / ``kr_max`` (= max κ·r) from each usage polyline. ``V`` is voxel-union of
    sample spheres (segments once) at group / aggregate.
    """
    groups: dict[str, Any] = {}
    for g in graphs:
        if g.seg_xyz is None or g.seg_radius is None:
            g.refresh(backend="numpy", store=True)
        seg_xyz = np.asarray(g.seg_xyz, dtype=float)
        seg_radius = np.asarray(g.seg_radius, dtype=float)
        usages: dict[str, Any] = {}
        seen_seg: set[int] = set()
        g_xyz, g_r = [], []
        for path in g.usage_paths:
            pts, rads = [], []
            for sid, rev in zip(path.segments, path.reverse):
                xyz = seg_xyz[sid]
                radius = seg_radius[sid]
                if sid not in seen_seg:
                    seen_seg.add(sid)
                    g_xyz.append(xyz)
                    g_r.append(radius)
                if rev:
                    xyz, radius = xyz[::-1], radius[::-1]
                if pts:
                    xyz, radius = xyz[1:], radius[1:]
                pts.append(xyz)
                rads.append(radius)
            if not pts:
                raise ValueError(f"path has no samples")
            xyz = np.concatenate(pts, axis=0)
            radius = np.concatenate(rads, axis=0)
            kappa = np.asarray(
                kappa_menger(
                    xyz[None],
                    t_start=np.asarray(path.v_start)[None],
                    t_end=np.asarray(path.v_end)[None],
                )
            )[0]
            key = f"{path.usage[0]}->{path.usage[1]}"
            usages[key] = {
                "L": float(np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1).sum()),
                "kr_max": float(np.max(kappa * radius)) if kappa.size else 0.0,
                "kappa": kappa.tolist(),
            }
        g_xyz = np.concatenate(g_xyz, axis=0) if g_xyz else np.zeros((0, 3))
        g_r = np.concatenate(g_r, axis=0) if g_r else np.zeros((0,))
        groups[g.id] = {
            "usages": usages,
            "L": sum(u["L"] for u in usages.values()),
            "kr_max": max((u["kr_max"] for u in usages.values()), default=0.0),
            "V": tube_volume(g_xyz, g_r, voxel=voxel),
        }
    agg = {
        "L": sum(g["L"] for g in groups.values()),
        "kr_max": max((g["kr_max"] for g in groups.values()), default=0.0),
        "V": sum(g["V"] for g in groups.values()),
    }
    return {"groups": groups, "aggregate": agg}


def solve_report(
    recorder: Recorder,
    *,
    wall_time: float | None = None,
    late_frac: float = 0.25,
) -> dict[str, Any]:
    """Solve-process metrics (§8.1.3): time, steps, J drop / late stability."""
    J = recorder.J_curve()
    n = int(J.size)
    if n == 0:
        raise ValueError("recorder has no J values")
    J0, JT = float(J[0]), float(J[-1])
    i0 = max(0, int((1.0 - late_frac) * n))
    late = J[i0:]
    dJ = np.diff(J)
    return {
        "wall_time": wall_time,
        "steps": n,
        "J0": J0,
        "JT": JT,
        "best_J": float(recorder.best_J),
        "rel_drop": (J0 - JT) / max(abs(J0), 1e-12),
        "late_std": float(np.std(late)),
        "uphill_frac": float(np.mean(dJ > 0.0)) if dJ.size else 0.0,
    }


def performance_report(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    recorder: Recorder | None = None,
    *,
    clear_kind: ClearKind = "capsule",
    margin: float = 0.0,
    wall_time: float | None = None,
) -> dict[str, Any]:
    """Clearance + path + solve reports on current graphs."""
    for g in graphs:
        g.refresh(backend="numpy", store=True)
    clear = clearance_report(graphs, obstacles, kind=clear_kind, margin=margin)
    path = path_report(graphs)
    solve = (
        solve_report(recorder, wall_time=wall_time) if recorder is not None else None
    )
    return {"clearance": clear, "path": path, "solve": solve}
