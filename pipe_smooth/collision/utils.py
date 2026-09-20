"""AABB broadphase: OptimizeGraph segments vs obstacles and vs other groups.

Hermite → Bézier-control AABB (params from OptimizeUsagePath, same as sampling);
R-tree (3D) for queries. Narrowphase / active-set stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rtree import index as rtree_index

from pipe_smooth.geometry.hermite import hermite_bezier_controls
from pipe_smooth.geometry.assemble import OptimizeGraph, OptimizeUsagePath
from pipe_smooth.collision.obstacles import AabbObs, CapsuleObs, SphereObs

Obstacle = SphereObs | CapsuleObs | AabbObs


@dataclass(frozen=True)
class BroadObsPair:
    """Broadphase hit: one Hermite segment vs one static obstacle."""

    group_id: str
    seg_id: str
    obs_name: str


@dataclass(frozen=True)
class BroadGrpPair:
    """Broadphase hit: Hermite segments from two different groups (i < j)."""

    group_a: str
    seg_a: str
    group_b: str
    seg_b: str


def hermite_aabb(p0, v0, s0, p1, v1, s1, r: float = 0.0, margin: float = 0.0):
    """AABB of one cubic Hermite segment (Bézier controls ⊕ ball r).

    Args:
        p0, v0, s0, p1, v1, s1: Hermite ends (v unit, s > 0)
        r: tube radius pad
        margin: extra pad (e.g. δ_buf)

    Returns:
        mn, mx: shape (3,) each
    """
    C = hermite_bezier_controls(p0, v0, s0, p1, v1, s1, backend="numpy")
    e = float(r) + float(margin)
    return C.min(axis=0) - e, C.max(axis=0) + e


def find_broadphase_pairs(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle] = (),
    margin: float = 0.0,
) -> tuple[list[BroadObsPair], list[BroadGrpPair]]:
    """Coarse pairs: each group's segments vs obstacles, and cross-group segments.

    Args:
        graphs: sequence of OptimizeGraph (one per pipe group)
        obstacles: sequence of SphereObs | CapsuleObs | AabbObs
        margin: pad on segment and obstacle AABBs

    Returns:
        obs_pairs: list of BroadObsPair
        grp_pairs: list of BroadGrpPair (group index i < j only; no same-group pairs)
    """
    
    def _usage_seg_hermite(
        graph: OptimizeGraph, path: OptimizeUsagePath, i: int
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, float]:
        """Hermite ends for path.segments[i] (same convention as sample_usage_path)."""
        sid, rev = path.segments[i], path.reverse[i]
        seg = graph.segments[sid]
        nu, nv = graph.nodes[seg.u], graph.nodes[seg.v]
        if rev:
            p0, v0, s0 = nv.pos, -nv.v, nv.sigma  # type: ignore[operator]
            p1, v1, s1 = nu.pos, -nu.v, nu.sigma  # type: ignore[operator]
            r = max(float(nv.radius), float(nu.radius))
        else:
            p0, v0, s0 = nu.pos, nu.v, nu.sigma
            p1, v1, s1 = nv.pos, nv.v, nv.sigma
            r = max(float(nu.radius), float(nv.radius))
        nseg = len(path.segments)
        if i == 0:
            v0, s0 = path.v_start, path.sigma_start
        if i == nseg - 1:
            v1, s1 = path.v_end, path.sigma_end
        return (
            np.asarray(p0, float),
            np.asarray(v0, float),
            float(s0),  # type: ignore[arg-type]
            np.asarray(p1, float),
            np.asarray(v1, float),
            float(s1),  # type: ignore[arg-type]
            r,
        )
    
    graphs = list(graphs)
    obs_pairs: list[BroadObsPair] = []
    grp_pairs: list[BroadGrpPair] = []
    if not graphs:
        return obs_pairs, grp_pairs

    # --- segment boxes per group: [(seg_id, mn, mx), ...] via usage_paths ---
    boxes: list[list[tuple[str, np.ndarray, np.ndarray]]] = []
    for g in graphs:
        segs: list[tuple[str, np.ndarray, np.ndarray]] = []
        seen: set[str] = set()
        for path in g.usage_paths:
            for i, sid in enumerate(path.segments):
                if sid in seen:
                    continue
                seen.add(sid)
                p0, v0, s0, p1, v1, s1, r = _usage_seg_hermite(g, path, i)
                mn, mx = hermite_aabb(p0, v0, s0, p1, v1, s1, r=r, margin=margin)
                segs.append((sid, mn, mx))  # type: ignore[arg-type]
        boxes.append(segs)

    # --- vs static obstacles ---
    if obstacles:
        prop = rtree_index.Property()
        prop.dimension = 3
        obs_tree = rtree_index.Index(properties=prop)
        obs_names = []
        for j, obs in enumerate(obstacles):
            obs_names.append(obs.name)
            mn, mx = obs.aabb(margin)
            obs_tree.insert(j, (*mn.tolist(), *mx.tolist()))
        for g, segs in zip(graphs, boxes):
            for sid, mn, mx in segs:
                for j in obs_tree.intersection((*mn.tolist(), *mx.tolist())):
                    obs_pairs.append(BroadObsPair(g.id, sid, obs_names[int(j)]))

    # --- cross-group segment–segment ---
    for i in range(len(graphs)):
        if not boxes[i]:
            continue
        prop = rtree_index.Property()
        prop.dimension = 3
        tree = rtree_index.Index(properties=prop)
        for k, (sid, mn, mx) in enumerate(boxes[i]):
            tree.insert(k, (*mn.tolist(), *mx.tolist()))
        sids_i = [sid for sid, _, __ in boxes[i]]
        for j in range(i + 1, len(graphs)):
            for sid_b, mn, mx in boxes[j]:
                for k in tree.intersection((*mn.tolist(), *mx.tolist())):
                    grp_pairs.append(
                        BroadGrpPair(graphs[i].id, sids_i[k], graphs[j].id, sid_b)
                    )

    return obs_pairs, grp_pairs
