"""AABB broadphase: OptimizeGraph segments vs obstacles and vs other groups.

Hermite → Bézier-control AABB via ``hermite_ends`` / ``seg_ends`` (stored u→v);
R-tree (3D) for queries. Narrowphase / active-set stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rtree import index as rtree_index

from pipe_smooth.geometry.hermite import hermite_bezier_controls
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.collision.obstacles import AabbObs, CapsuleObs, SphereObs

Obstacle = SphereObs | CapsuleObs | AabbObs


@dataclass(frozen=True)
class BroadObsPair:
    """Broadphase hit: one Hermite segment vs one static obstacle."""

    group_id: str
    seg_id: int
    obs_name: str


@dataclass(frozen=True)
class BroadGrpPair:
    """Broadphase hit: Hermite segments from two different groups (i < j)."""

    group_a: str
    seg_a: int
    group_b: str
    seg_b: int


def hermite_aabb(p0, v0, p1, v1, r: float = 0.0, margin: float = 0.0):
    """AABB of one cubic Hermite segment (Bézier controls ⊕ ball r).

    Args:
        p0, v0, p1, v1: Hermite ends; v is the end derivative γ'
        r: tube radius pad
        margin: extra pad (same δ_margin as clearance d_req)

    Returns:
        mn, mx: shape (3,) each
    """
    C = hermite_bezier_controls(p0, v0, p1, v1, backend="numpy")
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
    graphs = list(graphs)
    obs_pairs: list[BroadObsPair] = []
    grp_pairs: list[BroadGrpPair] = []
    if not graphs:
        return obs_pairs, grp_pairs

    # --- segment boxes per group: [(seg_id, mn, mx), ...] ---
    boxes: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    for g in graphs:
        ends = np.asarray(
            g.seg_ends if g.seg_ends is not None else g.hermite_ends(backend="numpy"),
            dtype=float,
        )
        rad = np.asarray(g.radius, dtype=float)
        segs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for sid, seg in enumerate(g.segments):
            p0, v0, p1, v1 = ends[sid]
            r = max(float(rad[seg.u]), float(rad[seg.v]))
            mn, mx = hermite_aabb(p0, v0, p1, v1, r=r, margin=margin)
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
