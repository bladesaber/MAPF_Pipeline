"""Built-in optimize-bench scenarios (graphs ± obstacles)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pipe_smooth.collision import SphereObs
from pipe_smooth.geometry.assemble import (
    HermiteNode,
    HermiteSegment,
    OptimizeGraph,
    OptimizeUsagePath,
    assemble_hermite_graph,
)
from pipe_smooth.topology.junction import PartitionedTree, build_junctions
from pipe_smooth.topology.tree import GroupTree, build_tree


def _polyline_graph(
    gid: str,
    positions: np.ndarray,
    radius: float,
    usage: tuple[str, str],
    v_start,
    v_end,
    *,
    corner_v: dict[int, np.ndarray] | None = None,
) -> OptimizeGraph:
    """SoA OptimizeGraph along a polyline; ends fixed as ports."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    N = int(pos.shape[0])
    if N < 2:
        raise ValueError("need >= 2 waypoints")
    rad = np.full(N, float(radius))
    vel = np.zeros((N, 3))
    nodes = [
        HermiteNode(
            fixed_p=(i == 0 or i == N - 1),
            fixed_v=(i == 0 or i == N - 1),
            role="port" if (i == 0 or i == N - 1) else "trunk",
        )
        for i in range(N)
    ]
    for i in range(N - 1):
        d = pos[i + 1] - pos[i]
        L = float(np.linalg.norm(d))
        tang = d / (L + 1e-12)
        if not nodes[i].fixed_v:
            vel[i] = tang * L
        if not nodes[i + 1].fixed_v:
            vel[i + 1] = tang * L
    if corner_v:
        for i, vv in corner_v.items():
            if not nodes[i].fixed_v:
                vel[i] = np.asarray(vv, float).reshape(3)
    S = N - 1
    segs = [HermiteSegment(edge="e0", kind="trunk", u=i, v=i + 1) for i in range(S)]
    g = OptimizeGraph(
        id=gid,
        nodes=nodes,
        segments=segs,
        pos=pos,
        v=vel,
        radius=rad,
        u_sign=np.ones(S),
        v_sign=np.ones(S),
        edge_segs={"e0": list(range(S))},
        usage_paths=[
            OptimizeUsagePath(
                usage=usage,
                edges=["e0"],
                edges_reverse=[False],
                segments=list(range(S)),
                reverse=[False] * S,
                v_start=np.asarray(v_start, float).reshape(3),
                v_end=np.asarray(v_end, float).reshape(3),
            )
        ],
    )
    g.check_usage_tangents_colinear()
    return g.bind_ends_tables()


def make_elbow90(count: int = 5, resolution: float = 1.0, radius: float = 0.5) -> OptimizeGraph:
    """L-path: ``count`` waypoints along +x, then ``count`` along +y (shared corner)."""
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    sp = float(resolution)
    pts = [[i * sp, 0.0, 0.0] for i in range(count)]
    pts += [[(count - 1) * sp, j * sp, 0.0] for j in range(1, count)]
    corner = count - 1
    return _polyline_graph(
        "elbow_90",
        np.asarray(pts, float),
        radius,
        ("A", "B"),
        [sp, 0.0, 0.0],
        [0.0, sp, 0.0],
        corner_v={corner: (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)) * sp},
    )


def make_s_curve(
    count: int = 5,
    resolution: float = 1.0,
    radius: float = 0.5,
    obs_r0: float | None = None,
    obs_r1: float | None = None,
) -> tuple[OptimizeGraph, list[SphereObs]]:
    """Planar S: +x, +y, -x, +y, +x. Two sphere obstacles sit in the concave bays."""
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    sp, r = float(resolution), float(radius)
    W = (count - 1) * sp
    pts: list[list[float]] = []
    pts += [[i * sp, 0.0, 0.0] for i in range(count)]
    pts += [[W, j * sp, 0.0] for j in range(1, count)]
    pts += [[W - k * sp, W, 0.0] for k in range(1, count)]
    pts += [[0.0, W + m * sp, 0.0] for m in range(1, count)]
    pts += [[n * sp, 2.0 * W, 0.0] for n in range(1, count)]
    c0 = count - 1
    c1 = c0 + (count - 1)
    c2 = c1 + (count - 1)
    c3 = c2 + (count - 1)
    g = _polyline_graph(
        "s_curve",
        np.asarray(pts, float),
        r,
        ("A", "B"),
        [sp, 0.0, 0.0],
        [sp, 0.0, 0.0],
        corner_v={
            c0: np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0) * sp,
            c1: np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0) * sp,
            c2: np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0) * sp,
            c3: np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0) * sp,
        },
    )
    default_obs_r = max(r * 1.2, sp * 0.35)
    r0 = float(default_obs_r if obs_r0 is None else obs_r0)
    r1 = float(default_obs_r if obs_r1 is None else obs_r1)
    obstacles = [
        SphereObs(c=np.array([0.5 * W, 0.5 * W, 0.0]), r=r0, name="bay0"),
        SphereObs(c=np.array([0.5 * W, 1.5 * W, 0.0]), r=r1, name="bay1"),
    ]
    return g, obstacles


def make_spiral_entangle(
    count: int = 24,
    turns: float = 2.0,
    helix_r: float = 2.0,
    pitch: float = 4.0,
    radius: float = 0.6,
) -> list[OptimizeGraph]:
    """Two phase-offset helices (π apart) for group–group clearance."""
    if count < 3:
        raise ValueError(f"count must be >= 3, got {count}")
    n, R, p, r = int(count), float(helix_r), float(pitch), float(radius)
    turns = float(turns)
    graphs: list[OptimizeGraph] = []
    for gid, phase, usage in (
        ("spiral_a", 0.0, ("A0", "A1")),
        ("spiral_b", np.pi, ("B0", "B1")),
    ):
        pts = []
        for i in range(n):
            t = i / (n - 1)
            ang = 2.0 * np.pi * turns * t + phase
            pts.append([R * np.cos(ang), R * np.sin(ang), p * turns * t])
        pos = np.asarray(pts, float)
        d0 = pos[1] - pos[0]
        d1 = pos[-1] - pos[-2]
        v0 = d0 / (np.linalg.norm(d0) + 1e-12) * float(np.linalg.norm(d0))
        v1 = d1 / (np.linalg.norm(d1) + 1e-12) * float(np.linalg.norm(d1))
        graphs.append(_polyline_graph(gid, pos, r, usage, v0, v1))
    return graphs


MAX_BRANCH = 8

M_USAGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("C", "B"),
)


def _polyline(a, b, n_cells: int) -> list[list[float]]:
    """``n_cells`` segments from ``a`` to ``b`` (``n_cells+1`` samples)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return [
        ((1.0 - j / n_cells) * a + (j / n_cells) * b).tolist()
        for j in range(n_cells + 1)
    ]


def _assemble(
    gid: str,
    terminals: list[dict],
    polylines: list[list[list[float]]],
    usages: Sequence[tuple[str, str]],
    domain: dict,
    radius: float,
    *,
    tau0: float = 2.0,
    alpha: float = 2.0,
) -> tuple[OptimizeGraph, GroupTree, PartitionedTree]:
    """tree → junction → Hermite (same path as ``pipeline.build_graphs``)."""
    sp = float(domain["resolution"])
    tree = build_tree(
        {"id": gid, "terminals": terminals},
        {"paths": [{"xyz": xyz, "radius": float(radius)} for xyz in polylines]},
        {"usages": [[a, b] for a, b in usages]},
    )
    part = build_junctions(tree, domain, tau0=tau0, alpha=alpha)
    g = assemble_hermite_graph(part, length_max=sp)
    g.refresh(backend="numpy", store=True)
    return g, tree, part


def make_branch_union(
    count: int = 4,
    n_branch: int = 3,
    resolution: float = 1.0,
    radius: float = 0.5,
    fork_gap: float | None = None,
) -> tuple[OptimizeGraph, GroupTree, PartitionedTree]:
    """3D multi-branch Y via tree→junction→Hermite.

    Port A on y=0; tip ports on y=max; fork interior. Arms fan in xz (not coplanar).
    """
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    if n_branch < 2 or n_branch > MAX_BRANCH:
        raise ValueError(f"n_branch must be in [2, {MAX_BRANCH}], got {n_branch}")
    sp, r, nb = float(resolution), float(radius), int(n_branch)
    n_cells = int(count) - 1
    gap = float(fork_gap) if fork_gap is not None else max(3.0 * r, 1.5 * sp)
    hy = n_cells * sp
    pad = 2.0 * sp
    cx = pad + gap
    cz = pad + gap
    size_x = int(math.ceil((cx + gap + pad) / sp))
    size_y = int(math.ceil(hy / sp))
    size_z = int(math.ceil((cz + gap + pad) / sp))
    domain = {
        "resolution": sp,
        "size_x": max(size_x, 1),
        "size_y": max(size_y, 1),
        "size_z": max(size_z, 1),
    }
    y_top = domain["size_y"] * sp
    A = [cx, 0.0, cz]
    F = [cx, 0.5 * y_top, cz]
    terminals = [{"id": "A", "pos": list(A), "radius": r}]
    tips = []
    for k in range(nb):
        ang = 2.0 * np.pi * k / nb
        tip = [cx + gap * np.cos(ang), y_top, cz + gap * np.sin(ang)]
        tips.append(tip)
        terminals.append({"id": f"B{k}", "pos": tip, "radius": r})

    polylines = [_polyline(A, F, max(n_cells, 1))]
    for tip in tips:
        polylines.append(_polyline(F, tip, max(n_cells, 1)))
    usages = [("A", f"B{k}") for k in range(nb)]
    return _assemble("branch_union", terminals, polylines, usages, domain, r)


def make_m_shape(
    step: int = 2,
    height: int = 2,
    resolution: float = 1.0,
    radius: float = 0.5,
    usages: Sequence[tuple[str, str]] | None = None,
) -> tuple[OptimizeGraph, GroupTree, PartitionedTree]:
    """Goalpost M via tree→junction→Hermite: uprights A/B/C + top At–Bt–Ct."""
    step_n, height_n = int(step), int(height)
    sp, r = float(resolution), float(radius)
    if step_n < 1 or height_n < 1:
        raise ValueError(f"step and height must be >= 1, got {step_n}, {height_n}")
    if usages is None:
        usages = [("A", "B"), ("B", "C"), ("A", "C")]
    usages = [(str(a), str(b)) for a, b in usages]
    if not usages:
        raise ValueError("usages must be non-empty")
    for u in usages:
        if u not in M_USAGE_PAIRS:
            raise ValueError(f"unknown usage {u}; choose from {M_USAGE_PAIRS}")

    sx, hy = step_n * sp, height_n * sp
    pad = 2.0 * sp
    z_mid = 2.0 * sp
    A = [pad, 0.0, z_mid]
    B = [pad + sx, 0.0, z_mid]
    C = [pad + 2.0 * sx, 0.0, z_mid]
    At = [pad, hy, z_mid]
    Bt = [pad + sx, hy, z_mid]
    Ct = [pad + 2.0 * sx, hy, z_mid]
    size_x = int(math.ceil((pad + 2.0 * sx + pad) / sp))
    size_y = int(math.ceil((hy + pad) / sp))
    size_z = int(math.ceil((z_mid + pad) / sp))
    domain = {
        "resolution": sp,
        "size_x": max(size_x, 1),
        "size_y": max(size_y, 1),
        "size_z": max(size_z, 1),
    }
    terminals = [
        {"id": "A", "pos": A, "radius": r},
        {"id": "B", "pos": B, "radius": r},
        {"id": "C", "pos": C, "radius": r},
    ]
    polylines = [
        _polyline(A, At, height_n),
        _polyline(B, Bt, height_n),
        _polyline(C, Ct, height_n),
        _polyline(At, Bt, step_n),
        _polyline(Bt, Ct, step_n),
    ]
    return _assemble("m_shape", terminals, polylines, usages, domain, r)
