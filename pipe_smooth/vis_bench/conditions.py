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


def make_elbow90(count: int = 5, resolution: float = 1.0, radius: float = 0.5) -> OptimizeGraph:
    """L-path: ``count`` waypoints along +x, then ``count`` along +y (shared corner)."""
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    sp, r = float(resolution), float(radius)
    nodes: dict[str, HermiteNode] = {}
    chain: list[str] = []
    for i in range(count):
        nid = f"a{i}"
        chain.append(nid)
        nodes[nid] = HermiteNode(
            pos=np.array([i * sp, 0.0, 0.0]),
            radius=r,
            v=None,
            fixed_p=(i == 0),
            fixed_v=(i == 0),
            role="port" if i == 0 else "trunk",
        )
    corner = chain[-1]
    for j in range(1, count):
        nid = f"b{j}"
        chain.append(nid)
        is_end = j == count - 1
        nodes[nid] = HermiteNode(
            pos=np.array([(count - 1) * sp, j * sp, 0.0]),
            radius=r,
            v=None,
            fixed_p=is_end,
            fixed_v=is_end,
            role="port" if is_end else "trunk",
        )

    segments: dict[str, HermiteSegment] = {}
    seg_ids: list[str] = []
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        p0, p1 = nodes[u].pos, nodes[v].pos
        tang = p1 - p0
        L = float(np.linalg.norm(tang))
        tang = tang / (L + 1e-12)
        if not nodes[u].fixed_v:
            nodes[u].v = tang * L
        if not nodes[v].fixed_v:
            nodes[v].v = tang * L
        sid = f"s{i}"
        segments[sid] = HermiteSegment(id=sid, edge="e0", kind="trunk", u=u, v=v)
        seg_ids.append(sid)

    c = nodes[corner]
    if not c.fixed_v:
        c.v = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)) * sp

    g = OptimizeGraph(
        id="elbow_90",
        nodes=nodes,
        segments=segments,
        edge_segs={"e0": list(seg_ids)},
        junctions={},
        usage_paths=[
            OptimizeUsagePath(
                usage=("A", "B"),
                edges=["e0"],
                edges_reverse=[False],
                segments=seg_ids,
                reverse=[False] * len(seg_ids),
                v_start=np.array([sp, 0.0, 0.0]),
                v_end=np.array([0.0, sp, 0.0]),
            )
        ],
    )
    g.check_usage_tangents_colinear()
    return g


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
    nodes: dict[str, HermiteNode] = {}
    chain: list[str] = []

    def _add(nid: str, pos, *, fixed: bool = False, role: str = "trunk"):
        nodes[nid] = HermiteNode(
            pos=np.asarray(pos, float),
            radius=r,
            v=None,
            fixed_p=fixed,
            fixed_v=fixed,
            role=role,
        )
        chain.append(nid)

    for i in range(count):
        _add(f"a{i}", [i * sp, 0.0, 0.0], fixed=(i == 0), role="port" if i == 0 else "trunk")
    for j in range(1, count):
        _add(f"b{j}", [W, j * sp, 0.0])
    for k in range(1, count):
        _add(f"c{k}", [W - k * sp, W, 0.0])
    for m in range(1, count):
        _add(f"d{m}", [0.0, W + m * sp, 0.0])
    for n in range(1, count):
        end = n == count - 1
        _add(f"e{n}", [n * sp, 2.0 * W, 0.0], fixed=end, role="port" if end else "trunk")

    segments: dict[str, HermiteSegment] = {}
    seg_ids: list[str] = []
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        p0, p1 = nodes[u].pos, nodes[v].pos
        tang = p1 - p0
        L = float(np.linalg.norm(tang))
        tang = tang / (L + 1e-12)
        if not nodes[u].fixed_v:
            nodes[u].v = tang * L
        if not nodes[v].fixed_v:
            nodes[v].v = tang * L
        sid = f"s{i}"
        segments[sid] = HermiteSegment(id=sid, edge="e0", kind="trunk", u=u, v=v)
        seg_ids.append(sid)

    # soften four corners (±x ↔ ±y)
    for nid, d in (
        (f"a{count - 1}", np.array([1.0, 1.0, 0.0])),
        (f"b{count - 1}", np.array([-1.0, 1.0, 0.0])),
        (f"c{count - 1}", np.array([-1.0, 1.0, 0.0])),
        (f"d{count - 1}", np.array([1.0, 1.0, 0.0])),
    ):
        node = nodes[nid]
        if not node.fixed_v:
            node.v = d / (np.linalg.norm(d) + 1e-12) * sp

    g = OptimizeGraph(
        id="s_curve",
        nodes=nodes,
        segments=segments,
        edge_segs={"e0": list(seg_ids)},
        junctions={},
        usage_paths=[
            OptimizeUsagePath(
                usage=("A", "B"),
                edges=["e0"],
                edges_reverse=[False],
                segments=seg_ids,
                reverse=[False] * len(seg_ids),
                v_start=np.array([sp, 0.0, 0.0]),
                v_end=np.array([sp, 0.0, 0.0]),
            )
        ],
    )
    g.check_usage_tangents_colinear()
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
    turns: float = 2.0,  # number of full helix revolutions
    helix_r: float = 2.0,  # cylinder radius of the helix centerline
    pitch: float = 4.0,  # axial advance (+z) per turn
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
        nodes: dict[str, HermiteNode] = {}
        chain: list[str] = []
        for i in range(n):
            t = i / (n - 1)
            ang = 2.0 * np.pi * turns * t + phase
            pos = np.array([R * np.cos(ang), R * np.sin(ang), p * turns * t], float)
            nid = f"n{i}"
            chain.append(nid)
            end = i == 0 or i == n - 1
            nodes[nid] = HermiteNode(
                pos=pos,
                radius=r,
                v=None,
                fixed_p=end,
                fixed_v=end,
                role="port" if end else "trunk",
            )
        segments: dict[str, HermiteSegment] = {}
        seg_ids: list[str] = []
        for i in range(n - 1):
            u, v = chain[i], chain[i + 1]
            d = nodes[v].pos - nodes[u].pos
            L = float(np.linalg.norm(d))
            tang = d / (L + 1e-12)
            if not nodes[u].fixed_v:
                nodes[u].v = tang * L
            if not nodes[v].fixed_v:
                nodes[v].v = tang * L
            sid = f"s{i}"
            segments[sid] = HermiteSegment(id=sid, edge="e0", kind="trunk", u=u, v=v)
            seg_ids.append(sid)
        # port tangents along first/last chords
        d0 = nodes[chain[1]].pos - nodes[chain[0]].pos
        d1 = nodes[chain[-1]].pos - nodes[chain[-2]].pos
        v0 = d0 / (np.linalg.norm(d0) + 1e-12) * float(np.linalg.norm(d0))
        v1 = d1 / (np.linalg.norm(d1) + 1e-12) * float(np.linalg.norm(d1))
        nodes[chain[0]].v = v0
        nodes[chain[-1]].v = v1
        g = OptimizeGraph(
            id=gid,
            nodes=nodes,
            segments=segments,
            edge_segs={"e0": list(seg_ids)},
            junctions={},
            usage_paths=[
                OptimizeUsagePath(
                    usage=usage,
                    edges=["e0"],
                    edges_reverse=[False],
                    segments=seg_ids,
                    reverse=[False] * len(seg_ids),
                    v_start=v0.copy(),
                    v_end=v1.copy(),
                )
            ],
        )
        g.check_usage_tangents_colinear()
        graphs.append(g)
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
    g.refresh(backend="numpy")  # default ts_count (≥3 interiors for union)
    return g, tree, part


def make_branch_union(
    count: int = 4,
    n_branch: int = 3,
    resolution: float = 1.0,
    radius: float = 0.5,
    fork_gap: float | None = None,  # tip circle radius on top face (default max(3r,1.5·res))
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
    # tips on y=max face; inset from x/z walls
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
    step: int = 2,  # grid cells A–B / B–C (length = step·resolution)
    height: int = 2,  # grid cells upright (length = height·resolution)
    resolution: float = 1.0,  # cell size
    radius: float = 0.5,
    usages: Sequence[tuple[str, str]] | None = None,
) -> tuple[OptimizeGraph, GroupTree, PartitionedTree]:
    """Goalpost M via tree→junction→Hermite: uprights A/B/C + top At–Bt–Ct.

    Ports A,B,C on y=0 (degree 1); Bt is the degree-3 branch.
    """
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
    #     At ---- Bt ---- Ct
    #     |       |       |
    #     A       B       C   (on y=0 face)
    A = [pad, 0.0, z_mid]
    B = [pad + sx, 0.0, z_mid]
    C = [pad + 2.0 * sx, 0.0, z_mid]
    At = [pad, hy, z_mid]
    Bt = [pad + sx, hy, z_mid]
    Ct = [pad + 2.0 * sx, hy, z_mid]
    size_x = int(math.ceil((pad + 2.0 * sx + pad) / sp))
    size_y = int(math.ceil((hy + pad) / sp))  # tops interior (not on y=max)
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
