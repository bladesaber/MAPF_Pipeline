"""Build optimize Hermite graph from junction trunk/junction edges + port tangents.

PartitionedTree already splits junction into distinct labeled edges when usage
runs differ (e.g. junction1_1 vs junction1_2). Assemble densifies each edge once;
ports / interfaces / shared labeled-edge knots stay shared by node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from topology.junction import PartitionedTree, LabeledEdge
from .hermite import unit, hermite_length, sample_hermite


@dataclass
class HermiteNode:
    pos: np.ndarray
    v: np.ndarray | None = None
    sigma: float | None = None
    fixed_p: bool = False
    fixed_v: bool = False
    role: str = "trunk"  # port | interface | trunk | junction


@dataclass
class HermiteSegment:
    id: str
    edge: str
    kind: str  # trunk | junction
    u: str
    v: str
    radius: float


@dataclass
class OptimizeUsagePath:
    """Usage as ordered Hermite segment ids (travel direction vs stored u→v)."""

    usage: tuple[str, str]
    segments: list[str]
    reverse: list[bool]
    v_start: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    sigma_start: float = 1.0
    v_end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    sigma_end: float = 1.0


@dataclass
class OptimizeGraph:
    id: str
    nodes: dict[str, HermiteNode]
    segments: dict[str, HermiteSegment]
    usage_paths: list[OptimizeUsagePath] = field(default_factory=list)

    def sample_usage_path(self, path: OptimizeUsagePath, resolution: float = 0.5):
        """Sample γ along one usage. `resolution` ≈ arc-length spacing.

        If reverse[i]: swap endpoints and flip both tangents (same trace, opposite travel).
        First/last segment ends use path.v_start/σ_start and v_end/σ_end (domain port clamps).
        Consecutive segments drop the duplicate joint point.
        Returns dict with xyz (N,3), radius (N,), usage.
        """
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        pts, rads = [], []
        nseg = len(path.segments)
        for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
            seg = self.segments[sid]
            nu, nv = self.nodes[seg.u], self.nodes[seg.v]
            if rev:
                p0, v0, s0 = nv.pos, -nv.v, nv.sigma  # type: ignore
                p1, v1, s1 = nu.pos, -nu.v, nu.sigma  # type: ignore
            else:
                p0, v0, s0 = nu.pos, nu.v, nu.sigma
                p1, v1, s1 = nv.pos, nv.v, nv.sigma
            if i == 0:
                v0, s0 = path.v_start, path.sigma_start
            if i == nseg - 1:
                v1, s1 = path.v_end, path.sigma_end
            L = float(hermite_length(p0, v0, s0, p1, v1, s1, backend="numpy"))
            n = max(2, int(math.ceil(L / resolution)))
            xyz = sample_hermite(p0, v0, s0, p1, v1, s1, n=n, backend="numpy")
            if pts:
                xyz = xyz[1:]  # drop duplicate joint
            pts.append(xyz)
            rads.append(np.full(xyz.shape[0], float(seg.radius)))
        if not pts:
            return {
                "usage": path.usage,
                "xyz": np.zeros((0, 3)),
                "radius": np.zeros((0,)),
            }
        return {
            "usage": path.usage,
            "xyz": np.concatenate(pts, axis=0),
            "radius": np.concatenate(rads, axis=0),
        }


def assemble_hermite_graph(junction: PartitionedTree, length_max: float = 8.0) -> OptimizeGraph:
    """One Hermite per consecutive waypoint pair; split a pair only if chord > length_max."""
    jnodes = junction.nodes
    shared = {nid for nid, n in jnodes.items() if n.terminal_id}
    for Z in junction.complexes:
        shared.update(Z.interfaces)

    by_eid = {e.id: e for e in list(junction.trunk_edges) + list(junction.junction_edges)}
    nodes_out: dict[str, HermiteNode] = {}
    segments: dict[str, HermiteSegment] = {}
    edge_segs: dict[str, list[str]] = {}  # eid -> hermite segment ids along e.nodes

    def add_node(kid, pos, role):
        if kid not in nodes_out:
            is_port = role == "port"
            nodes_out[kid] = HermiteNode(
                pos=np.asarray(pos, dtype=float),
                v=None if is_port else np.array([1.0, 0.0, 0.0]),
                sigma=None if is_port else 1.0,
                fixed_p=is_port,
                fixed_v=is_port,
                role=role,
            )

    def densify_edge(e: LabeledEdge):
        nids = e.nodes
        if len(nids) < 2:
            edge_segs[e.id] = []
            return
        segs_here = []

        def local_id(nid: str) -> str:
            # ports / interfaces stay global; all other knots are private to this labeled edge
            return nid if nid in shared else f"{e.id}:{nid}"

        for i in range(len(nids) - 1):
            id_a, id_b = nids[i], nids[i + 1]
            p_a = np.asarray(jnodes[id_a].pos, dtype=float)
            p_b = np.asarray(jnodes[id_b].pos, dtype=float)
            chord = float(np.linalg.norm(p_b - p_a))
            if chord < 1e-12:
                raise ValueError(f"Zero-length waypoint pair on {e.id}: {id_a}->{id_b}")

            n_split = max(1, int(math.ceil(chord / length_max)))
            chain_ids = [local_id(id_a)]
            chain_pos = [p_a]
            for k in range(1, n_split):
                w = k / n_split
                chain_ids.append(f"{e.id}_{i}_{k}")
                chain_pos.append(p_a + w * (p_b - p_a))
            chain_ids.append(local_id(id_b))
            chain_pos.append(p_b)

            for kid, p in zip(chain_ids, chain_pos):
                if kid in shared:
                    role = "port" if (kid in jnodes and jnodes[kid].terminal_id) else "interface"
                else:
                    role = e.kind
                add_node(kid, p, role)

            for j in range(len(chain_ids) - 1):
                p0, p1 = chain_pos[j], chain_pos[j + 1]
                seg_L = float(np.linalg.norm(p1 - p0))
                tang = unit(p1 - p0, backend="numpy")
                u, v_id = chain_ids[j], chain_ids[j + 1]
                if not nodes_out[u].fixed_v:
                    nodes_out[u].v = tang
                    nodes_out[u].sigma = seg_L
                if not nodes_out[v_id].fixed_v:
                    nodes_out[v_id].v = tang
                    nodes_out[v_id].sigma = seg_L
                sid = f"{e.id}_{i}_{j}"
                segments[sid] = HermiteSegment(
                    id=sid, edge=e.id, kind=e.kind, u=u, v=v_id, radius=e.radius
                )
                segs_here.append(sid)
        edge_segs[e.id] = segs_here

    # walk usages; densify each labeled edge at most once
    usage_paths = []
    for up in junction.usage_paths:
        seg_ids, rev_flags = [], []
        for eid, rev in zip(up.edges, up.reverse):
            if eid not in edge_segs:
                densify_edge(by_eid[eid])
            segs = list(edge_segs[eid])
            if rev:
                segs.reverse()
            for sid in segs:
                seg_ids.append(sid)
                rev_flags.append(rev)
        usage_paths.append(
            OptimizeUsagePath(
                usage=(up.usage[0], up.usage[1]),
                segments=seg_ids,
                reverse=rev_flags,
                v_start=np.asarray(up.v_start, dtype=float),
                sigma_start=float(up.sigma_start),
                v_end=np.asarray(up.v_end, dtype=float),
                sigma_end=float(up.sigma_end),
            )
        )

    return OptimizeGraph(
        id=junction.id,
        nodes=nodes_out,
        segments=segments,
        usage_paths=usage_paths,
    )
