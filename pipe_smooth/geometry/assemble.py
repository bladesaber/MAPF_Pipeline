"""Build optimize Hermite graph from junction trunk/junction edges + port tangents.

PartitionedTree already splits junction into distinct labeled edges when usage
runs differ (e.g. junction1_1 vs junction1_2). Assemble densifies each edge once;
ports / interfaces / shared labeled-edge knots stay shared by node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from pipe_smooth.topology.junction import PartitionedTree, LabeledEdge
from pipe_smooth.geometry.hermite import unit, hermite_length, sample_hermite


@dataclass
class HermiteNode:
    pos: np.ndarray
    radius: float = 0.5
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
    ts_count: int = 17  # sample count along [0,1]; set via ts_count_for / sampling
    # temporary sample cache (numpy; refill each opt step for AD-out detection)
    _xyz: np.ndarray | None = field(default=None, repr=False)
    _radius: np.ndarray | None = field(default=None, repr=False)
    _ts: np.ndarray | None = field(default=None, repr=False)

    def radius_at(self, t: float, nodes: dict[str, HermiteNode]) -> float:
        """Linear radius on [0, 1] between endpoint node radii."""
        r0 = float(nodes[self.u].radius)
        r1 = float(nodes[self.v].radius)
        return (1.0 - t) * r0 + t * r1

    def get_xyz(self) -> np.ndarray:
        if self._xyz is None:
            raise RuntimeError(f"segment {self.id}: samples not refreshed")
        return self._xyz

    def get_radius(self) -> np.ndarray:
        if self._radius is None:
            raise RuntimeError(f"segment {self.id}: samples not refreshed")
        return self._radius

    def get_ts(self) -> np.ndarray:
        if self._ts is None:
            raise RuntimeError(f"segment {self.id}: samples not refreshed")
        return self._ts

    def refresh_samples(
        self,
        nodes: dict[str, HermiteNode],
        p0=None,
        v0=None,
        s0=None,
        p1=None,
        v1=None,
        s1=None,
        r0: float | None = None,
        r1: float | None = None,
        resolution: float | None = None,
    ) -> None:
        """Fill ``_xyz`` / ``_radius`` / ``_ts`` (numpy). Ends default to node u→v.

        If ``resolution`` is set, update ``ts_count`` via ``ts_count_for`` first.
        """
        nu, nv = nodes[self.u], nodes[self.v]
        if p0 is None:
            p0, v0, s0 = nu.pos, nu.v, nu.sigma
            p1, v1, s1 = nv.pos, nv.v, nv.sigma
        if r0 is None:
            r0, r1 = float(nu.radius), float(nv.radius)
        if resolution is not None:
            self.ts_count = HermiteSegment.ts_count_for(
                p0, v0, s0, p1, v1, s1, resolution
            )
        n = max(2, int(self.ts_count))
        self._ts = np.linspace(0.0, 1.0, n)
        self._xyz = np.asarray(
            sample_hermite(p0, v0, s0, p1, v1, s1, n=n, backend="numpy"), float
        )
        self._radius = (1.0 - self._ts) * float(r0) + self._ts * float(r1)  # type: ignore[arg-type]

    def clear_samples(self) -> None:
        """Drop temporary sample cache."""
        self._xyz = self._radius = self._ts = None

    @staticmethod
    def ts_count_for(p0, v0, s0, p1, v1, s1, resolution: float) -> int:
        """Number of t-samples for ≈arc-length spacing ``resolution``.

        Same rule as ``sample_usage_path``: n = max(2, ceil(L / resolution)).
        """
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        L = float(hermite_length(p0, v0, s0, p1, v1, s1, backend="numpy"))
        return max(2, int(math.ceil(L / resolution)))


@dataclass
class OptimizeUsagePath:
    """Usage as ordered labeled edges and densified Hermite segments.

    ``edges`` / ``edges_reverse``: PartitionedTree labeled-edge walk (same as topology).
    ``segments`` / ``reverse``: Hermite segment ids along that walk (travel vs stored u→v).
    """

    usage: tuple[str, str]
    edges: list[str] = field(default_factory=list)
    edges_reverse: list[bool] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    reverse: list[bool] = field(default_factory=list)
    v_start: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    sigma_start: float = 1.0
    v_end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    sigma_end: float = 1.0


@dataclass
class OptimizeGraph:
    id: str
    nodes: dict[str, HermiteNode]
    segments: dict[str, HermiteSegment]
    edge_segs: dict[str, list[str]] = field(default_factory=dict)  # labeled edge -> Hermite segs
    junctions: dict[str, list[str]] = field(default_factory=dict)  # complex id -> labeled edge ids
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
            n = HermiteSegment.ts_count_for(p0, v0, s0, p1, v1, s1, resolution)
            seg.ts_count = n
            xyz = sample_hermite(p0, v0, s0, p1, v1, s1, n=n, backend="numpy")
            # radius from travel-direction endpoints; linear in same t as sample_hermite
            r0 = float(nv.radius if rev else nu.radius)
            r1 = float(nu.radius if rev else nv.radius)
            ts = np.linspace(0.0, 1.0, n)
            if pts:
                xyz = xyz[1:]  # drop duplicate joint
                ts = ts[1:]
            pts.append(xyz)
            rads.append((1.0 - ts) * r0 + ts * r1)
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

    def add_node(kid, pos, role, radius):
        if kid not in nodes_out:
            is_port = role == "port"
            nodes_out[kid] = HermiteNode(
                pos=np.asarray(pos, dtype=float),
                radius=float(radius),
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
            r_a = float(jnodes[id_a].radius)
            r_b = float(jnodes[id_b].radius)
            chord = float(np.linalg.norm(p_b - p_a))
            if chord < 1e-12:
                raise ValueError(f"Zero-length waypoint pair on {e.id}: {id_a}->{id_b}")

            n_split = max(1, int(math.ceil(chord / length_max)))
            chain_ids = [local_id(id_a)]
            chain_pos = [p_a]
            chain_r = [r_a]
            for k in range(1, n_split):
                w = k / n_split
                chain_ids.append(f"{e.id}_{i}_{k}")
                chain_pos.append(p_a + w * (p_b - p_a))
                chain_r.append((1.0 - w) * r_a + w * r_b)
            chain_ids.append(local_id(id_b))
            chain_pos.append(p_b)
            chain_r.append(r_b)

            for kid, p, r in zip(chain_ids, chain_pos, chain_r):
                if kid in shared:
                    role = "port" if (kid in jnodes and jnodes[kid].terminal_id) else "interface"
                    r = float(jnodes[kid].radius) if kid in jnodes else r
                else:
                    role = e.kind
                add_node(kid, p, role, r)

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
                    id=sid, edge=e.id, kind=e.kind, u=u, v=v_id
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
                edges=list(up.edges),
                edges_reverse=list(up.reverse),
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
        edge_segs=edge_segs,
        junctions={Z.id: list(Z.edges) for Z in junction.complexes},
        usage_paths=usage_paths,
    )
