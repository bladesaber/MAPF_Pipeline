"""Build optimize Hermite graph from junction trunk/junction edges + port tangents.

PartitionedTree already splits junction into distinct labeled edges when usage
runs differ (e.g. junction1_1 vs junction1_2). Assemble densifies each edge once;
ports / interfaces / shared labeled-edge knots stay shared by node id.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipe_smooth.topology.junction import PartitionedTree, LabeledEdge
from pipe_smooth.geometry.hermite import unit, hermite_length, sample_hermite
from pipe_smooth.utils import backend as _backend, to_float_list


@dataclass
class HermiteNode:
    pos: np.ndarray
    radius: float = 0.5
    v: np.ndarray | None = None
    fixed_p: bool = False
    fixed_v: bool = False
    role: str = "trunk"  # port | interface | trunk | junction

    def export(self) -> dict[str, Any]:
        return {
            "pos": to_float_list(self.pos),
            "radius": float(self.radius),
            "v": None if self.v is None else to_float_list(self.v),
            "fixed_p": bool(self.fixed_p),
            "fixed_v": bool(self.fixed_v),
            "role": self.role,
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "HermiteNode":
        v = data.get("v")
        return cls(
            pos=np.asarray(data["pos"], dtype=float),
            radius=float(data.get("radius", 0.5)),
            v=None if v is None else np.asarray(v, dtype=float),
            fixed_p=bool(data.get("fixed_p", False)),
            fixed_v=bool(data.get("fixed_v", False)),
            role=str(data.get("role", "trunk")),
        )


@dataclass
class HermiteSegment:
    id: str
    edge: str
    kind: str  # trunk | junction
    u: str
    v: str
    ts_count: int = 10  # sample count along [0,1]; set via ts_count_for / sampling
    
    ts: object | None = None  # (n,) cached parameters
    xyz: object | None = None  # (n, 3) cached samples along stored u→v
    radius: object | None = None  # (n,) linear radius along stored u→v

    def export(self) -> dict[str, Any]:
        """Topology only; caches (ts/xyz/radius) rebuilt by ``refresh``."""
        return {
            "id": self.id,
            "edge": self.edge,
            "kind": self.kind,
            "u": self.u,
            "v": self.v,
            "ts_count": int(self.ts_count),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "HermiteSegment":
        return cls(
            id=str(data["id"]),
            edge=str(data["edge"]),
            kind=str(data["kind"]),
            u=str(data["u"]),
            v=str(data["v"]),
            ts_count=int(data.get("ts_count", 17)),
        )

    def radius_at(self, t: float, nodes: dict[str, HermiteNode]) -> float:
        """Linear radius on [0, 1] between endpoint node radii."""
        r0 = float(nodes[self.u].radius)
        r1 = float(nodes[self.v].radius)
        return (1.0 - t) * r0 + t * r1

    @staticmethod
    def ts_count_for(p0, v0, p1, v1, resolution: float) -> int:
        """Number of t-samples for ≈arc-length spacing ``resolution``.

        Same rule as ``sample_usage_path``: n = max(2, ceil(L / resolution)).
        """
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        L = float(hermite_length(np.asarray(p0), np.asarray(v0), np.asarray(p1), np.asarray(v1), backend="numpy"))
        return max(2, int(math.ceil(L / resolution)))

    def refresh(
        self,
        p0,
        p1,
        v0,
        v1,
        r0: float,
        r1: float,
        resolution: float | None = None,
        backend: str = "numpy",
    ):
        if resolution is not None:
            self.ts_count = self.ts_count_for(p0, v0, p1, v1, resolution)
        xp = _backend(backend)
        n = max(2, int(self.ts_count))
        self.ts = xp.linspace(0.0, 1.0, n)
        self.xyz = sample_hermite(p0, v0, p1, v1, n=n, backend=backend)
        self.radius = (1.0 - self.ts) * float(r0) + self.ts * float(r1)
        return self

    def samples(self, backend: str = "numpy"):
        """Cached (ts, xyz, radius) along u→v, converted for callers outside AD."""
        if self.xyz is None or self.ts is None or self.radius is None:
            raise RuntimeError(f"segment {self.id}: call refresh before samples")
        xp = _backend(backend)
        return xp.asarray(self.ts), xp.asarray(self.xyz), xp.asarray(self.radius)


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
    v_end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def export(self) -> dict[str, Any]:
        return {
            "usage": [self.usage[0], self.usage[1]],
            "edges": list(self.edges),
            "edges_reverse": [bool(x) for x in self.edges_reverse],
            "segments": list(self.segments),
            "reverse": [bool(x) for x in self.reverse],
            "v_start": to_float_list(self.v_start),
            "v_end": to_float_list(self.v_end),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "OptimizeUsagePath":
        u = data["usage"]
        return cls(
            usage=(str(u[0]), str(u[1])),
            edges=[str(e) for e in data.get("edges") or []],
            edges_reverse=[bool(x) for x in data.get("edges_reverse") or []],
            segments=[str(s) for s in data.get("segments") or []],
            reverse=[bool(x) for x in data.get("reverse") or []],
            v_start=np.asarray(data.get("v_start", [1.0, 0.0, 0.0]), dtype=float),
            v_end=np.asarray(data.get("v_end", [1.0, 0.0, 0.0]), dtype=float),
        )

    def port_derivatives(self, backend: str = "numpy"):
        """Port end derivatives (v_start, v_end) for the requested backend."""
        xp = _backend(backend)
        return xp.asarray(self.v_start), xp.asarray(self.v_end)


@dataclass
class OptimizeGraph:
    """Hermite graph for one pipe group.

    **Tangent assumption:** each Hermite end has one geometric derivative direction.
    Usages that traverse a segment forward/backward only **flip the sign** (±);
    they never assign non-colinear port tangents to the same stored end. Therefore
    ``hermite_ends`` may keep the first usage's clamp per ``seg_id`` safely.
    Call ``check_usage_tangents_colinear`` after assemble to verify.
    """

    id: str
    nodes: dict[str, HermiteNode]
    segments: dict[str, HermiteSegment]
    edge_segs: dict[str, list[str]] = field(default_factory=dict)  # labeled edge -> Hermite segs
    junctions: dict[str, list[str]] = field(default_factory=dict)  # complex id -> labeled edge ids
    usage_paths: list[OptimizeUsagePath] = field(default_factory=list)

    def export(self, path: str | Path) -> None:
        """Write graph JSON (nested types via their ``export``; no sample caches)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "nodes": {nid: n.export() for nid, n in self.nodes.items()},
            "segments": {sid: s.export() for sid, s in self.segments.items()},
            "edge_segs": {k: list(v) for k, v in self.edge_segs.items()},
            "junctions": {k: list(v) for k, v in self.junctions.items()},
            "usage_paths": [p.export() for p in self.usage_paths],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OptimizeGraph":
        data = json.loads(Path(path).read_text())
        g = cls(
            id=str(data["id"]),
            nodes={nid: HermiteNode.load(n) for nid, n in (data.get("nodes") or {}).items()},
            segments={
                sid: HermiteSegment.load(s) for sid, s in (data.get("segments") or {}).items()
            },
            edge_segs={k: list(v) for k, v in (data.get("edge_segs") or {}).items()},
            junctions={k: list(v) for k, v in (data.get("junctions") or {}).items()},
            usage_paths=[OptimizeUsagePath.load(p) for p in data.get("usage_paths") or []],
        )
        return g

    def refresh(self, resolution: float | None = None, backend: str = "numpy"):
        """Refresh every segment once. Call after unpack. Port ends override v0/v1."""
        ends = self.hermite_ends(backend=backend)
        for sid, (p0, v0, p1, v1) in ends.items():
            seg = self.segments[sid]
            nu, nv = self.nodes[seg.u], self.nodes[seg.v]
            seg.refresh(
                p0,
                p1,
                v0,
                v1,
                float(nu.radius),
                float(nv.radius),
                resolution=resolution,
                backend=backend,
            )
        return self

    def hermite_ends(self, backend: str = "numpy") -> dict[str, tuple]:
        """seg_id → (p0, v0, p1, v1) along stored u→v (same port clamps as refresh).

        First usage that touches a seg wins the port clamp write. Safe under the
        colinear ± assumption (see class docstring / ``check_usage_tangents_colinear``).
        """
        xp = _backend(backend)
        out: dict[str, tuple] = {}
        seen: set[str] = set()
        for path in self.usage_paths:
            nseg = len(path.segments)
            v_start, v_end = path.port_derivatives(backend=backend)
            for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
                if sid in seen:
                    continue
                seen.add(sid)
                seg = self.segments[sid]
                nu, nv = self.nodes[seg.u], self.nodes[seg.v]
                p0, p1 = nu.pos, nv.pos
                v0, v1 = nu.v, nv.v
                if rev:
                    if i == 0:
                        v1 = -v_start
                    if i == nseg - 1:
                        v0 = -v_end
                else:
                    if i == 0:
                        v0 = v_start
                    if i == nseg - 1:
                        v1 = v_end
                out[sid] = (
                    xp.asarray(p0),
                    xp.asarray(v0),
                    xp.asarray(p1),
                    xp.asarray(v1),
                )
        for sid in self.segments:
            if sid not in seen:
                raise RuntimeError(f"segment {sid}: not found in usage paths")
        return out

    def check_usage_tangents_colinear(self, atol: float = 1e-6) -> None:
        """Raise if two usages disagree on the geometric derivative at one stored end.

        For each path start/end on a segment, map the travel port derivative into the
        stored u→v frame (reverse ⇒ sign flip on the opposite end). All claims at
        ``(seg_id, 'u'|'v')`` must match up to sign: same ``||v||`` and parallel
        direction (hermite_ends / ± assumption).
        """

        def _same_end_derivative(a, b) -> bool:
            a = np.asarray(a, dtype=float).reshape(3)
            b = np.asarray(b, dtype=float).reshape(3)
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if not np.isclose(na, nb, rtol=atol, atol=atol):
                return False
            if na <= atol or nb <= atol:
                return True
            return abs(abs(float(np.dot(a / na, b / nb))) - 1.0) <= atol

        # sid -> end -> list of claimed stored-frame derivatives
        claims: dict[str, dict[str, list]] = {
            sid: {"u": [], "v": []} for sid in self.segments
        }
        for path in self.usage_paths:
            nseg = len(path.segments)
            for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
                if i == 0:
                    # travel start → stored end
                    if rev:
                        claims[sid]["v"].append(-np.asarray(path.v_start, dtype=float))
                    else:
                        claims[sid]["u"].append(np.asarray(path.v_start, dtype=float))
                if i == nseg - 1:
                    if rev:
                        claims[sid]["u"].append(-np.asarray(path.v_end, dtype=float))
                    else:
                        claims[sid]["v"].append(np.asarray(path.v_end, dtype=float))

        for sid, ends in claims.items():
            for which, vecs in ends.items():
                if len(vecs) < 2:
                    continue
                ref = vecs[0]
                for k, v in enumerate(vecs[1:], start=1):
                    if not _same_end_derivative(ref, v):
                        na = float(np.linalg.norm(ref))
                        nb = float(np.linalg.norm(v))
                        raise ValueError(
                            f"OptimizeGraph {self.id!r}: segment {sid!r} stored-{which} "
                            f"has inconsistent usage derivatives (claim 0 vs {k}: "
                            f"||v||={na:.6g} vs {nb:.6g}); "
                            f"hermite_ends assumes usages only flip ± with equal magnitude"
                        )

    def sample_usage_path(self, path: OptimizeUsagePath, resolution: float = 0.5):
        """Sample γ along one usage. `resolution` ≈ arc-length spacing.

        If reverse[i]: swap endpoints and flip both tangents (same trace, opposite travel).
        First/last segment ends use path.v_start and v_end (domain port derivatives).
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
                p0, v0 = nv.pos, -nv.v  # type: ignore
                p1, v1 = nu.pos, -nu.v  # type: ignore
            else:
                p0, v0 = nu.pos, nu.v
                p1, v1 = nv.pos, nv.v
            if i == 0:
                v0 = path.v_start
            if i == nseg - 1:
                v1 = path.v_end
            n = HermiteSegment.ts_count_for(p0, v0, p1, v1, resolution)
            seg.ts_count = n
            xyz = sample_hermite(p0, v0, p1, v1, n=n, backend="numpy")
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
                    nodes_out[u].v = tang * seg_L
                if not nodes_out[v_id].fixed_v:
                    nodes_out[v_id].v = tang * seg_L
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
                v_start=np.asarray(up.v_start, dtype=float) * float(up.sigma_start),
                v_end=np.asarray(up.v_end, dtype=float) * float(up.sigma_end),
            )
        )

    g = OptimizeGraph(
        id=junction.id,
        nodes=nodes_out,
        segments=segments,
        edge_segs=edge_segs,
        junctions={Z.id: list(Z.edges) for Z in junction.complexes},
        usage_paths=usage_paths,
    )
    g.check_usage_tangents_colinear()
    return g
