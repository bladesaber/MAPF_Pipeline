"""Build optimize Hermite graph from junction trunk/junction edges + port tangents.

Geometry true source is SoA ``pos``/``v``/``radius``. ``refresh`` returns
``(ends, xyz, radius)``; pass ``store=True`` to cache on ``seg_*`` for detection.
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
from pipe_smooth.utils import backend as _backend


@dataclass
class HermiteNode:
    """Node metadata; geometry lives on ``OptimizeGraph.pos/v/radius``."""

    fixed_p: bool = False
    fixed_v: bool = False
    role: str = "trunk"  # port | interface | trunk | junction

    def export(self) -> dict[str, Any]:
        return {
            "fixed_p": bool(self.fixed_p),
            "fixed_v": bool(self.fixed_v),
            "role": self.role,
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "HermiteNode":
        return cls(
            fixed_p=bool(data.get("fixed_p", False)),
            fixed_v=bool(data.get("fixed_v", False)),
            role=str(data.get("role", "trunk")),
        )


@dataclass
class HermiteSegment:
    edge: str
    kind: str  # trunk | junction
    u: int  # node index
    v: int  # node index

    def export(self) -> dict[str, Any]:
        return {
            "edge": self.edge,
            "kind": self.kind,
            "u": int(self.u),
            "v": int(self.v),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "HermiteSegment":
        return cls(
            edge=str(data["edge"]),
            kind=str(data["kind"]),
            u=int(data["u"]),
            v=int(data["v"]),
        )

    @staticmethod
    def ts_count_for(p0, v0, p1, v1, resolution: float) -> int:
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        L = float(
            hermite_length(
                np.asarray(p0), np.asarray(v0), np.asarray(p1), np.asarray(v1),
                backend="numpy",
            )
        )
        return max(5, int(math.ceil(L / resolution)))


@dataclass
class OptimizeUsagePath:
    """Usage walk; ``segments`` are int indices into ``OptimizeGraph.segments``."""

    usage: tuple[str, str]
    edges: list[str] = field(default_factory=list)
    edges_reverse: list[bool] = field(default_factory=list)
    segments: list[int] = field(default_factory=list)
    reverse: list[bool] = field(default_factory=list)
    v_start: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    v_end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def export(self) -> dict[str, Any]:
        return {
            "usage": [self.usage[0], self.usage[1]],
            "edges": list(self.edges),
            "edges_reverse": [bool(x) for x in self.edges_reverse],
            "segments": [int(s) for s in self.segments],
            "reverse": [bool(x) for x in self.reverse],
            "v_start": np.asarray(self.v_start).tolist(),
            "v_end": np.asarray(self.v_end).tolist(),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "OptimizeUsagePath":
        u = data["usage"]
        return cls(
            usage=(str(u[0]), str(u[1])),
            edges=[str(e) for e in data.get("edges") or []],
            edges_reverse=[bool(x) for x in data.get("edges_reverse") or []],
            segments=[int(s) for s in data.get("segments") or []],
            reverse=[bool(x) for x in data.get("reverse") or []],
            v_start=np.asarray(data.get("v_start", [1.0, 0.0, 0.0]), dtype=float),
            v_end=np.asarray(data.get("v_end", [1.0, 0.0, 0.0]), dtype=float),
        )


@dataclass
class OptimizeGraph:
    """Hermite graph: ``nodes[i]`` / ``segments[s]``; SoA rows share the same indices."""

    id: str
    nodes: list[HermiteNode]
    segments: list[HermiteSegment]

    # SoA node geometry
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    # static ends gather (filled by ``bind_ends_tables``; not recomputed in hot paths)
    # segment indices for u and v ends
    u_idx: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    v_idx: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    u_sign: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    v_sign: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    # segment indices for port v0 and v1 ends
    port_v0_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    port_v1_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    port_v0: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    port_v1: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # unique 3-point joints for curvature: travel-order prev→curr (first sighting)
    joint_s0: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    joint_s1: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    joint_flip0: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    joint_flip1: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))

    # SoA segment samples / ends (derived from pos/v; refresh(..., store=True))
    seg_ends: object | None = None  # (S, 4, 3) last hermite_ends
    seg_xyz: object | None = None  # (S, T, 3)
    seg_radius: object | None = None  # (S, T)

    edge_segs: dict[str, list[int]] = field(default_factory=dict)
    junctions: dict[str, list[str]] = field(default_factory=dict)
    usage_paths: list[OptimizeUsagePath] = field(default_factory=list)

    def bind_ends_tables(self) -> "OptimizeGraph":
        """Build static seg→node gather + port clamps. Requires ``u_sign``/``v_sign`` already set."""
        S = len(self.segments)
        self.u_idx = np.array([seg.u for seg in self.segments], dtype=np.int32)
        self.v_idx = np.array([seg.v for seg in self.segments], dtype=np.int32)
        if len(self.u_sign) != S or len(self.v_sign) != S:
            raise RuntimeError("u_sign/v_sign must be set (length S) before bind_ends_tables")
        port_v0_mask = np.zeros(S, dtype=bool)
        port_v1_mask = np.zeros(S, dtype=bool)
        port_v0 = np.zeros((S, 3), dtype=float)
        port_v1 = np.zeros((S, 3), dtype=float)
        seen = np.zeros(S, dtype=bool)
        for path in self.usage_paths:
            nseg = len(path.segments)
            for i, (s, rev) in enumerate(zip(path.segments, path.reverse)):
                if seen[s]:
                    continue
                seen[s] = True
                if rev:
                    if i == 0:
                        port_v1_mask[s] = True
                        port_v1[s] = -np.asarray(path.v_start, dtype=float)
                    if i == nseg - 1:
                        port_v0_mask[s] = True
                        port_v0[s] = -np.asarray(path.v_end, dtype=float)
                else:
                    if i == 0:
                        port_v0_mask[s] = True
                        port_v0[s] = np.asarray(path.v_start, dtype=float)
                    if i == nseg - 1:
                        port_v1_mask[s] = True
                        port_v1[s] = np.asarray(path.v_end, dtype=float)
        if S and not bool(seen.all()):
            raise RuntimeError(f"segments missing from usage paths: {np.where(~seen)[0]}")
        self.port_v0_mask, self.port_v1_mask = port_v0_mask, port_v1_mask
        self.port_v0, self.port_v1 = port_v0, port_v1
        return self.bind_joint_tables()

    def bind_joint_tables(self) -> "OptimizeGraph":
        """Unique unordered joints; orientation from first travel sighting."""
        s0s: list[int] = []
        s1s: list[int] = []
        f0s: list[bool] = []
        f1s: list[bool] = []
        seen: set[tuple[int, int]] = set()
        for path in self.usage_paths:
            prev_s: int | None = None
            prev_rev = False
            for s, rev in zip(path.segments, path.reverse):
                if prev_s is not None:
                    key = (min(prev_s, s), max(prev_s, s))
                    if key not in seen:
                        seen.add(key)
                        s0s.append(prev_s)
                        s1s.append(s)
                        f0s.append(prev_rev)
                        f1s.append(rev)
                prev_s, prev_rev = s, rev
        self.joint_s0 = np.asarray(s0s, dtype=np.int32)
        self.joint_s1 = np.asarray(s1s, dtype=np.int32)
        self.joint_flip0 = np.asarray(f0s, dtype=bool)
        self.joint_flip1 = np.asarray(f1s, dtype=bool)
        return self

    def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "OptimizeGraph",
            "id": self.id,
            "nodes": [n.export() for n in self.nodes],
            "pos": np.asarray(self.pos).tolist(),
            "v": np.asarray(self.v).tolist(),
            "radius": np.asarray(self.radius).tolist(),
            "segments": [s.export() for s in self.segments],
            "u_sign": np.asarray(self.u_sign).tolist(),
            "v_sign": np.asarray(self.v_sign).tolist(),
            "edge_segs": {k: list(v) for k, v in self.edge_segs.items()},
            "junctions": {k: list(v) for k, v in self.junctions.items()},
            "usage_paths": [p.export() for p in self.usage_paths],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OptimizeGraph":
        data = json.loads(Path(path).read_text())
        nodes = [HermiteNode.load(nd) for nd in (data.get("nodes") or [])]
        segments = [HermiteSegment.load(s) for s in (data.get("segments") or [])]
        S = len(segments)
        g = cls(
            id=str(data["id"]),
            nodes=nodes,
            segments=segments,
            pos=np.asarray(data["pos"], dtype=float).reshape(-1, 3),
            v=np.asarray(data["v"], dtype=float).reshape(-1, 3),
            radius=np.asarray(data["radius"], dtype=float).reshape(-1),
            u_sign=np.asarray(data["u_sign"], dtype=float).reshape(S),
            v_sign=np.asarray(data["v_sign"], dtype=float).reshape(S),
            edge_segs={k: [int(x) for x in vv] for k, vv in (data.get("edge_segs") or {}).items()},
            junctions={k: list(vv) for k, vv in (data.get("junctions") or {}).items()},
            usage_paths=[OptimizeUsagePath.load(p) for p in data.get("usage_paths") or []],
        )
        return g.bind_ends_tables()

    def hermite_ends(self, pos=None, v=None, backend: str = "numpy"):
        """Gather Hermite end states ``(S, 4, 3)`` = p0, v0, p1, v1 (+ port clamps).

        Args:
            pos: node positions ``(N, 3)``; default ``self.pos``
            v: node tangents ``(N, 3)``; default ``self.v``
            backend: ``'numpy'`` | ``'jax'``

        Returns:
            ends: ``(S, 4, 3)`` segment ends in stored u→v order
        """
        xp = _backend(backend)
        pos = xp.asarray(self.pos if pos is None else pos)
        vel = xp.asarray(self.v if v is None else v)
        u, vv = self.u_idx, self.v_idx
        p0, p1 = pos[u], pos[vv]
        v0 = xp.asarray(self.u_sign)[:, None] * vel[u]
        v1 = xp.asarray(self.v_sign)[:, None] * vel[vv]
        m0 = xp.asarray(self.port_v0_mask)[:, None]
        m1 = xp.asarray(self.port_v1_mask)[:, None]
        v0 = xp.where(m0, xp.asarray(self.port_v0), v0)
        v1 = xp.where(m1, xp.asarray(self.port_v1), v1)
        return xp.stack([p0, v0, p1, v1], axis=1)

    def refresh(
        self, pos=None, v=None, ts_count: int = 6, backend: str = "numpy", store: bool = False
    ):
        """Sample all Hermite segments from ``pos``/``v``.

        Args:
            pos: node positions ``(N, 3)``; default ``self.pos``
            v: node tangents ``(N, 3)``; default ``self.v``
            ts_count: samples per segment (inclusive endpoints)
            backend: ``'numpy'`` | ``'jax'``
            store: if True (numpy only), write ``seg_ends`` / ``seg_xyz`` / ``seg_radius``

        Returns:
            ends: ``(S, 4, 3)`` from ``hermite_ends``
            xyz: ``(S, T, 3)`` curve samples
            radius: ``(S, T)`` linearly interpolated radii
        """
        xp = _backend(backend)
        ends = self.hermite_ends(pos, v, backend)
        T = max(int(ts_count), 2)
        xyz = sample_hermite(ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3], n=T, backend=backend)
        ts = xp.linspace(0.0, 1.0, T)
        r0 = xp.asarray(self.radius)[self.u_idx]
        r1 = xp.asarray(self.radius)[self.v_idx]
        radius = (1.0 - ts[None, :]) * r0[:, None] + ts[None, :] * r1[:, None]
        if store:
            assert backend == "numpy", "store=True only supported for numpy backend"
            self.seg_ends, self.seg_xyz, self.seg_radius = ends, xyz, radius
        return ends, xyz, radius

    def check_usage_tangents_colinear(self, atol: float = 1e-6) -> None:
        def _same_end_derivative(a, b) -> bool:
            a = np.asarray(a, dtype=float).reshape(3)
            b = np.asarray(b, dtype=float).reshape(3)
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if not np.isclose(na, nb, rtol=atol, atol=atol):
                return False
            if na <= atol or nb <= atol:
                return True
            return abs(abs(float(np.dot(a / na, b / nb))) - 1.0) <= atol

        claims: list[dict[str, list]] = [{"u": [], "v": []} for _ in self.segments]
        for path in self.usage_paths:
            nseg = len(path.segments)
            for i, (s, rev) in enumerate(zip(path.segments, path.reverse)):
                if i == 0:
                    if rev:
                        claims[s]["v"].append(-np.asarray(path.v_start, dtype=float))
                    else:
                        claims[s]["u"].append(np.asarray(path.v_start, dtype=float))
                if i == nseg - 1:
                    if rev:
                        claims[s]["u"].append(-np.asarray(path.v_end, dtype=float))
                    else:
                        claims[s]["v"].append(np.asarray(path.v_end, dtype=float))

        for s, ends in enumerate(claims):
            for which, vecs in ends.items():
                if len(vecs) < 2:
                    continue
                ref = vecs[0]
                for k, vec in enumerate(vecs[1:], start=1):
                    if not _same_end_derivative(ref, vec):
                        na = float(np.linalg.norm(ref))
                        nb = float(np.linalg.norm(vec))
                        raise ValueError(
                            f"OptimizeGraph {self.id!r}: segment {s} stored-{which} "
                            f"has inconsistent usage derivatives (claim 0 vs {k}: "
                            f"||v||={na:.6g} vs {nb:.6g})"
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
        for i, (s, rev) in enumerate(zip(path.segments, path.reverse)):
            seg = self.segments[s]
            iu, iv = seg.u, seg.v
            if rev:
                p0, v0 = self.pos[iv], -self.v_sign[s] * self.v[iv]
                p1, v1 = self.pos[iu], -self.u_sign[s] * self.v[iu]
                r0, r1 = float(self.radius[iv]), float(self.radius[iu])
            else:
                p0, v0 = self.pos[iu], self.u_sign[s] * self.v[iu]
                p1, v1 = self.pos[iv], self.v_sign[s] * self.v[iv]
                r0, r1 = float(self.radius[iu]), float(self.radius[iv])
            if i == 0:
                v0 = path.v_start
            if i == nseg - 1:
                v1 = path.v_end
            n = HermiteSegment.ts_count_for(p0, v0, p1, v1, resolution)
            xyz = sample_hermite(p0, v0, p1, v1, n=n, backend="numpy")
            ts = np.linspace(0.0, 1.0, n)
            if pts:
                xyz, ts = xyz[1:], ts[1:]
            pts.append(xyz)
            rads.append((1.0 - ts) * r0 + ts * r1)
        if not pts:
            return {"usage": path.usage, "xyz": np.zeros((0, 3)), "radius": np.zeros((0,))}
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

    # assemble-time string keys → int node index (dropped after build)
    name_to_i: dict[str, int] = {}
    nodes: list[HermiteNode] = []
    pos_tmp: list[np.ndarray] = []
    v_tmp: list[np.ndarray | None] = []
    radius_tmp: list[float] = []
    segments: list[HermiteSegment] = []
    u_signs: list[float | None] = []
    v_signs: list[float | None] = []
    edge_segs: dict[str, list[int]] = {}

    def add_node(kid, pos, role, radius) -> int:
        if kid in name_to_i:
            return name_to_i[kid]
        is_port = role == "port"
        name_to_i[kid] = len(nodes)
        nodes.append(HermiteNode(fixed_p=is_port, fixed_v=is_port, role=role))
        pos_tmp.append(np.asarray(pos, dtype=float))
        v_tmp.append(None)
        radius_tmp.append(float(radius))
        return name_to_i[kid]

    def densify_edge(e: LabeledEdge):
        nids = e.nodes
        if len(nids) < 2:
            raise ValueError(f"Edge {e.id} has less than 2 nodes")
        segs_here: list[int] = []

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

            chain_i: list[int] = []
            for kid, p, r in zip(chain_ids, chain_pos, chain_r):
                if kid in shared:
                    role = "port" if (kid in jnodes and jnodes[kid].terminal_id) else "interface"
                    r = float(jnodes[kid].radius) if kid in jnodes else r
                else:
                    role = e.kind
                chain_i.append(add_node(kid, p, role, r))

            for j in range(len(chain_i) - 1):
                p0, p1 = chain_pos[j], chain_pos[j + 1]
                seg_L = float(np.linalg.norm(p1 - p0))
                tang = unit(p1 - p0, backend="numpy")
                u_id, v_id = chain_i[j], chain_i[j + 1]

                if e.kind == "trunk":
                    # trunk anchors node.v; stored frame signs are +1
                    if not nodes[u_id].fixed_v and v_tmp[u_id] is None:
                        v_tmp[u_id] = tang * seg_L
                    if not nodes[v_id].fixed_v and v_tmp[v_id] is None:
                        v_tmp[v_id] = tang * seg_L
                    u_sign = v_sign = 1.0
                else:
                    # junction: init unset knots; interface already set → sign later
                    if v_tmp[u_id] is None and not nodes[u_id].fixed_v:
                        v_tmp[u_id] = tang * seg_L
                        u_sign = 1.0
                    else:
                        u_sign = None if not nodes[u_id].fixed_v else 1.0
                    if v_tmp[v_id] is None and not nodes[v_id].fixed_v:
                        v_tmp[v_id] = tang * seg_L
                        v_sign = 1.0
                    else:
                        v_sign = None if not nodes[v_id].fixed_v else 1.0

                segs_here.append(len(segments))
                segments.append(HermiteSegment(edge=e.id, kind=e.kind, u=u_id, v=v_id))
                u_signs.append(u_sign)
                v_signs.append(v_sign)
        edge_segs[e.id] = segs_here

    # trunk first (anchors shared v), then junction
    for e in junction.trunk_edges:
        densify_edge(e)
    for e in junction.junction_edges:
        densify_edge(e)

    usage_paths: list[OptimizeUsagePath] = []
    for up in junction.usage_paths:
        seg_ids, rev_flags = [], []
        for eid, rev in zip(up.edges, up.reverse):
            segs = list(edge_segs[eid])
            if rev:
                segs.reverse()
            for s in segs:
                seg_ids.append(s)
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

    # fill junction interface signs from travel continuity (adjacent reverse)
    for path in usage_paths:
        for i in range(len(path.segments) - 1):
            sa, ra = path.segments[i], path.reverse[i]
            sb, rb = path.segments[i + 1], path.reverse[i + 1]
            sA, sB = (-1.0 if ra else 1.0), (-1.0 if rb else 1.0)
            eA = u_signs[sa] if ra else v_signs[sa]
            eB = v_signs[sb] if rb else u_signs[sb]
            if eA is None and eB is None:
                raise RuntimeError(f"both signs unset at joint {sa}|{sb}")
            if eA is None:
                eA = sA * sB * eB  # type: ignore[operator]
                if ra:
                    u_signs[sa] = eA
                else:
                    v_signs[sa] = eA
            elif eB is None:
                eB = sB * sA * eA
                if rb:
                    v_signs[sb] = eB
                else:
                    u_signs[sb] = eB
            else:
                if not np.isclose(sA * eA, sB * eB):
                    raise RuntimeError(
                        f"sign mismatch at joint {sa}|{sb}: {sA*eA} vs {sB*eB}"
                    )

    for s in range(len(segments)):
        if u_signs[s] is None or v_signs[s] is None:
            raise RuntimeError(f"segment {s}: unresolved u_sign/v_sign")

    pos = np.stack(pos_tmp) if pos_tmp else np.zeros((0, 3))
    radius = np.asarray(radius_tmp, dtype=float)
    v = np.zeros((len(nodes), 3), dtype=float)
    for i, vv in enumerate(v_tmp):
        if vv is not None:
            v[i] = vv

    g = OptimizeGraph(
        id=junction.id,
        nodes=nodes,
        segments=segments,
        pos=pos,
        v=v,
        radius=radius,
        u_sign=np.asarray(u_signs, dtype=float),
        v_sign=np.asarray(v_signs, dtype=float),
        edge_segs=edge_segs,
        junctions={Z.id: list(Z.edges) for Z in junction.complexes},
        usage_paths=usage_paths,
    )
    g.check_usage_tangents_colinear()
    return g.bind_ends_tables()
