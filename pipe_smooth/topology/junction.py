"""§6.2 — label junction/trunk on the chain graph, absorb short trunks, extract paths."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import heapq
import math

from pipe_smooth.utils import to_float_list

from .tree import GroupTree, TreeNode


@dataclass
class LabeledEdge:
    """Trunk or junction run on the fine chain (deduped across usages).

    Positions live in PartitionedTree.nodes[id].pos — look up via `nodes` (u→…→v).
    Radius lives on nodes; interpolate between endpoint node radii when needed.
    """

    id: str
    kind: str  # "trunk" | "junction"
    u: str
    v: str
    nodes: list[str] = field(default_factory=list)

    def export(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "u": self.u,
            "v": self.v,
            "nodes": list(self.nodes),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "LabeledEdge":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            u=str(data["u"]),
            v=str(data["v"]),
            nodes=[str(n) for n in data.get("nodes") or []],
        )


@dataclass
class UsagePath:
    """One usage as ordered labeled-edge ids plus travel direction vs stored edge.

    `reverse[i]` True ⇒ traverse edges[i] as nodes[::-1] (stored u→v is opposite usage).
    Port clamps: v_start / v_end from domain face normals (start = inward, end = outward);
    sigma_* = tangent magnitude scales (default 1.0).
    """

    usage: tuple[str, str]
    edges: list[str]
    reverse: list[bool]
    v_start: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    sigma_start: float = 1.0
    v_end: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    sigma_end: float = 1.0

    def export(self) -> dict[str, Any]:
        return {
            "usage": [self.usage[0], self.usage[1]],
            "edges": list(self.edges),
            "reverse": [bool(x) for x in self.reverse],
            "v_start": to_float_list(self.v_start),
            "sigma_start": float(self.sigma_start),
            "v_end": to_float_list(self.v_end),
            "sigma_end": float(self.sigma_end),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "UsagePath":
        u = data["usage"]
        return cls(
            usage=(str(u[0]), str(u[1])),
            edges=[str(e) for e in data.get("edges") or []],
            reverse=[bool(x) for x in data.get("reverse") or []],
            v_start=to_float_list(data.get("v_start", [1.0, 0.0, 0.0])),
            sigma_start=float(data.get("sigma_start", 1.0)),
            v_end=to_float_list(data.get("v_end", [1.0, 0.0, 0.0])),
            sigma_end=float(data.get("sigma_end", 1.0)),
        )


@dataclass
class JunctionComplex:
    """One junction connected component (metadata)."""

    id: str
    seeds: list[str]
    interfaces: list[str]
    nodes: list[str]
    edges: list[str]

    def export(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seeds": list(self.seeds),
            "interfaces": list(self.interfaces),
            "nodes": list(self.nodes),
            "edges": list(self.edges),
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "JunctionComplex":
        return cls(
            id=str(data["id"]),
            seeds=[str(x) for x in data.get("seeds") or []],
            interfaces=[str(x) for x in data.get("interfaces") or []],
            nodes=[str(x) for x in data.get("nodes") or []],
            edges=[str(x) for x in data.get("edges") or []],
        )


@dataclass
class PartitionedTree:
    """GroupTree after §6.2 trunk/junction partition (fine nodes + labeled edges).

    Same connectivity as 𝒯; junction is a label/feature, not a separate tree type.
    """

    id: str
    # Fine chain vertices: structural GroupTree nodes ∪ intermediate samples along
    # every coarse edge polyline (ids like '{edge_id}_p{i}'). Unlike GroupTree.nodes,
    # this is dense — not terminals/branches only.
    nodes: dict[str, TreeNode]
    labels: dict[str, str]  # node_id -> "trunk" | "junction"
    trunk_edges: list[LabeledEdge]
    junction_edges: list[LabeledEdge]
    usage_paths: list[UsagePath]
    complexes: list[JunctionComplex]

    def export(self, path: str | Path) -> None:
        """Write partitioned tree JSON (for debug / skip rebuild)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "nodes": {nid: n.export() for nid, n in self.nodes.items()},
            "labels": dict(self.labels),
            "trunk_edges": [e.export() for e in self.trunk_edges],
            "junction_edges": [e.export() for e in self.junction_edges],
            "usage_paths": [p.export() for p in self.usage_paths],
            "complexes": [c.export() for c in self.complexes],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PartitionedTree":
        data = json.loads(Path(path).read_text())
        return cls(
            id=str(data["id"]),
            nodes={nid: TreeNode.load(n) for nid, n in (data.get("nodes") or {}).items()},
            labels={str(k): str(v) for k, v in (data.get("labels") or {}).items()},
            trunk_edges=[LabeledEdge.load(e) for e in data.get("trunk_edges") or []],
            junction_edges=[LabeledEdge.load(e) for e in data.get("junction_edges") or []],
            usage_paths=[UsagePath.load(p) for p in data.get("usage_paths") or []],
            complexes=[JunctionComplex.load(c) for c in data.get("complexes") or []],
        )


def _face_inward(pos, domain: dict, tol: float | None = None) -> list[float]:
    """Unit inward normal of the domain face nearest to `pos`.

    Domain box [0, size_x*res] × [0, size_y*res] × [0, size_z*res] (from grid YAML).
    E.g. terminal on x≈0 → (1,0,0). Raises if on an edge/corner (near ≥2 faces).
    """
    res = float(domain.get("resolution", 1.0))
    sx = float(domain["size_x"]) * res
    sy = float(domain["size_y"]) * res
    sz = float(domain["size_z"]) * res
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if tol is None:
        tol = 0.51 * res
    faces = [
        (x - 0.0, [1.0, 0.0, 0.0], "x=0"),
        (sx - x, [-1.0, 0.0, 0.0], "x=max"),
        (y - 0.0, [0.0, 1.0, 0.0], "y=0"),
        (sy - y, [0.0, -1.0, 0.0], "y=max"),
        (z - 0.0, [0.0, 0.0, 1.0], "z=0"),
        (sz - z, [0.0, 0.0, -1.0], "z=max"),
    ]
    near = [(d, n, name) for d, n, name in faces if d <= tol]
    if len(near) == 0:
        raise ValueError(f"terminal pos {pos} not near any domain face")
    if len(near) > 1:
        names = [name for _, _, name in near]
        raise ValueError(
            f"terminal pos {pos} lies on multiple faces {names} (edge/corner); "
            f"expected exactly one face"
        )
    return near[0][1]


def build_junctions(
    tree: GroupTree,
    domain: dict,
    tau0: float = 1.0,
    alpha: float = 2.0,
    beta: float = 0.0,
    R_bend: float = 1.0,
    tau_trunk: float | None = None,
) -> PartitionedTree:
    """Grow junction by tree length ≤ τ(seed), then rebuild paths from usages.

    `domain` from instance grid/meta: size_x, size_y, size_z, resolution.
    1. Chain every edge polyline into a fine graph (structural ends keep tree ids).
    2. Nodes with dist_to_seed ≤ τ(seed) → junction; else trunk.
    3. If tau_trunk is set, short trunk components flip to junction.
    4. Walk each usage, split by label into trunk/junction runs, dedupe as edges;
       each usage becomes a list of those edge ids + face-based port tangents.
    """
    base = tree.nodes
    edges = tree.edges
    deg = defaultdict(int)
    for e in edges:
        deg[e.u] += 1
        deg[e.v] += 1
    seeds = [nid for nid, d in deg.items() if d >= 3]

    def tau_of(nid: str) -> float:
        return max(tau0, alpha * float(base[nid].radius), beta * R_bend)

    # --- fine chain graph (samples along edges) ---
    nodes: dict[str, TreeNode] = {
        nid: TreeNode(pos=list(n.pos), radius=n.radius, terminal_id=n.terminal_id)
        for nid, n in base.items()
    }
    adj = defaultdict(list)  # fid -> [(nbr, seg_len, src_edge)]

    for e in edges:
        xyz = e.xyz
        ids = []
        for i, p in enumerate(xyz):
            if i == 0:
                fid = e.u
            elif i == len(xyz) - 1:
                fid = e.v
            else:
                fid = f"{e.id}_p{i}"
                nodes[fid] = TreeNode(pos=list(p), radius=e.radius, terminal_id=None)
            ids.append(fid)
        for a, b in zip(ids[:-1], ids[1:]):
            L = math.dist(nodes[a].pos, nodes[b].pos)
            adj[a].append((b, L, e.id))
            adj[b].append((a, L, e.id))

    if not seeds:
        return PartitionedTree(
            id=tree.id,
            nodes=nodes,
            labels={fid: "trunk" for fid in nodes},
            trunk_edges=[],
            junction_edges=[],
            usage_paths=[],
            complexes=[],
        )

    # --- dist to nearest seed ---
    dist = {fid: math.inf for fid in nodes}
    owner = {fid: None for fid in nodes}
    pq = []
    for s in seeds:
        dist[s] = 0.0
        owner[s] = s
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist[x]:
            continue
        for y, L, _ in adj[x]:
            nd = d + L
            if nd < dist[y]:
                dist[y] = nd
                owner[y] = owner[x]
                heapq.heappush(pq, (nd, y))

    # --- initial junction / trunk labels ---
    label: dict[str, str] = {}
    for fid in nodes:
        o = owner[fid]
        label[fid] = "junction" if dist[fid] <= tau_of(o) else "trunk"  # type: ignore
    for s in seeds:
        label[s] = "junction"

    # --- absorb short trunk lines into junction ---
    def components(want: str):
        seen, comps = set(), []
        for f in nodes:
            if f in seen or label[f] != want:
                continue
            comp, dq = [], deque([f])
            seen.add(f)
            while dq:
                x = dq.popleft()
                comp.append(x)
                for y, L, _ in adj[x]:
                    if y not in seen and label[y] == want:
                        seen.add(y)
                        dq.append(y)
            comps.append(comp)
        return comps

    if tau_trunk is not None:
        for comp in components("trunk"):
            ids = set(comp)
            Lsum = 0.0
            for x in comp:
                for y, L, _ in adj[x]:
                    if y in ids and x < y:
                        Lsum += L
            if Lsum < tau_trunk:
                for x in comp:
                    label[x] = "junction"

    # --- walk every usage on fine adj: split by label → dedupe edges ---
    term_node = tree.terminal_node
    trunk_map: dict = {}
    junc_map: dict = {}
    usage_paths: list[UsagePath] = []

    for src_t, dst_t in tree.usages:
        s, t = term_node[src_t], term_node[dst_t]
        prev = {s: None}
        dq = deque([s])
        while dq and t not in prev:
            x = dq.popleft()
            for y, _, __ in adj[x]:
                if y not in prev:
                    prev[y] = x  # type: ignore
                    dq.append(y)
        if t not in prev:
            continue
        fine = [t]
        while fine[-1] != s:
            fine.append(prev[fine[-1]])  # type: ignore
        fine.reverse()

        edge_ids = []
        reverse_flags = []
        i = 0
        while i < len(fine):
            kind = label[fine[i]]
            j = i + 1
            while j < len(fine) and label[fine[j]] == kind:
                j += 1
            # Trunk owns the trunk↔junction segment: absorb adjacent interface
            # nodes so consecutive labeled edges share an endpoint (no gap).
            if kind == "trunk":
                # Absorb adjacent interfaces (shared endpoints with junction runs).
                travel = fine[max(0, i - 1) : min(len(fine), j + 1)]
            else:
                travel = fine[i:j]
            if kind == "junction" and len(travel) < 3:
                raise ValueError(
                    f"junction run in group {tree.id} has {len(travel)} node(s), "
                    f"need >= 3 (too small τ or degenerate junction)"
                )
            # canonicalize storage by node-id order (same as LabeledEdge.u/v)
            canon = list(reversed(travel)) if travel[0] > travel[-1] else list(travel)
            rev = travel[0] != canon[0]
            key = (kind, tuple(canon))
            store = trunk_map if kind == "trunk" else junc_map
            if key not in store:
                eid = f"{'t' if kind == 'trunk' else 'j'}{len(store)}"
                store[key] = LabeledEdge(
                    id=eid,
                    kind=kind,
                    u=canon[0],
                    v=canon[-1],
                    nodes=canon,
                )
            edge_ids.append(store[key].id)
            reverse_flags.append(rev)
            i = j

        # port clamps from domain faces: start = inward, end = outward
        nid_s = term_node[src_t]
        nid_t = term_node[dst_t]
        pos_s = base[nid_s].pos
        pos_t = base[nid_t].pos
        v_s = _face_inward(pos_s, domain)
        v_e = [-c for c in _face_inward(pos_t, domain)]
        sig_s = 1.0
        sig_e = 1.0

        usage_paths.append(
            UsagePath(
                usage=(src_t, dst_t),
                edges=edge_ids,
                reverse=reverse_flags,
                v_start=v_s,
                sigma_start=sig_s,
                v_end=v_e,
                sigma_end=sig_e,
            )
        )

    trunk_edges = list(trunk_map.values())
    junction_edges = list(junc_map.values())

    complexes = []
    for ci, comp in enumerate(components("junction")):
        ids = set(comp)
        interfaces = sorted(
            x for x in comp if any(label[y] == "trunk" for y, _, __ in adj[x])
        )
        for x in interfaces:
            d = len(adj[x])
            if d not in (1, 2):
                raise ValueError(
                    f"interface {x} in group {tree.id} has degree {d}, expected 1 or 2 "
                    f"(interface must not be a seed / branch)"
                )
        complexes.append(
            JunctionComplex(
                id=f"Z{ci}",
                seeds=sorted(s for s in seeds if s in ids),
                interfaces=interfaces,
                nodes=sorted(comp),
                edges=[e.id for e in junction_edges if any(n in ids for n in e.nodes)],
            )
        )

    return PartitionedTree(
        id=tree.id,
        nodes=nodes,
        labels=label,
        trunk_edges=trunk_edges,
        junction_edges=junction_edges,
        usage_paths=usage_paths,
        complexes=complexes,
    )


def build_junctions_all(trees: list[GroupTree], **kwargs) -> list[PartitionedTree]:
    return [build_junctions(T, **kwargs) for T in trees]
