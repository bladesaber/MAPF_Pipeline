"""§6.1 — coarse polylines + terminals → undirected tree 𝒯 (per group)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipe_smooth.utils import to_float_list


@dataclass
class TreeNode:
    """Structural tree vertex: terminal, branch, or path endpoint (not dense samples)."""

    pos: list[float]
    radius: float
    terminal_id: str | None = None

    def export(self) -> dict[str, Any]:
        return {
            "pos": to_float_list(self.pos),
            "radius": float(self.radius),
            "terminal_id": self.terminal_id,
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "TreeNode":
        tid = data.get("terminal_id")
        return cls(
            pos=to_float_list(data["pos"]),
            radius=float(data["radius"]),
            terminal_id=None if tid is None else str(tid),
        )


@dataclass
class TreeEdge:
    """Undirected tree edge with coarse polyline embedding."""

    id: str
    u: str
    v: str
    xyz: list[list[float]]
    radius: float
    usages: list[list[str]] = field(default_factory=list)

    def export(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "u": self.u,
            "v": self.v,
            "xyz": [to_float_list(p) for p in self.xyz],
            "radius": float(self.radius),
            "usages": [[str(a), str(b)] for a, b in self.usages],
        }

    @classmethod
    def load(cls, data: dict[str, Any]) -> "TreeEdge":
        return cls(
            id=str(data["id"]),
            u=str(data["u"]),
            v=str(data["v"]),
            xyz=[to_float_list(p) for p in data.get("xyz") or []],
            radius=float(data["radius"]),
            usages=[[str(a), str(b)] for a, b in data.get("usages") or []],
        )


@dataclass
class GroupTree:
    """One group's undirected tree 𝒯 (§6.1).

    `nodes` = terminals ∪ branch points (on ≥2 polylines) ∪ other path endpoints.
    """

    id: str
    nodes: dict[str, TreeNode]
    edges: list[TreeEdge]
    terminal_node: dict[str, str]  # terminal_id -> node_id
    usages: list[tuple[str, str]]

    def export(self, path: str | Path) -> None:
        """Write group tree JSON (for debug / skip rebuild)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "GroupTree",
            "id": self.id,
            "nodes": {nid: n.export() for nid, n in self.nodes.items()},
            "edges": [e.export() for e in self.edges],
            "terminal_node": {str(k): str(v) for k, v in self.terminal_node.items()},
            "usages": [[a, b] for a, b in self.usages],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GroupTree":
        data = json.loads(Path(path).read_text())
        return cls(
            id=str(data["id"]),
            nodes={nid: TreeNode.load(n) for nid, n in (data.get("nodes") or {}).items()},
            edges=[TreeEdge.load(e) for e in data.get("edges") or []],
            terminal_node={str(k): str(v) for k, v in (data.get("terminal_node") or {}).items()},
            usages=[(str(a), str(b)) for a, b in data.get("usages") or []],
        )


def build_tree(group: dict, paths: dict, usages: dict, tol: float = 1e-6) -> GroupTree:
    """Build one group's tree from solution polylines."""
    terminals = group["terminals"]
    raw_paths = paths.get("paths") or []
    polylines = [list(map(list, p["xyz"])) for p in raw_paths]
    path_radii = [float(p["radius"]) for p in raw_paths]
    usage_pairs = [tuple(u) for u in usages.get("usages") or []]
    r_default = float(terminals[0]["radius"]) if terminals else 0.5

    def key(p):
        return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))

    hit = Counter()
    for poly in polylines:
        for p in poly:
            hit[key(p)] += 1

    term_by_key = {key(t["pos"]): t for t in terminals}
    node_keys = set(term_by_key)
    node_keys.update(k for k, c in hit.items() if c >= 2)
    for poly in polylines:
        if poly:
            node_keys.add(key(poly[0]))
            node_keys.add(key(poly[-1]))

    nodes: dict[str, TreeNode] = {}
    key_to_nid = {}
    terminal_node = {}
    for i, k in enumerate(sorted(node_keys)):
        nid = f"n{i}"
        key_to_nid[k] = nid
        term = term_by_key.get(k)
        if term is not None:
            pos = list(term["pos"])
            radius = float(term["radius"])
            terminal_node[term["id"]] = nid
            tid = term["id"]
        else:
            pos = next(p for poly in polylines for p in poly if key(p) == k)
            radius = r_default
            tid = None
        nodes[nid] = TreeNode(pos=pos, radius=radius, terminal_id=tid)

    edges: list[TreeEdge] = []
    adj = defaultdict(list)  # nid -> list of (nbr, edge_index)
    for poly, r_path in zip(polylines, path_radii):
        cuts = [i for i, p in enumerate(poly) if key(p) in node_keys]
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            ka, kb = key(poly[i0]), key(poly[i1])
            assert ka != kb, f"degenerate edge in group {group['id']} at poly indices {i0}:{i1}"
            ua, ub = key_to_nid[ka], key_to_nid[kb]
            xyz = [list(p) for p in poly[i0 : i1 + 1]]
            eid = f"e{len(edges)}"
            edges.append(
                TreeEdge(id=eid, u=ua, v=ub, xyz=xyz, radius=r_path, usages=[])
            )
            ei = len(edges) - 1
            adj[ua].append((ub, ei))
            adj[ub].append((ua, ei))

    for src_t, dst_t in usage_pairs:
        s, t = terminal_node[src_t], terminal_node[dst_t]
        prev = {s: None}
        dq = deque([s])
        while dq and t not in prev:
            x = dq.popleft()
            for y, ei in adj[x]:
                if y not in prev:
                    prev[y] = (x, ei)  # type: ignore
                    dq.append(y)
        if t not in prev:
            raise ValueError(f"usage {src_t}->{dst_t}: no tree path in group {group['id']}")
        cur = t
        while cur != s:
            x, ei = prev[cur]  # type: ignore
            pair = [src_t, dst_t]
            if pair not in edges[ei].usages:
                edges[ei].usages.append(pair)
            cur = x

    # Terminals must be leaves (degree 1); a deg≥2 "port" is a topology bug.
    for tid, nid in terminal_node.items():
        d = len(adj[nid])
        if d != 1:
            raise ValueError(
                f"terminal {tid} (node {nid}) in group {group['id']} has degree {d}, "
                f"expected 1"
            )

    return GroupTree(
        id=group["id"],
        nodes=nodes,
        edges=edges,
        terminal_node=terminal_node,
        usages=usage_pairs,
    )


def build_trees(env: dict, tol: float = 1e-6) -> list[GroupTree]:
    """§6.1 for every group in a prepared env dict."""
    by_id_paths = {p["id"]: p for p in env.get("paths") or []}
    by_id_usages = {u["id"]: u for u in env.get("usages") or []}
    return [
        build_tree(g, by_id_paths[g["id"]], by_id_usages[g["id"]], tol=tol)
        for g in env.get("groups") or []
    ]
