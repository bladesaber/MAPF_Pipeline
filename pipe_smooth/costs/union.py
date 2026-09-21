"""§6.4.3 — junction union pair finding (detection outside AD).

Per complex Z: sample Hermite segs on Z's edges; KD-tree NN across usages;
freeze (junction, edge, seg, t[]) hits with d < d_hi for a later AD cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.neighbors import KDTree

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import sample_hermite


@dataclass(frozen=True)
class UnionHit:
    """Directed Chamfer matches inside one complex: samples on a → NN on b.

    One record per (junction, edge_a, edge_b, seg_a, seg_b); ``ts_*`` / ``d`` are (M,).
    """

    junction: str
    edge_a: str
    edge_b: str
    seg_a: str
    seg_b: str
    ts_a: np.ndarray  # (M,)
    ts_b: np.ndarray  # (M,)
    d: np.ndarray  # (M,)


def find_union_pairs(graph: OptimizeGraph, alpha: float = 1.0) -> list[UnionHit]:
    """KD-tree NN pairs per junction complex (§6.4.3).

    Only segments whose ``edge`` is in ``graph.junctions[Z]``. Both Chamfer
    directions. Keep d < α (r_a + r_b); group matches by segment pair into arrays.
    """
    hits: list[UnionHit] = []

    for zid, z_edges in graph.junctions.items():
        edge_set = set(z_edges)
        curves: dict[tuple[str, str], tuple] = {}
        for path in graph.usage_paths:
            pts, rads, eids, sids, tvals = [], [], [], [], []
            nseg = len(path.segments)
            for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
                seg = graph.segments[sid]
                if seg.edge not in edge_set:
                    continue
                nu, nv = graph.nodes[seg.u], graph.nodes[seg.v]
                if rev:
                    p0, v0 = nv.pos, -nv.v  # type: ignore[operator]
                    p1, v1 = nu.pos, -nu.v  # type: ignore[operator]
                    r0, r1 = float(nv.radius), float(nu.radius)
                else:
                    p0, v0 = nu.pos, nu.v
                    p1, v1 = nv.pos, nv.v
                    r0, r1 = float(nu.radius), float(nv.radius)
                if i == 0:
                    v0 = path.v_start
                if i == nseg - 1:
                    v1 = path.v_end
                n = max(2, int(seg.ts_count))
                xyz = sample_hermite(p0, v0, p1, v1, n=n, backend="numpy")
                ts = np.linspace(0.0, 1.0, n)
                radius = (1.0 - ts) * r0 + ts * r1
                if pts:
                    xyz, ts, radius = xyz[1:], ts[1:], radius[1:]
                m = len(xyz)
                pts.append(xyz)
                rads.append(radius)
                eids.append(np.full(m, seg.edge, dtype=object))
                sids.append(np.full(m, sid, dtype=object))
                tvals.append(ts)
            if pts:
                curves[path.usage] = (
                    np.concatenate(pts),
                    np.concatenate(rads),
                    np.concatenate(eids),
                    np.concatenate(sids),
                    np.concatenate(tvals),
                )

        for ua, ub in combinations(curves.keys(), 2):
            xa, ra, ea, sa, ta = curves[ua]
            xb, rb, eb, sb, tb = curves[ub]
            for src_x, src_r, src_e, src_s, src_t, dst_x, dst_r, dst_e, dst_s, dst_t in (
                (xa, ra, ea, sa, ta, xb, rb, eb, sb, tb),
                (xb, rb, eb, sb, tb, xa, ra, ea, sa, ta),
            ):
                tree = KDTree(dst_x)
                d, jj = tree.query(src_x, k=1)
                d = np.asarray(d, float).reshape(-1)
                jj = np.asarray(jj, int).reshape(-1)
                mask = d < float(alpha) * (src_r + dst_r[jj])
                if not np.any(mask):
                    continue
                jj_m = jj[mask]
                d_m = d[mask]
                ea_m = src_e[mask]
                eb_m = dst_e[jj_m]
                sa_m = src_s[mask]
                sb_m = dst_s[jj_m]
                ta_m = src_t[mask]
                tb_m = dst_t[jj_m]

                rec = np.empty(
                    d_m.shape[0],
                    dtype=[("ea", "O"), ("eb", "O"), ("sa", "O"), ("sb", "O")],
                )
                rec["ea"], rec["eb"], rec["sa"], rec["sb"] = ea_m, eb_m, sa_m, sb_m
                uniq, inv = np.unique(rec, return_inverse=True)
                for k, u in enumerate(uniq):
                    m = inv == k
                    hits.append(
                        UnionHit(
                            zid,
                            str(u["ea"]),
                            str(u["eb"]),
                            str(u["sa"]),
                            str(u["sb"]),
                            ta_m[m],
                            tb_m[m],
                            d_m[m],
                        )
                    )
    return hits
