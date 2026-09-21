"""§6.4.3 — junction union: pair finding (outside AD) + soft cost (AD).

Detection: sample interiors on Z's edges; KD-tree NN; freeze hits with d < d_hi.
Cost: re-eval Hermite at frozen ts; λ Σ ||x_a-x_b||² Δs_a (Chamfer source weight).
Requires refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.neighbors import KDTree

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import eval_hermite_ts


@dataclass(frozen=True)
class UnionHit:
    """Directed Chamfer matches inside one complex: samples on a → NN on b.

    One record per (junction, edge_a, edge_b, seg_a, seg_b); arrays length (M,).
    ``ts_*`` are Hermite parameters in stored u→v (same as ``HermiteSegment.samples``).
    ``ds_a`` is source-sample Δs (½ of adjacent chords) for Chamfer weight.
    """

    junction: str
    edge_a: str
    edge_b: str
    seg_a: str
    seg_b: str
    ts_a: np.ndarray  # (M,)
    ts_b: np.ndarray  # (M,)
    d: np.ndarray  # (M,) detection distance (debug / diagnostics)
    ds_a: np.ndarray  # (M,) arc-length weight on source samples


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
            pts, rads, eids, sids, tvals, dls = [], [], [], [], [], []
            for sid in path.segments:
                seg = graph.segments[sid]
                if seg.edge not in edge_set:
                    continue
                ts, xyz, radius = seg.samples(backend="numpy")
                ts = np.asarray(ts, float)
                xyz = np.asarray(xyz, float)
                radius = np.asarray(radius, float)
                # Interiors only — endpoints are ports/joints, not union Chamfer.
                if xyz.shape[0] < 3:
                    raise RuntimeError(f"segment {sid}: not enough samples for union Chamfer")
                # Δs on full chain, then take interior samples (ROADMAP Δℓ)
                edge_len = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
                dl = 0.5 * (edge_len[:-1] + edge_len[1:])  # aligns with xyz[1:-1]
                xyz, ts, radius = xyz[1:-1], ts[1:-1], radius[1:-1]
                m = len(xyz)
                pts.append(xyz)
                rads.append(radius)
                eids.append(np.full(m, seg.edge, dtype=object))
                sids.append(np.full(m, sid, dtype=object))
                tvals.append(ts)
                dls.append(dl)
            if pts:
                curves[path.usage] = (
                    np.concatenate(pts),
                    np.concatenate(rads),
                    np.concatenate(eids),
                    np.concatenate(sids),
                    np.concatenate(tvals),
                    np.concatenate(dls),
                )

        for ua, ub in combinations(curves.keys(), 2):
            xa, ra, ea, sa, ta, dla = curves[ua]
            xb, rb, eb, sb, tb, dlb = curves[ub]
            for src_x, src_r, src_e, src_s, src_t, src_dl, dst_x, dst_r, dst_e, dst_s, dst_t in (
                (xa, ra, ea, sa, ta, dla, xb, rb, eb, sb, tb),
                (xb, rb, eb, sb, tb, dlb, xa, ra, ea, sa, ta),
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
                dl_m = src_dl[mask]

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
                            dl_m[m],
                        )
                    )
    return hits


def union_cost(
    hits: Sequence[UnionHit],
    graph: OptimizeGraph,
    lam: float = 1.0,
    backend: str = "numpy",
):
    """Soft J_union (§6.4.3): λ Σ ||x_a(t_a)-x_b(t_b)||² Δs_a on frozen Chamfer hits."""
    xp = _backend(backend)
    ends = graph.hermite_ends(backend=backend)
    j = xp.asarray(0.0)
    for h in hits:
        a0, av0, a1, av1 = ends[h.seg_a]
        b0, bv0, b1, bv1 = ends[h.seg_b]
        xa = eval_hermite_ts(a0, av0, a1, av1, h.ts_a, backend=backend)
        xb = eval_hermite_ts(b0, bv0, b1, bv1, h.ts_b, backend=backend)
        d2 = xp.sum((xa - xb) ** 2, axis=-1)
        j = j + xp.sum(d2 * xp.asarray(h.ds_a))
    return lam * j
