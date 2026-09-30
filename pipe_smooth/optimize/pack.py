"""Pack free Hermite DOF into one vector x (1D).

Per graph, free ``pos`` then free ``v`` are contiguous blocks in ``x``.
Bridge: ``free_pos_idx`` / ``free_v_idx`` + ``free_pos_offset`` / ``free_v_offset``
→ few block copies into ``OptimizeGraph.pos`` / ``.v``.

Also holds static ends gather + clear/union hit caps for JIT-stable costs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.utils import backend as _backend


@dataclass
class PackLayout:
    """1D ``x`` layout: per-graph free-DOF bridges into ``graph.pos`` / ``graph.v``.

    For graph ``gi`` (Np free positions, Nv free tangents)::

        x[free_pos_offset[gi] : + 3*Np]  ↔  graph.pos[free_pos_idx[gi]]
        x[free_v_offset[gi]   : + 3*Nv]  ↔  graph.v[free_v_idx[gi]]

    ``ends_off[s,k] >= 0`` → ``x[off:off+3]`` for segment ends.
    """

    n: int
    
    # --- DOF layout ---
    # per graph: which rows of OptimizeGraph.pos / .v are free DOF (packed into x)
    free_pos_idx: list[np.ndarray] = field(default_factory=list)  # → graph.pos[i]
    free_v_idx: list[np.ndarray] = field(default_factory=list)  # → graph.v[i]
    # per graph: start index of that contiguous block inside 1D x
    free_pos_offset: list[int] = field(default_factory=list)
    free_v_offset: list[int] = field(default_factory=list)
    
    # --- ends gather (for JIT-stable clearance / union) ---
    seg_index: dict[tuple[str, int], int] = field(default_factory=dict)  # (group_id, seg_id) → row s in ends_*
    ends_off: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))  # (S,4) x-offsets; -1 = const
    ends_const: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3), dtype=np.float64))  # (S,4,3) when ends_off < 0
    # (S,4): scale on from-x ends only; pos=1, dir=u_sign/v_sign (const already signed)
    ends_sign: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
       
    p_max_obs: int = 0
    p_max_grp: int = 0
    p_max_union: int = 0

    def off_map(self) -> dict[tuple[int, int, str], int]:
        """(gi, node_i, 'pos'|'dir') → x offset."""
        m: dict[tuple[int, int, str], int] = {}
        for gi, (pn, vn, po, vo) in enumerate(
            zip(self.free_pos_idx, self.free_v_idx, self.free_pos_offset, self.free_v_offset)
        ):
            for k, i in enumerate(pn):
                m[(gi, int(i), "pos")] = po + 3 * k
            for k, i in enumerate(vn):
                m[(gi, int(i), "dir")] = vo + 3 * k
        return m

    def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "free_pos_idx": [a.tolist() for a in self.free_pos_idx],
            "free_v_idx": [a.tolist() for a in self.free_v_idx],
            "free_pos_offset": list(self.free_pos_offset),
            "free_v_offset": list(self.free_v_offset),
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def clear_gather(
        graphs: Sequence[OptimizeGraph],
        off_map: dict[tuple[int, int, str], int],
    ) -> tuple[dict[tuple[str, int], int], np.ndarray, np.ndarray, np.ndarray]:
        seg_index: dict[tuple[str, int], int] = {}
        rows_offset: list[list[int]] = []
        rows_const: list[list[list[float]]] = []
        rows_sign: list[list[float]] = []
        for gi, g in enumerate(graphs):
            ends = g.hermite_ends(backend="numpy")
            for s, seg in enumerate(g.segments):
                p0, v0, p1, v1 = ends[s, 0], ends[s, 1], ends[s, 2], ends[s, 3]
                row_offset: list[int] = []
                row_const: list[list[float]] = []
                for ni, fld, val in (
                    (seg.u, "pos", p0),
                    (seg.u, "dir", v0),
                    (seg.v, "pos", p1),
                    (seg.v, "dir", v1),
                ):
                    key = (gi, ni, fld)
                    if key in off_map:
                        row_offset.append(int(off_map[key]))  # x-offset for p0,v0,p1,v1
                        row_const.append([0.0, 0.0, 0.0])  # default value for fixed/port
                    else:
                        row_offset.append(-1)  # -1 = const value
                        row_const.append(np.asarray(val, dtype=float).reshape(3).tolist())  # fixed/port value
                seg_index[(g.id, s)] = len(rows_offset)
                rows_offset.append(row_offset)
                rows_const.append(row_const)
                rows_sign.append([1.0, float(g.u_sign[s]), 1.0, float(g.v_sign[s])])
        if rows_offset:
            ends_off = np.asarray(rows_offset, dtype=np.int32)
            ends_const = np.asarray(rows_const, dtype=np.float64)
            ends_sign = np.asarray(rows_sign, dtype=np.float64)
        else:
            ends_off = np.zeros((0, 4), dtype=np.int32)
            ends_const = np.zeros((0, 4, 3), dtype=np.float64)
            ends_sign = np.zeros((0, 4), dtype=np.float64)
        return seg_index, ends_off, ends_const, ends_sign

    def ends_from_x(self, x, backend: str = "numpy"):
        xp = _backend(backend)
        x = xp.asarray(x)
        off = xp.asarray(self.ends_off)
        const = xp.asarray(self.ends_const)
        sign = xp.asarray(self.ends_sign)[..., None]
        safe = xp.maximum(off, 0)
        from_x = x[safe[:, :, None] + xp.arange(3)] * sign
        return xp.where(off[..., None] >= 0, from_x, const)

    @classmethod
    def load(
        cls,
        path: str | Path,
        graphs: Sequence[OptimizeGraph] | None = None,
        p_max_obs: int | None = None,
        p_max_grp: int | None = None,
        p_max_union: int | None = None,
    ) -> "PackLayout":
        data = json.loads(Path(path).read_text())
        n = int(data["n"])
        free_pos_idx = [np.asarray(a, dtype=np.int32) for a in data["free_pos_idx"]]
        free_v_idx = [np.asarray(a, dtype=np.int32) for a in data["free_v_idx"]]
        free_pos_offset = [int(x) for x in data["free_pos_offset"]]
        free_v_offset = [int(x) for x in data["free_v_offset"]]
        lay = cls(
            n=n,
            free_pos_idx=free_pos_idx,
            free_v_idx=free_v_idx,
            free_pos_offset=free_pos_offset,
            free_v_offset=free_v_offset,
        )
        if (
            graphs is not None
            and p_max_obs is not None
            and p_max_grp is not None
            and p_max_union is not None
        ):
            seg_index, ends_off, ends_const, ends_sign = cls.clear_gather(graphs, lay.off_map())
            lay.seg_index = seg_index
            lay.ends_off, lay.ends_const, lay.ends_sign = ends_off, ends_const, ends_sign
            lay.p_max_obs = int(p_max_obs)
            lay.p_max_grp = int(p_max_grp)
            lay.p_max_union = int(p_max_union)
        return lay


def layout(
    graphs: Sequence[OptimizeGraph],
    p_max_obs: int = 0,
    p_max_grp: int = 0,
    p_max_union: int = 0,
) -> PackLayout:
    """Build 1D x: per graph, contiguous free-pos block then free-v block."""
    free_pos_idx: list[np.ndarray] = []
    free_v_idx: list[np.ndarray] = []
    free_pos_offset: list[int] = []
    free_v_offset: list[int] = []
    n = 0
    for g in graphs:
        pn = np.array([i for i, node in enumerate(g.nodes) if not node.fixed_p], dtype=np.int32)
        vn = np.array([i for i, node in enumerate(g.nodes) if not node.fixed_v], dtype=np.int32)
        free_pos_offset.append(n)
        free_pos_idx.append(pn)
        n += 3 * int(pn.size)
        free_v_offset.append(n)
        free_v_idx.append(vn)
        n += 3 * int(vn.size)

    lay = PackLayout(
        n=n,
        free_pos_idx=free_pos_idx,
        free_v_idx=free_v_idx,
        free_pos_offset=free_pos_offset,
        free_v_offset=free_v_offset,
        p_max_obs=int(p_max_obs),
        p_max_grp=int(p_max_grp),
        p_max_union=int(p_max_union),
    )
    seg_index, ends_off, ends_const, ends_sign = PackLayout.clear_gather(graphs, lay.off_map())
    lay.seg_index = seg_index
    lay.ends_off, lay.ends_const, lay.ends_sign = ends_off, ends_const, ends_sign
    return lay


def pack(graphs: Sequence[OptimizeGraph], layout: PackLayout, backend: str = "numpy"):
    xp = _backend(backend)
    x = xp.zeros((layout.n,))
    for gi, g in enumerate(graphs):
        pn, vn = layout.free_pos_idx[gi], layout.free_v_idx[gi]
        po, vo = layout.free_pos_offset[gi], layout.free_v_offset[gi]
        if pn.size:
            block = xp.asarray(g.pos[pn], dtype=x.dtype).reshape(-1)
            if backend in ("jax", "jnp"):
                x = x.at[po : po + block.shape[0]].set(block)  # type: ignore
            else:
                x[po : po + block.shape[0]] = block
        if vn.size:
            block = xp.asarray(g.v[vn], dtype=x.dtype).reshape(-1)
            if backend in ("jax", "jnp"):
                x = x.at[vo : vo + block.shape[0]].set(block)  # type: ignore
            else:
                x[vo : vo + block.shape[0]] = block
    return x


def unpack(x: Any, graphs: Sequence[OptimizeGraph], layout: PackLayout, backend: str = "numpy"):
    xp = _backend(backend)
    x = xp.asarray(x)
    for gi, g in enumerate(graphs):
        pn, vn = layout.free_pos_idx[gi], layout.free_v_idx[gi]
        po, vo = layout.free_pos_offset[gi], layout.free_v_offset[gi]
        if backend in ("jax", "jnp"):
            pos, vel = xp.asarray(g.pos), xp.asarray(g.v)
            if pn.size:
                pos = pos.at[pn].set(x[po : po + 3 * pn.size].reshape(-1, 3))  # type: ignore
            if vn.size:
                vel = vel.at[vn].set(x[vo : vo + 3 * vn.size].reshape(-1, 3))  # type: ignore
            g.pos, g.v = pos, vel  # type: ignore
        else:
            # Always replace with writable NumPy buffers. JIT ``unpack(..., jax)`` may
            # leave ``g.pos``/``g.v`` as immutable JAX arrays on the same objects.
            pos = np.array(g.pos, dtype=float, copy=True)
            vel = np.array(g.v, dtype=float, copy=True)
            if pn.size:
                pos[pn] = np.asarray(x[po : po + 3 * pn.size], dtype=float).reshape(-1, 3)
            if vn.size:
                vel[vn] = np.asarray(x[vo : vo + 3 * vn.size], dtype=float).reshape(-1, 3)
            g.pos, g.v = pos, vel
    return graphs


def refresh_from_x(
    x: Any,
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    ts_count: int = 10,
    backend: str = "jax",
):
    """Per-graph ``(ends, xyz, radius)`` from packed ``x``; does not write ``graphs``."""
    xp = _backend(backend)
    x = xp.asarray(x)
    out = []
    for gi, g in enumerate(graphs):
        pos, vel = xp.asarray(g.pos), xp.asarray(g.v)
        pn, vn = layout.free_pos_idx[gi], layout.free_v_idx[gi]
        po, vo = layout.free_pos_offset[gi], layout.free_v_offset[gi]
        if pn.size:
            pos = pos.at[pn].set(x[po : po + 3 * pn.size].reshape(-1, 3))  # type: ignore
        if vn.size:
            vel = vel.at[vn].set(x[vo : vo + 3 * vn.size].reshape(-1, 3))  # type: ignore
        out.append(g.refresh(pos, vel, ts_count, backend))
    return out
