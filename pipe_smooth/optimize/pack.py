"""Pack free Hermite DOF into one vector x.

Free: node.pos if not fixed_p; derivative v = (vx, vy, vz) if not fixed_v.
||v|| is the tangent scale (old σ). Ports / clamped interfaces stay out of x
via HermiteNode.fixed_p / fixed_v.

Also holds static ends gather + clear/union hit caps for JIT-stable costs.
Weighted J / ∇J: see ``optimize.loss.loss_and_grad``.
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
    """Free DOF slots + ends gather table + fixed clear/union hit budgets.

    slots: (graph_index, node_id, field, offset); field is 'pos' (3) or 'dir' (3).
    ``ends_off[s, k] >= 0`` → ``x[off:off+3]``; else ``ends_const[s, k]``.
    Field order k: p0, v0, p1, v1 (same as ``hermite_ends``).
    """

    slots: list[tuple[int, str, str, int]]  # (graph_index, node_id, 'pos'|'dir', offset in x)
    n: int  # length of packed free-DOF vector x
    # --- ends gather (for JIT-stable clearance / union) ---
    seg_index: dict[tuple[str, str], int] = field(default_factory=dict)  # (group_id, seg_id) → row s in ends_*
    ends_off: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))  # (S,4) x-offsets; -1 = const
    ends_const: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3), dtype=np.float64))  # (S,4,3) when ends_off < 0
    # (S,4): scale on from-x ends only; pos=1, dir=u_sign/v_sign (const already signed)
    ends_sign: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    p_max_obs: int = 0  # max packed obstacle–seg hits
    p_max_grp: int = 0  # max packed group–group hits
    p_max_union: int = 0  # max packed union Chamfer samples (all graphs flattened)

    def export(self, path: str | Path) -> None:
        """Write DOF layout JSON for resume / viz (pairs with recorded ``x``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "slots": [[gi, nid, field, off] for gi, nid, field, off in self.slots],
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def clear_gather(
        graphs: Sequence[OptimizeGraph],
        slots: list[tuple[int, str, str, int]],
    ) -> tuple[dict[tuple[str, str], int], np.ndarray, np.ndarray, np.ndarray]:
        """Map each Hermite seg → (x-offset | const) for (p0,v0,p1,v1)."""
        off_map = {(gi, nid, fld): off for gi, nid, fld, off in slots}
        seg_index: dict[tuple[str, str], int] = {}
        rows_offset: list[list[int]] = []
        rows_const: list[list[list[float]]] = []
        rows_sign: list[list[float]] = []
        for gi, g in enumerate(graphs):
            for sid, (p0, v0, p1, v1) in g.hermite_ends(backend="numpy").items():
                seg = g.segments[sid]
                row_offset: list[int] = []
                row_const: list[list[float]] = []
                for nid, fld, val in (
                    (seg.u, "pos", p0),
                    (seg.u, "dir", v0),
                    (seg.v, "pos", p1),
                    (seg.v, "dir", v1),
                ):
                    key = (gi, nid, fld)
                    if key in off_map:
                        row_offset.append(int(off_map[key]))  # x-offset for p0,v0,p1,v1
                        row_const.append([0.0, 0.0, 0.0])  # default value for fixed/port
                    else:
                        row_offset.append(-1)  # -1 = const value
                        row_const.append(np.asarray(val, dtype=float).reshape(3).tolist())  # fixed/port value
                seg_index[(g.id, sid)] = len(rows_offset)  # (group_id, seg_id) → row s in ends_*
                rows_offset.append(row_offset)
                rows_const.append(row_const)
                rows_sign.append([1.0, float(seg.u_sign), 1.0, float(seg.v_sign)])  # type: ignore
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
        """``ends[S,4,3]`` from packed ``x`` (same semantics as ``hermite_ends``)."""
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
        """Load DOF slots; if ``graphs`` and all ``p_max_*`` set, also fill ends gather."""
        data = json.loads(Path(path).read_text())
        slots = [
            (int(gi), str(nid), str(fld), int(off))
            for gi, nid, fld, off in data["slots"]
        ]
        n = int(data["n"])
        if (
            graphs is not None
            and p_max_obs is not None
            and p_max_grp is not None
            and p_max_union is not None
        ):
            seg_index, ends_off, ends_const, ends_sign = cls.clear_gather(graphs, slots)
            return cls(
                slots=slots,
                n=n,
                seg_index=seg_index,
                ends_off=ends_off,
                ends_const=ends_const,
                ends_sign=ends_sign,
                p_max_obs=int(p_max_obs),
                p_max_grp=int(p_max_grp),
                p_max_union=int(p_max_union),
            )
        return cls(slots=slots, n=n)


def layout(
    graphs: Sequence[OptimizeGraph],
    p_max_obs: int = 0,
    p_max_grp: int = 0,
    p_max_union: int = 0,
) -> PackLayout:
    """Free DOF slots + ends gather; store clear/union hit caps."""
    slots: list[tuple[int, str, str, int]] = []
    n = 0
    for gi, g in enumerate(graphs):
        for nid, node in g.nodes.items():
            if not node.fixed_p:
                slots.append((gi, nid, "pos", n))
                n += 3  # free position (x, y, z)
            if not node.fixed_v and node.v is not None:
                slots.append((gi, nid, "dir", n))
                n += 3  # free derivative (vx, vy, vz); ||v|| is the tangent scale

    seg_index, ends_off, ends_const, ends_sign = PackLayout.clear_gather(graphs, slots)
    return PackLayout(
        slots=slots,
        n=n,
        seg_index=seg_index,
        ends_off=ends_off,
        ends_const=ends_const,
        ends_sign=ends_sign,
        p_max_obs=int(p_max_obs),
        p_max_grp=int(p_max_grp),
        p_max_union=int(p_max_union),
    )


def pack(graphs: Sequence[OptimizeGraph], layout: PackLayout, backend: str = "numpy"):
    """Current free DOF → vector x."""
    xp = _backend(backend)
    x = xp.zeros((layout.n,))
    for gi, nid, field, off in layout.slots:
        node = graphs[gi].nodes[nid]
        block = xp.asarray(node.pos if field == "pos" else node.v, dtype=x.dtype)
        if backend in ("jax", "jnp"):
            x = x.at[off : off + 3].set(block)  # type: ignore
        else:
            x[off : off + 3] = block
    return x


def unpack(x: Any, graphs: Sequence[OptimizeGraph], layout: PackLayout, backend: str = "numpy"):
    """Write x into node.pos / node.v. v is the end derivative γ', not a unit tangent."""
    xp = _backend(backend)
    x = xp.asarray(x)
    for gi, nid, field, off in layout.slots:
        node = graphs[gi].nodes[nid]
        if field == "pos":
            node.pos = x[off : off + 3]
        else:
            node.v = x[off : off + 3]
    return graphs
