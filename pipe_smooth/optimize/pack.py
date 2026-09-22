"""Pack free Hermite DOF into one vector x.

Free: node.pos if not fixed_p; derivative v = (vx, vy, vz) if not fixed_v.
||v|| is the tangent scale (old σ). Ports / clamped interfaces stay out of x
via HermiteNode.fixed_p / fixed_v.

Weighted J / ∇J: see ``optimize.loss.loss_and_grad``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.utils import backend as _backend


@dataclass
class PackLayout:
    """slots: (graph_index, node_id, field, offset). field is 'pos' (3) or 'dir' (3)."""

    slots: list[tuple[int, str, str, int]]
    n: int

    def export(self, path: str | Path) -> None:
        """Write layout JSON for resume / viz (pairs with recorded ``x``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "slots": [[gi, nid, field, off] for gi, nid, field, off in self.slots],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PackLayout":
        data = json.loads(Path(path).read_text())
        slots = [
            (int(gi), str(nid), str(field), int(off))
            for gi, nid, field, off in data["slots"]
        ]
        return cls(slots=slots, n=int(data["n"]))


def layout(graphs: Sequence[OptimizeGraph]) -> PackLayout:
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
    return PackLayout(slots, n)


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
