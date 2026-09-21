"""Pack free Hermite DOF into one vector and differentiate a loss through it.

Free: node.pos if not fixed_p; derivative v = (vx, vy, vz) if not fixed_v.
||v|| is the tangent scale (old σ). Ports / clamped interfaces stay out of x
via HermiteNode.fixed_p / fixed_v.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.utils import backend as _backend


@dataclass
class PackLayout:
    """slots: (graph_index, node_id, field, offset). field is 'pos' (3) or 'dir' (3)."""

    slots: list[tuple[int, str, str, int]]
    n: int


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


def loss_grad(
    loss_fn: Callable[[Sequence[OptimizeGraph]], Any],
    x: Any,
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
):
    """(value, ∇_x loss). loss_fn(graphs) must read node fields with backend='jax'.

    Active-set / collision detection stays outside this call: freeze pairs first,
    then let loss_fn close over them.
    """
    import jax

    def f(xx):
        unpack(xx, graphs, layout, backend="jax")
        return loss_fn(graphs)

    return jax.value_and_grad(f)(x)
