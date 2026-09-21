"""§7.1 — sample spacing regularizer J_space.

Soft AD: penalize nonuniform Δs on Hermite samples (anti parameter-evasion).
SQP: optional relative band residuals about mean Δs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import sample_hermite
from pipe_smooth.costs.path import softplus


def space_cost(xyz, eta: float = 1.0, ds_max=None, soft_sigma: float = 1e-2, backend: str = "numpy"):
    """η Σ (Δs_i - mean(Δs))²; optional η Σ [Δs_i - Δs_max]_+^σ ²."""
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    mean = xp.mean(ds)
    j = xp.sum((ds - mean) ** 2)
    if ds_max is not None:
        j = j + xp.sum(softplus(ds - ds_max, sigma=soft_sigma, backend=backend) ** 2)
    return eta * j


def space_band_residuals(xyz, eps: float = 0.2, upper_only: bool = False, backend: str = "numpy"):
    """SQP-style residuals ≤ 0: ds ≤ (1+ε)mean; optionally ds ≥ (1-ε)mean.

    Returns flat vector of g(X) with g≤0 feasible (§7.1 (2)). Linearize at iterate in SQP.
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    mean = xp.mean(ds)
    hi = ds - (1.0 + eps) * mean
    if upper_only:
        return hi
    lo = (1.0 - eps) * mean - ds
    return xp.concatenate([hi, lo], axis=0)


def graph_space(
    graph: OptimizeGraph,
    eta: float = 1.0,
    ds_max=None,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Σ space_cost over Hermite segments; each seg_id once (usage walk for port clamps)."""
    xp = _backend(backend)
    total = xp.asarray(0.0)
    seen: set[str] = set()
    for path in graph.usage_paths:
        nseg = len(path.segments)
        for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
            if sid in seen:
                continue
            seen.add(sid)
            seg = graph.segments[sid]
            nu, nv = graph.nodes[seg.u], graph.nodes[seg.v]
            if rev:
                p0, v0 = nv.pos, -nv.v  # type: ignore[operator]
                p1, v1 = nu.pos, -nu.v  # type: ignore[operator]
            else:
                p0, v0 = nu.pos, nu.v
                p1, v1 = nv.pos, nv.v
            if i == 0:
                v0 = path.v_start
            if i == nseg - 1:
                v1 = path.v_end
            n = max(2, int(seg.ts_count))
            xyz = sample_hermite(p0, v0, p1, v1, n=n, backend=backend)
            total = total + space_cost(
                xyz, eta=eta, ds_max=ds_max, soft_sigma=soft_sigma, backend=backend
            )
    return total
