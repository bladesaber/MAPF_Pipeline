"""Plain Adam on packed Hermite DOF (soft Stage-1 loop)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights, loss_and_grad
from pipe_smooth.optimize.pack import PackLayout, pack, unpack
from pipe_smooth.optimize.recorder import Recorder


def adam(
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    weights: LossWeights,
    *,
    steps: int = 200,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    clear_kind: str = "capsule",
    refresh_clear: Callable[[Sequence[OptimizeGraph]], Any] | None = None,
    refresh_union: Callable[[Sequence[OptimizeGraph]], dict] | None = None,
    refresh_every: int = 10,
    log_every: int = 5,
    recorder: Recorder,
    verbose: bool = False,
) -> None:
    """Minimize weighted soft J with Adam.

    Active sets come only from refresh callbacks (outside AD), every ``refresh_every``
    steps (including step 1):

    - ``refresh_clear(graphs) -> clear_active``
    - ``refresh_union(graphs) -> {graph_id: hits}``

    ``recorder`` logs J / parts every ``log_every`` steps (plus first & last).
    """
    if not recorder.phase:
        recorder.phase = "adam"

    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    clear_active = None
    union_hits = None

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy")
            if refresh_clear is not None:
                clear_active = refresh_clear(graphs)
            if refresh_union is not None:
                union_hits = refresh_union(graphs)

        val, g, parts = loss_and_grad(
            x,
            graphs,
            layout,
            weights,
            clear_active=clear_active,
            clear_kind=clear_kind,  # type: ignore[arg-type]
            union_hits=union_hits,
        )
        if t == 1 or t == steps or t % log_every == 0:
            val_f = float(val)
            recorder.record(t, val_f, x, **{k: float(v) for k, v in parts.items()})  # type: ignore[arg-type]
            if verbose:
                print(f"adam {t}/{steps}  J={val_f:.6g}")

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        x = x - lr * mhat / (jnp.sqrt(vhat) + eps)

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy")


def adam_annealed(
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    weights: LossWeights,
    *,
    steps: int = 200,
    eta_max: float = 1e-3,
    delta: float = 1e-2,
    T0: float = 1e-3,
    anneal: str = "linear",
    rho: float = 0.95,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    clear_kind: str = "capsule",
    refresh_clear: Callable[[Sequence[OptimizeGraph]], Any] | None = None,
    refresh_union: Callable[[Sequence[OptimizeGraph]], dict] | None = None,
    refresh_every: int = 10,
    log_every: int = 5,
    key: int = 0,
    recorder: Recorder,
    verbose: bool = False,
) -> None:
    """Adam + annealed noise (§7.3.2): x ← x - η û + √(2ηT) ξ, T↓0.

    η = min(η_max, δ / (||û||+ε)). ``anneal``: ``linear`` → T=0 at last step;
    ``exp`` → T = T0 * ρ^{t-1}. Logs every ``log_every`` steps (plus first & last).
    """
    if anneal not in ("linear", "exp"):
        raise ValueError(f"anneal must be 'linear' or 'exp', got {anneal!r}")
    if not recorder.phase:
        recorder.phase = "adam_annealed"

    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    clear_active = None
    union_hits = None
    rng = jax.random.PRNGKey(key)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy")
            if refresh_clear is not None:
                clear_active = refresh_clear(graphs)
            if refresh_union is not None:
                union_hits = refresh_union(graphs)

        val, g, parts = loss_and_grad(
            x,
            graphs,
            layout,
            weights,
            clear_active=clear_active,
            clear_kind=clear_kind,  # type: ignore[arg-type]
            union_hits=union_hits,
        )

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        uhat = mhat / (jnp.sqrt(vhat) + eps)

        eta = jnp.minimum(eta_max, delta / (jnp.linalg.norm(uhat) + eps))
        if anneal == "linear":
            T = T0 * (steps - t) / max(steps - 1, 1)
        else:
            T = T0 * (rho ** (t - 1))
        if t == 1 or t == steps or t % log_every == 0:
            val_f = float(val)
            recorder.record(
                t, val_f, x, eta=float(eta), T=float(T),
                **{k: float(v) for k, v in parts.items()},  # type: ignore[arg-type]
            )
            if verbose:
                print(f"adam_annealed {t}/{steps}  J={val_f:.6g}  η={float(eta):.3g}  T={float(T):.3g}")

        rng, sub = jax.random.split(rng)
        xi = jax.random.normal(sub, shape=x.shape, dtype=x.dtype)
        x = x - eta * uhat + jnp.sqrt(2.0 * eta * T) * xi

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy")
