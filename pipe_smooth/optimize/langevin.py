"""Discrete overdamped Langevin on packed Hermite DOF (§7.3)."""

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


def langevin(
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
    """Euler–Maruyama: x ← x - η ∇J + √(2ηT) ξ, with capped η and annealed T.

    η = min(η_max, δ / (||∇J||+ε)). ``anneal``: ``linear`` | ``exp`` (same as adam_annealed).
    Active sets only from ``refresh_clear`` / ``refresh_union`` (outside AD).
    Logs every ``log_every`` steps (plus first & last).
    """
    if anneal not in ("linear", "exp"):
        raise ValueError(f"anneal must be 'linear' or 'exp', got {anneal!r}")
    if not recorder.phase:
        recorder.phase = "langevin"

    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    clear_active = None
    union_hits = None
    rng = jax.random.PRNGKey(key)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy", store=True)
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

        eta = jnp.minimum(eta_max, delta / (jnp.linalg.norm(g) + eps))
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
                print(f"langevin {t}/{steps}  J={val_f:.6g}  η={float(eta):.3g}  T={float(T):.3g}")

        rng, sub = jax.random.split(rng)
        xi = jax.random.normal(sub, shape=x.shape, dtype=x.dtype)
        x = x - eta * g + jnp.sqrt(2.0 * eta * T) * xi

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy", store=True)


def langevin_ema(
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
    beta: float = 0.9,
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
    """Langevin with EMA-smoothed step (§7.3.1).

    η_raw = min(η_max, δ/(||∇J||+ε)),  η̄ = β η̄ + (1-β) η_raw,
    then x ← x - η̄ ∇J + √(2 η̄ T) ξ. Same T anneal / refresh as ``langevin``.
    Logs every ``log_every`` steps (plus first & last).
    """
    if anneal not in ("linear", "exp"):
        raise ValueError(f"anneal must be 'linear' or 'exp', got {anneal!r}")
    if not recorder.phase:
        recorder.phase = "langevin_ema"

    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    eta_bar = jnp.asarray(eta_max, dtype=x.dtype)
    clear_active = None
    union_hits = None
    rng = jax.random.PRNGKey(key)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy", store=True)
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

        eta_raw = jnp.minimum(eta_max, delta / (jnp.linalg.norm(g) + eps))
        eta_bar = beta * eta_bar + (1.0 - beta) * eta_raw
        if anneal == "linear":
            T = T0 * (steps - t) / max(steps - 1, 1)
        else:
            T = T0 * (rho ** (t - 1))
        if t == 1 or t == steps or t % log_every == 0:
            val_f = float(val)
            recorder.record(
                t, val_f, x, eta=float(eta_bar), T=float(T),
                **{k: float(v) for k, v in parts.items()},  # type: ignore[arg-type]
            )
            if verbose:
                print(
                    f"langevin_ema {t}/{steps}  J={val_f:.6g}  "
                    f"η̄={float(eta_bar):.3g}  T={float(T):.3g}"
                )

        rng, sub = jax.random.split(rng)
        xi = jax.random.normal(sub, shape=x.shape, dtype=x.dtype)
        x = x - eta_bar * g + jnp.sqrt(2.0 * eta_bar * T) * xi

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy", store=True)
