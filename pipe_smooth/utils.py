"""Shared array helpers for numpy | jax backends."""

from __future__ import annotations

from typing import Any

import numpy as np


def backend(name: str = "numpy"):
    """Return array module: 'numpy'/'np' -> numpy, 'jax'/'jnp' -> jax.numpy."""
    if name in ("numpy", "np"):
        return np
    if name in ("jax", "jnp"):
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"backend must be 'numpy' or 'jax', got {name!r}")


def to_float_list(x: Any) -> list[float]:
    """Flatten array-like to a JSON-friendly list of floats."""
    return [float(v) for v in np.asarray(x, dtype=float).reshape(-1)]
