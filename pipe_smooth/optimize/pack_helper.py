"""Strategies for PackLayout.p_max_* vs active-set size.

adaptive: grow/shrink ``p_max`` with hysteresis (always ``p_max >= n_need`` after fit).
silence:  keep ``p_max`` fixed; overflow is truncated by ``pack_*`` (first slots).
``p_max`` is never below a per-channel ``p_min`` so JIT shapes stay non-empty.
"""

from __future__ import annotations

import math
from typing import Literal

Strategy = Literal["adaptive", "silence"]


def apply_budget(
    n_need: int,
    p_max: int,
    *,
    strategy: Strategy = "silence",
    ratio_expand: float = 1.2,
    ratio_shrink: float = 0.5,
    p_min: int = 32,
) -> int:
    """Fit one budget; return ``p_max_new`` (``>= p_min``)."""
    if ratio_expand <= 1.0:
        raise ValueError(f"ratio_expand must be > 1, got {ratio_expand}")
    if not (0.0 < ratio_shrink < 1.0):
        raise ValueError(f"ratio_shrink must be in (0, 1), got {ratio_shrink}")
    p_min = max(1, int(p_min))

    n = max(0, int(n_need))
    p = max(p_min, int(p_max))

    if strategy == "adaptive":
        if n > p:
            p = int(math.ceil(n * ratio_expand))
        elif n < p * ratio_shrink:
            p = int(max(n * ratio_expand, p * ratio_shrink))
        p = max(p_min, p)
    elif strategy != "silence":
        raise ValueError(f"unknown strategy {strategy!r}")
    return p


def sync_layout_budgets(
    layout,
    *,
    n_obs: int = 0,
    n_grp: int = 0,
    n_union: int = 0,
    strategy: Strategy = "silence",
    ratio_expand: float = 1.2,
    ratio_shrink: float = 0.5,
    p_min_obs: int = 128,
    p_min_grp: int = 128,
    p_min_union: int = 64,
) -> None:
    """Update ``layout.p_max_obs/grp/union`` in-place."""
    base = dict(strategy=strategy, ratio_expand=ratio_expand, ratio_shrink=ratio_shrink)
    layout.p_max_obs = apply_budget(n_obs, layout.p_max_obs, p_min=p_min_obs, **base)  # type: ignore
    layout.p_max_grp = apply_budget(n_grp, layout.p_max_grp, p_min=p_min_grp, **base)  # type: ignore
    layout.p_max_union = apply_budget(n_union, layout.p_max_union, p_min=p_min_union, **base)  # type: ignore
