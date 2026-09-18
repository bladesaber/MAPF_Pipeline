"""§6.4.1 — path costs L, K, J_path on samples.

Inputs align with OptimizeGraph.sample_usage_path: xyz (N,3), radius (N,), optional w (N,).
Optional unit end tangents t_start / t_end (port clamps). backend='numpy'|'jax'.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend

_EPS = 1e-12


def softplus(z, sigma: float = 1e-2, backend: str = "numpy"):
    """Smooth hinge [z]_+^σ = σ log(1 + exp(z/σ))."""
    xp = _backend(backend)
    z = xp.asarray(z)
    return sigma * xp.log1p(xp.exp(z / sigma))


def path_length(xyz, w=None, backend: str = "numpy"):
    """L_u = Σ w̄_i Δs_i,  w̄_i = (w_i + w_{i+1})/2."""
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    if w is None:
        return xp.sum(ds)
    w = xp.asarray(w)
    return xp.sum(0.5 * (w[:-1] + w[1:]) * ds)


def path_curvature(
    xyz,
    radius,
    w=None,
    t_start=None,
    t_end=None,
    rho: float = 0.0,
    c_pref: float = 0.25,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Radius-aware K_u = K^bend + ρ K^thr on polyline samples (§6.4.1).

    Bend (MVP): Σ ||Δt||² r² / Δℓ with edge tangents from xyz. Optional unit
    `t_start` / `t_end` (port / usage clamps) add end turns vs the first / last
    chord; σ is not used. `radius` is (N,) like `xyz` (scalar broadcasts).
    Threshold term (ρ>0) stays on interior Menger samples only.
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)
    if radius.ndim == 0:
        radius = xp.full((xyz.shape[0],), radius)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    t = (xyz[1:] - xyz[:-1]) / (ds[:, None] + _EPS)
    dl = 0.5 * (ds[:-1] + ds[1:])
    wi = xp.ones((xyz.shape[0],), dtype=xyz.dtype) if w is None else xp.asarray(w)
    r_i = radius[1:-1]
    w_i = wi[1:-1]

    # MVP: ||t_i - t_{i-1}||² = 4 sin²(θ/2) ≈ θ² for small bends
    dt2 = xp.sum((t[1:] - t[:-1]) ** 2, axis=-1)
    K_bend = xp.sum(w_i * dt2 * (r_i ** 2) / (dl + _EPS))

    if t_start is not None:
        ts = xp.asarray(t_start)
        ts = ts / (xp.linalg.norm(ts) + _EPS)
        K_bend = K_bend + wi[0] * xp.sum((t[0] - ts) ** 2) * (radius[0] ** 2) / (ds[0] + _EPS)
    if t_end is not None:
        te = xp.asarray(t_end)
        te = te / (xp.linalg.norm(te) + _EPS)
        K_bend = K_bend + wi[-1] * xp.sum((te - t[-1]) ** 2) * (radius[-1] ** 2) / (
            ds[-1] + _EPS
        )

    if rho == 0.0:
        return K_bend

    # Menger κ at interior samples
    a = xyz[1:-1] - xyz[:-2]
    b = xyz[2:] - xyz[1:-1]
    c = xyz[2:] - xyz[:-2]
    cross = xp.linalg.norm(xp.cross(a, b), axis=-1)
    denom = (
        xp.linalg.norm(a, axis=-1) * xp.linalg.norm(b, axis=-1) * xp.linalg.norm(c, axis=-1)
        + _EPS
    )
    kappa_m = cross / denom
    thr = softplus(kappa_m * r_i - c_pref, sigma=soft_sigma, backend=backend)
    K_thr = xp.sum(w_i * (thr ** 2) * dl)
    return K_bend + rho * K_thr


def j_path(
    xyz,
    radius,
    w=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    backend: str = "numpy",
    **kwargs,
):
    """α L_u + β K_u for one usage sample path.

    `xyz` (N,3), `radius` (N,); optional `w`, `t_start`, `t_end` via kwargs.
    """
    L = path_length(xyz, w=w, backend=backend)
    K = path_curvature(xyz, radius, w=w, backend=backend, **kwargs)
    return alpha * L + beta * K


def j_path_sum(samples, alpha: float = 1.0, beta: float = 1.0, backend: str = "numpy", **kwargs):
    """Σ_u (α L_u + β K_u); sample dicts: xyz, radius, optional w / t_start / t_end."""
    xp = _backend(backend)
    total = 0.0
    for s in samples:
        kw = dict(kwargs)
        total = total + j_path(
            s["xyz"],
            s["radius"],
            w=s.get("w"),
            alpha=alpha,
            beta=beta,
            backend=backend,
            t_start=s.get("t_start", kw.pop("t_start", None)),
            t_end=s.get("t_end", kw.pop("t_end", None)),
            **kw,
        )
    return xp.asarray(total)
