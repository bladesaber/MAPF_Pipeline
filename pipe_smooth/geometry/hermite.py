"""§6.5.1 — cubic Hermite: eval, length, Bézier hull.

Callers pass NumPy or JAX arrays. Choose ops via backend='numpy'|'jax' (default numpy).
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend

# 4-point Gauss–Legendre on [0, 1]
_GL4_T = (0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262)
_GL4_W = (0.17392742256872692, 0.3260725774312731, 0.3260725774312731, 0.17392742256872692)


def unit(dx, backend: str = "numpy"):
    """Unit vector. `dx` is (3,) array already on that backend."""
    xp = _backend(backend)
    return dx / (xp.linalg.norm(dx) + 1e-12)


def eval_hermite(p0, v0, p1, v1, t, backend: str = "numpy"):
    """γ(t), t in [0, 1]. p and v are (3,); v is the end derivative γ' (not unit)."""
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def eval_hermite_d1(p0, v0, p1, v1, t, backend: str = "numpy"):
    """dγ/dt. Same conventions as eval_hermite."""
    t2 = t * t
    dh00 = 6 * t2 - 6 * t
    dh10 = 3 * t2 - 4 * t + 1
    dh01 = -6 * t2 + 6 * t
    dh11 = 3 * t2 - 2 * t
    return dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1


def eval_hermite_d2(p0, v0, p1, v1, t, backend: str = "numpy"):
    """d²γ/dt². Same conventions as eval_hermite."""
    d2h00 = 12 * t - 6
    d2h10 = 6 * t - 4
    d2h01 = -12 * t + 6
    d2h11 = 6 * t - 2
    return d2h00 * p0 + d2h10 * v0 + d2h01 * p1 + d2h11 * v1


def eval_hermite_kappa(p0, v0, p1, v1, t, backend: str = "numpy"):
    """Curvature κ(t) = ||γ' × γ''|| / ||γ'||³ (§6 / parameterized curve)."""
    xp = _backend(backend)
    d1 = eval_hermite_d1(p0, v0, p1, v1, t, backend=backend)
    d2 = eval_hermite_d2(p0, v0, p1, v1, t, backend=backend)
    return xp.linalg.norm(xp.cross(d1, d2)) / (xp.linalg.norm(d1) ** 3 + 1e-12)


def eval_hermite_ts(p0, v0, p1, v1, ts, backend: str = "numpy"):
    """γ(ts) for many parameters. `ts` shape (n,) → points (n, 3). Vectorized."""
    xp = _backend(backend)
    ts = xp.asarray(ts)
    t2, t3 = ts * ts, ts * ts * ts
    h00 = (2 * t3 - 3 * t2 + 1)[:, None]
    h10 = (t3 - 2 * t2 + ts)[:, None]
    h01 = (-2 * t3 + 3 * t2)[:, None]
    h11 = (t3 - t2)[:, None]
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def sample_hermite(p0, v0, p1, v1, n: int = 16, backend: str = "numpy"):
    """(n, 3) uniform samples on [0, 1], inclusive endpoints."""
    xp = _backend(backend)
    n = max(int(n), 2)
    return eval_hermite_ts(p0, v0, p1, v1, xp.linspace(0.0, 1.0, n), backend=backend)


def hermite_length(p0, v0, p1, v1, backend: str = "numpy"):
    """Arc length ∫₀¹ ||γ'(t)|| dt (4-point Gauss–Legendre)."""
    xp = _backend(backend)
    acc = 0.0
    for ti, wi in zip(_GL4_T, _GL4_W):
        d1 = eval_hermite_d1(p0, v0, p1, v1, ti, backend=backend)
        acc = acc + wi * xp.linalg.norm(d1)
    return acc


def hermite_bezier_controls(p0, v0, p1, v1, backend: str = "numpy"):
    """Hermite → Bézier controls C0..C3, shape (4, 3). Curve ⊂ CH{Ci}."""
    xp = _backend(backend)
    c0 = p0
    c1 = p0 + v0 / 3.0
    c2 = p1 - v1 / 3.0
    c3 = p1
    return xp.stack([c0, c1, c2, c3], axis=0)


def hermite_convex_hull(p0, v0, p1, v1, backend: str = "numpy"):
    """Outer convex bound of one segment = four Bézier controls (§7.3)."""
    return hermite_bezier_controls(p0, v0, p1, v1, backend=backend)


def bezier_on_interval(C, ta, tb, backend: str = "numpy"):
    """Restrict cubic Bézier controls C (4, 3) on [0, 1] to sub-interval [ta, tb].

    de Casteljau: split at tb → left = [0, tb]; then split left at ta/tb → right = [ta, tb].
    Same cubic as re-Hermite with endpoint pos + derivatives scaled by (tb - ta).
    """
    xp = _backend(backend)

    def split(P, t):
        p01 = (1 - t) * P[0] + t * P[1]
        p12 = (1 - t) * P[1] + t * P[2]
        p23 = (1 - t) * P[2] + t * P[3]
        p012 = (1 - t) * p01 + t * p12
        p123 = (1 - t) * p12 + t * p23
        p0123 = (1 - t) * p012 + t * p123
        left = xp.stack([P[0], p01, p012, p0123], axis=0)
        right = xp.stack([p0123, p123, p23, P[3]], axis=0)
        return left, right

    C = xp.asarray(C).reshape(4, 3)
    ta, tb = float(ta), float(tb)
    left, _ = split(C, tb) if tb < 1.0 - 1e-15 else (C, None)
    if ta <= 1e-15:
        return left
    _, mid = split(left, ta / tb)
    return mid
