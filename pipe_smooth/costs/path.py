"""§6.4.1 — path chord energy and curvature pieces.

Polyline samples: `path_length` (Σ Δs²), `k_bend` / `k_bend_approx` / `k_menger` /
`k_fd` (bend **energy**). Pointwise Menger κ for reports: `kappa_menger`.
Cubic Hermite segment: `k_hermite` (analytic κ(t)). Caller sums / weights costs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import eval_hermite, eval_hermite_d1, eval_hermite_d2

_EPS = 1e-12


def softplus(z, sigma: float = 1e-2, backend: str = "numpy"):
    """Smooth hinge [z]_+^σ = σ log(1 + exp(z/σ))."""
    xp = _backend(backend)
    z = xp.asarray(z)
    return sigma * xp.log1p(xp.exp(z / sigma))


def path_length(xyz, w=None, backend: str = "numpy"):
    """L_u = Σ w̄_i Δs_i² (chord energy); w̄_i = (w_i + w_{i+1})/2.

    Squared chords (no sqrt): same [length²] as ``space_cost``, stronger pull on
    long gaps; α/η stay scale-comparable under geometric rescaling.

    ``xyz`` shape ``(T, 3)`` → scalar; ``(S, T, 3)`` → sum over all segments
    (or pass ``w`` with matching leading dims for weighted chords).
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    d2 = xp.sum((xyz[..., 1:, :] - xyz[..., :-1, :]) ** 2, axis=-1)
    if w is None:
        return xp.sum(d2)
    w = xp.asarray(w)
    return xp.sum(0.5 * (w[..., :-1] + w[..., 1:]) * d2)


def _check_batch(xyz, radius, w):
    assert xyz.ndim == 3 and xyz.shape[-1] == 3, f"xyz need (B,T,3), got {getattr(xyz, 'shape', None)}"
    assert radius.ndim == 2 and radius.shape == xyz.shape[:2], (
        f"radius need (B,T)={xyz.shape[:2]}, got {getattr(radius, 'shape', None)}"
    )
    if w is not None:
        assert w.ndim == 2 and w.shape == xyz.shape[:2], (
            f"w need (B,T)={xyz.shape[:2]}, got {getattr(w, 'shape', None)}"
        )


def _check_ports(t_start, t_end, B):
    if t_start is not None:
        assert t_start.ndim == 2 and t_start.shape == (B, 3), (
            f"t_start need (B,3)=({B},3), got {getattr(t_start, 'shape', None)}"
        )
    if t_end is not None:
        assert t_end.ndim == 2 and t_end.shape == (B, 3), (
            f"t_end need (B,3)=({B},3), got {getattr(t_end, 'shape', None)}"
        )


def k_bend(
    xyz,
    radius,
    w=None,
    t_start=None,
    t_end=None,
    backend: str = "numpy",
):
    """Turning-energy bend: Σ (θ r)² / Δℓ with θ=atan2(||t×t||, t·t) (§6.4.1).

    Edge tangents from xyz chords. Optional unit `t_start` / `t_end` (port clamps)
    add end turns vs first / last chord. `radius` (N,) or scalar.
    """
    xp = _backend(backend)
    xyz, radius = xp.asarray(xyz), xp.asarray(radius)
    _check_batch(xyz, radius, w)
    B = int(xyz.shape[0])
    wi = xp.ones(xyz.shape[:2], dtype=xyz.dtype) if w is None else xp.asarray(w)
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1) # (B, T-1)
    t = (xyz[:, 1:] - xyz[:, :-1]) / (ds[..., None] + _EPS) # (B, T-1, 3)
    dl = 0.5 * (ds[:, :-1] + ds[:, 1:]) # (B, T-2)
    
    # θ ∈ [0,π]; avoid arccos (singular as cos→1).
    # Smooth ||cross|| so JAX grad is finite when chords are parallel (θ→0).
    c = xp.cross(t[:, :-1], t[:, 1:])
    cn = xp.sqrt(xp.sum(c * c, axis=-1) + _EPS)  # (B, T-2)
    theta = xp.arctan2(cn, xp.sum(t[:, :-1] * t[:, 1:], axis=-1))  # (B, T-2)
    K = xp.sum(wi[:, 1:-1] * (theta ** 2) * (radius[:, 1:-1] ** 2) / (dl + _EPS))

    if t_start is not None:
        ts = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        c0 = xp.cross(ts, t[:, 0])
        th0 = xp.arctan2(
            xp.sqrt(xp.sum(c0 * c0, axis=-1) + _EPS), xp.sum(ts * t[:, 0], axis=-1)
        )
        K = K + xp.sum(wi[:, 0] * (th0 ** 2) * (radius[:, 0] ** 2) / (ds[:, 0] + _EPS))
    if t_end is not None:
        te = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        c1 = xp.cross(t[:, -1], te)
        th1 = xp.arctan2(
            xp.sqrt(xp.sum(c1 * c1, axis=-1) + _EPS), xp.sum(t[:, -1] * te, axis=-1)
        )
        K = K + xp.sum(wi[:, -1] * (th1 ** 2) * (radius[:, -1] ** 2) / (ds[:, -1] + _EPS))
    return K


def k_bend_approx(
    xyz,
    radius,
    w=None,
    t_start=None,
    t_end=None,
    backend: str = "numpy",
):
    """Small-angle bend approx: Σ ||Δt||² r² / Δℓ (= 4 sin²(θ/2) r² / Δℓ).

    Same inputs as `k_bend`. Underestimates large turns; use `k_bend` (atan2 θ²) when
    sharp corners must be fully penalized.
    """
    xp = _backend(backend)
    xyz, radius = xp.asarray(xyz), xp.asarray(radius)
    _check_batch(xyz, radius, w)
    B = int(xyz.shape[0])
    wi = xp.ones(xyz.shape[:2], dtype=xyz.dtype) if w is None else xp.asarray(w)
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    t = (xyz[:, 1:] - xyz[:, :-1]) / (ds[..., None] + _EPS)
    dl = 0.5 * (ds[:, :-1] + ds[:, 1:])
    dt2 = xp.sum((t[:, 1:] - t[:, :-1]) ** 2, axis=-1)
    K = xp.sum(wi[:, 1:-1] * dt2 * (radius[:, 1:-1] ** 2) / (dl + _EPS))

    if t_start is not None:
        ts = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        K = K + xp.sum(
            wi[:, 0] * xp.sum((t[:, 0] - ts) ** 2, axis=-1) * (radius[:, 0] ** 2) / (ds[:, 0] + _EPS)
        )
    if t_end is not None:
        te = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        K = K + xp.sum(
            wi[:, -1] * xp.sum((te - t[:, -1]) ** 2, axis=-1) * (radius[:, -1] ** 2) / (ds[:, -1] + _EPS)
        )
    return K


def k_menger(
    xyz,
    radius,
    w=None,
    t_start=None,
    t_end=None,
    backend: str = "numpy",
):
    """Menger bend: Σ w (κ r)² Δℓ (no threshold).

    Interior: classic 3-point κ^M. Optional `t_start` / `t_end`: terminal κ ≈ θ/Δs
    with θ between the port tangent and the first / last chord (simpler than virtual
    Menger points; interior stays pure Menger).
    """
    xp = _backend(backend)
    xyz, radius = xp.asarray(xyz), xp.asarray(radius)
    _check_batch(xyz, radius, w)
    B = int(xyz.shape[0])
    wi = xp.ones(xyz.shape[:2], dtype=xyz.dtype) if w is None else xp.asarray(w)
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    dl = 0.5 * (ds[:, :-1] + ds[:, 1:])
    a = xyz[:, 1:-1] - xyz[:, :-2]
    b = xyz[:, 2:] - xyz[:, 1:-1]
    c = xyz[:, 2:] - xyz[:, :-2]
    ab = xp.cross(a, b)
    cross = xp.sqrt(xp.sum(ab * ab, axis=-1) + _EPS)
    denom = (
        xp.linalg.norm(a, axis=-1) * xp.linalg.norm(b, axis=-1) * xp.linalg.norm(c, axis=-1)
        + _EPS
    )
    kappa_m = 2.0 * cross / denom
    K = xp.sum(wi[:, 1:-1] * (kappa_m * radius[:, 1:-1]) ** 2 * dl)

    if t_start is not None:
        ts = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        t0 = (xyz[:, 1] - xyz[:, 0]) / (ds[:, 0, None] + _EPS)
        c0 = xp.cross(ts, t0)
        th0 = xp.arctan2(
            xp.sqrt(xp.sum(c0 * c0, axis=-1) + _EPS), xp.sum(ts * t0, axis=-1)
        )
        km0 = th0 / (ds[:, 0] + _EPS)
        K = K + xp.sum(wi[:, 0] * (km0 * radius[:, 0]) ** 2 * ds[:, 0])
    if t_end is not None:
        te = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        t1 = (xyz[:, -1] - xyz[:, -2]) / (ds[:, -1, None] + _EPS)
        c1 = xp.cross(t1, te)
        th1 = xp.arctan2(
            xp.sqrt(xp.sum(c1 * c1, axis=-1) + _EPS), xp.sum(t1 * te, axis=-1)
        )
        km1 = th1 / (ds[:, -1] + _EPS)
        K = K + xp.sum(wi[:, -1] * (km1 * radius[:, -1]) ** 2 * ds[:, -1])
    return K


def kappa_menger(xyz, t_start=None, t_end=None, backend: str = "numpy"):
    """Pointwise Menger curvature κ on a polyline (report / viz, not bend energy).

    Interior: κ_i = 2 ||(p_i-p_{i-1})×(p_{i+1}-p_i)|| / (|a||b||c|).
    Optional ``t_start`` / ``t_end``: end κ ≈ θ/Δs vs first / last chord.
    Returns shape (N,); undefined ends stay 0 when ports are omitted.
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    assert xyz.ndim == 3 and xyz.shape[-1] == 3, f"xyz need (B,T,3), got {xyz.shape}"
    B, n = int(xyz.shape[0]), int(xyz.shape[1])
    assert n >= 3, f"kappa_menger needs T>=3, got {n}"
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    a = xyz[:, 1:-1] - xyz[:, :-2]
    b = xyz[:, 2:] - xyz[:, 1:-1]
    c = xyz[:, 2:] - xyz[:, :-2]
    ab = xp.cross(a, b)
    cross = xp.sqrt(xp.sum(ab * ab, axis=-1) + _EPS)
    denom = (
        xp.linalg.norm(a, axis=-1)
        * xp.linalg.norm(b, axis=-1)
        * xp.linalg.norm(c, axis=-1)
        + _EPS
    )
    z0 = xp.zeros((B, 1), dtype=xyz.dtype)
    kappa = xp.concatenate([z0, 2.0 * cross / denom, z0], axis=1)
    if t_start is not None:
        ts = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        t0 = (xyz[:, 1] - xyz[:, 0]) / (ds[:, 0, None] + _EPS)
        c0 = xp.cross(ts, t0)
        th0 = xp.arctan2(xp.sqrt(xp.sum(c0 * c0, axis=-1) + _EPS), xp.sum(ts * t0, axis=-1))
        kappa = xp.concatenate([(th0 / (ds[:, 0] + _EPS))[:, None], kappa[:, 1:]], axis=1)
    if t_end is not None:
        te = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        t1 = (xyz[:, -1] - xyz[:, -2]) / (ds[:, -1, None] + _EPS)
        c1 = xp.cross(t1, te)
        th1 = xp.arctan2(xp.sqrt(xp.sum(c1 * c1, axis=-1) + _EPS), xp.sum(t1 * te, axis=-1))
        kappa = xp.concatenate([kappa[:, :-1], (th1 / (ds[:, -1] + _EPS))[:, None]], axis=1)
    return kappa


def k_fd(
    xyz,
    radius,
    w=None,
    t_start=None,
    t_end=None,
    backend: str = "numpy",
):
    """Finite-difference κ on polyline samples: Σ w (κ r)² Δℓ, κ=||γ'×γ''||/||γ'||³.

    Interior: centered γ', γ'' from chords. Optional `t_start` / `t_end`: one-sided
    at terminals — γ'=t_port, γ''=(t_chord - t_port)/Δs (no central stencil at ends).
    For cubic-Hermite κ(t) use `k_hermite`.
    """
    xp = _backend(backend)
    xyz, radius = xp.asarray(xyz), xp.asarray(radius)
    _check_batch(xyz, radius, w)
    B = int(xyz.shape[0])
    wi = xp.ones(xyz.shape[:2], dtype=xyz.dtype) if w is None else xp.asarray(w)
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    t = (xyz[:, 1:] - xyz[:, :-1]) / (ds[..., None] + _EPS)
    dl = 0.5 * (ds[:, :-1] + ds[:, 1:])
    d1 = (xyz[:, 2:] - xyz[:, :-2]) / ((ds[:, :-1] + ds[:, 1:])[..., None] + _EPS)
    d2 = (t[:, 1:] - t[:, :-1]) / (dl[..., None] + _EPS)
    kappa = xp.linalg.norm(xp.cross(d1, d2), axis=-1) / (
        xp.linalg.norm(d1, axis=-1) ** 3 + _EPS
    )
    K = xp.sum(wi[:, 1:-1] * (kappa * radius[:, 1:-1]) ** 2 * dl)

    if t_start is not None:
        ts = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        d2_0 = (t[:, 0] - ts) / (ds[:, 0, None] + _EPS)
        km0 = xp.linalg.norm(xp.cross(ts, d2_0), axis=-1) / (
            xp.linalg.norm(ts, axis=-1) ** 3 + _EPS
        )
        K = K + xp.sum(wi[:, 0] * (km0 * radius[:, 0]) ** 2 * ds[:, 0])
    if t_end is not None:
        te = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        d2_1 = (te - t[:, -1]) / (ds[:, -1, None] + _EPS)
        km1 = xp.linalg.norm(xp.cross(te, d2_1), axis=-1) / (
            xp.linalg.norm(te, axis=-1) ** 3 + _EPS
        )
        K = K + xp.sum(wi[:, -1] * (km1 * radius[:, -1]) ** 2 * ds[:, -1])
    return K


def k_hermite(
    p0,
    v0,
    p1,
    v1,
    radius,
    ts,
    w=None,
    t_start=None,
    t_end=None,
    backend: str = "numpy",
):
    """Cubic Hermite analytic κ: Σ w (κ(t) r(t))² Δs on one segment (§6.5.1).

    Interior `ts[1:-1]`: κ(t)=||γ'×γ''||/||γ'||³ via `eval_hermite_kappa`.
    Optional `t_start` / `t_end`: one-sided FD at terminals (same idea as `k_fd`) —
    γ'=t_port, γ''=(t_chord-t_port)/Δs — for port clamps that may differ from v0/v1.
    Prefer passing the port direction as v0/v1 when the Hermite is already clamped;
    then leave t_start/t_end as None. `ts` on [0,1], len>=3. `radius` scalar, (2,), or (n,).
    """
    xp = _backend(backend)
    p0, v0, p1, v1 = xp.asarray(p0), xp.asarray(v0), xp.asarray(p1), xp.asarray(v1)
    ts, radius = xp.asarray(ts), xp.asarray(radius)
    assert p0.ndim == 2 and p0.shape[-1] == 3, f"p0 need (B,3), got {p0.shape}"
    assert v0.shape == p0.shape and p1.shape == p0.shape and v1.shape == p0.shape
    assert ts.ndim == 1 and int(ts.shape[0]) >= 3, f"ts need (T,), T>=3, got {ts.shape}"
    B, n = int(p0.shape[0]), int(ts.shape[0])
    assert radius.shape == (B, n), f"radius need (B,T)=({B},{n}), got {radius.shape}"
    wi = xp.ones((B, n), dtype=p0.dtype) if w is None else xp.asarray(w)
    if w is not None:
        assert wi.shape == (B, n), f"w need (B,T)=({B},{n}), got {wi.shape}"
    if t_start is not None:
        t_start = xp.asarray(t_start)
    if t_end is not None:
        t_end = xp.asarray(t_end)
    _check_ports(t_start, t_end, B)

    xyz = eval_hermite(
        p0[:, None, :], v0[:, None, :], p1[:, None, :], v1[:, None, :], ts[None, :],
        backend=backend,
    )
    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    dl = 0.5 * (ds[:, :-1] + ds[:, 1:])
    t_chord = (xyz[:, 1:] - xyz[:, :-1]) / (ds[..., None] + _EPS)

    ti = ts[1:-1]
    d1 = eval_hermite_d1(
        p0[:, None, :], v0[:, None, :], p1[:, None, :], v1[:, None, :], ti[None, :],
        backend=backend,
    )
    d2 = eval_hermite_d2(
        p0[:, None, :], v0[:, None, :], p1[:, None, :], v1[:, None, :], ti[None, :],
        backend=backend,
    )
    kappa = xp.linalg.norm(xp.cross(d1, d2), axis=-1) / (
        xp.linalg.norm(d1, axis=-1) ** 3 + _EPS
    )
    K = xp.sum(wi[:, 1:-1] * (kappa * radius[:, 1:-1]) ** 2 * dl)

    if t_start is not None:
        tp = t_start / (xp.linalg.norm(t_start, axis=-1, keepdims=True) + _EPS)
        d2_0 = (t_chord[:, 0] - tp) / (ds[:, 0, None] + _EPS)
        km0 = xp.linalg.norm(xp.cross(tp, d2_0), axis=-1) / (
            xp.linalg.norm(tp, axis=-1) ** 3 + _EPS
        )
        K = K + xp.sum(wi[:, 0] * (km0 * radius[:, 0]) ** 2 * ds[:, 0])
    if t_end is not None:
        tp = t_end / (xp.linalg.norm(t_end, axis=-1, keepdims=True) + _EPS)
        d2_1 = (tp - t_chord[:, -1]) / (ds[:, -1, None] + _EPS)
        km1 = xp.linalg.norm(xp.cross(tp, d2_1), axis=-1) / (
            xp.linalg.norm(tp, axis=-1) ** 3 + _EPS
        )
        K = K + xp.sum(wi[:, -1] * (km1 * radius[:, -1]) ** 2 * ds[:, -1])
    return K


def graph_length(graph: OptimizeGraph, backend: str = "numpy", xyz=None):
    """Σ chord energy (path_length) over Hermite segments; each seg once.

    Pass ``xyz`` from ``refresh`` / ``refresh_from_x`` (no ``seg_*`` cache).
    """
    if xyz is None:
        _, xyz, _ = graph.refresh(backend=backend)
    return path_length(xyz, backend=backend)


def graph_curvature(graph: OptimizeGraph, backend: str = "numpy", xyz=None, radius=None):
    """Σ k_bend: each seg interior once + each unique 3-point joint once.

    Pass ``xyz``/``radius`` from ``refresh`` (no ``seg_*`` cache).
    """
    xp = _backend(backend)
    if xyz is None or radius is None:
        _, xyz, radius = graph.refresh(backend=backend)
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)

    # ports in u→v; unmasked → chord dir so port term is ~0
    t0 = (xyz[:, 1] - xyz[:, 0]) / (
        xp.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=-1, keepdims=True) + _EPS
    )
    t1 = (xyz[:, -1] - xyz[:, -2]) / (
        xp.linalg.norm(xyz[:, -1] - xyz[:, -2], axis=-1, keepdims=True) + _EPS
    )
    t_start = xp.where(xp.asarray(graph.port_v0_mask)[:, None], xp.asarray(graph.port_v0), t0)
    t_end = xp.where(xp.asarray(graph.port_v1_mask)[:, None], xp.asarray(graph.port_v1), t1)
    K = k_bend(xyz, radius, t_start=t_start, t_end=t_end, backend=backend)

    s0, s1 = xp.asarray(graph.joint_s0), xp.asarray(graph.joint_s1)
    if int(s0.shape[0]) == 0:
        return K
    f0, f1 = xp.asarray(graph.joint_flip0), xp.asarray(graph.joint_flip1)
    a = xp.where(f0[:, None, None], xyz[s0][:, ::-1, :], xyz[s0])
    b = xp.where(f1[:, None, None], xyz[s1][:, ::-1, :], xyz[s1])
    ra = xp.where(f0[:, None], radius[s0][:, ::-1], radius[s0])
    rb = xp.where(f1[:, None], radius[s1][:, ::-1], radius[s1])
    j_xyz = xp.concatenate([a[:, -2:, :], b[:, 1:2, :]], axis=1)
    j_r = xp.concatenate([ra[:, -2:], rb[:, 1:2]], axis=1)
    return K + k_bend(j_xyz, j_r, backend=backend)

