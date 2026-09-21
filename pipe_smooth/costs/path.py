"""§6.4.1 — path length and curvature pieces.

Polyline samples: `k_bend` / `k_bend_approx` / `k_menger` / `k_fd`.
Cubic Hermite segment: `k_hermite` (analytic κ(t)). Caller sums / weights costs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import eval_hermite_kappa, eval_hermite_ts, sample_hermite

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
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)
    if radius.ndim == 0:
        radius = xp.full((xyz.shape[0],), radius)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    t = (xyz[1:] - xyz[:-1]) / (ds[:, None] + _EPS)
    dl = 0.5 * (ds[:-1] + ds[1:])
    wi = xp.ones((xyz.shape[0],), dtype=xyz.dtype) if w is None else xp.asarray(w)

    # θ ∈ [0,π]; avoid arccos (singular as cos→1)
    theta = xp.arctan2(
        xp.linalg.norm(xp.cross(t[:-1], t[1:]), axis=-1),
        xp.sum(t[:-1] * t[1:], axis=-1),
    )
    K = xp.sum(wi[1:-1] * (theta ** 2) * (radius[1:-1] ** 2) / (dl + _EPS))

    if t_start is not None:
        ts = xp.asarray(t_start)
        ts = ts / (xp.linalg.norm(ts) + _EPS)
        th0 = xp.arctan2(xp.linalg.norm(xp.cross(ts, t[0])), xp.sum(ts * t[0]))
        K = K + wi[0] * (th0 ** 2) * (radius[0] ** 2) / (ds[0] + _EPS)
    if t_end is not None:
        te = xp.asarray(t_end)
        te = te / (xp.linalg.norm(te) + _EPS)
        th1 = xp.arctan2(xp.linalg.norm(xp.cross(t[-1], te)), xp.sum(t[-1] * te))
        K = K + wi[-1] * (th1 ** 2) * (radius[-1] ** 2) / (ds[-1] + _EPS)
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
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)
    if radius.ndim == 0:
        radius = xp.full((xyz.shape[0],), radius)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    t = (xyz[1:] - xyz[:-1]) / (ds[:, None] + _EPS)
    dl = 0.5 * (ds[:-1] + ds[1:])
    wi = xp.ones((xyz.shape[0],), dtype=xyz.dtype) if w is None else xp.asarray(w)

    dt2 = xp.sum((t[1:] - t[:-1]) ** 2, axis=-1)
    K = xp.sum(wi[1:-1] * dt2 * (radius[1:-1] ** 2) / (dl + _EPS))

    if t_start is not None:
        ts = xp.asarray(t_start)
        ts = ts / (xp.linalg.norm(ts) + _EPS)
        K = K + wi[0] * xp.sum((t[0] - ts) ** 2) * (radius[0] ** 2) / (ds[0] + _EPS)
    if t_end is not None:
        te = xp.asarray(t_end)
        te = te / (xp.linalg.norm(te) + _EPS)
        K = K + wi[-1] * xp.sum((te - t[-1]) ** 2) * (radius[-1] ** 2) / (ds[-1] + _EPS)
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
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)
    if radius.ndim == 0:
        radius = xp.full((xyz.shape[0],), radius)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    dl = 0.5 * (ds[:-1] + ds[1:])
    wi = xp.ones((xyz.shape[0],), dtype=xyz.dtype) if w is None else xp.asarray(w)

    a = xyz[1:-1] - xyz[:-2]
    b = xyz[2:] - xyz[1:-1]
    c = xyz[2:] - xyz[:-2]
    cross = xp.linalg.norm(xp.cross(a, b), axis=-1)
    denom = (
        xp.linalg.norm(a, axis=-1) * xp.linalg.norm(b, axis=-1) * xp.linalg.norm(c, axis=-1)
        + _EPS
    )
    kappa_m = 2.0 * cross / denom
    K = xp.sum(wi[1:-1] * (kappa_m * radius[1:-1]) ** 2 * dl)

    # Terminals: κ ≈ θ / Δs from port tangent vs first/last chord (not 3-point Menger)
    if t_start is not None:
        ts = xp.asarray(t_start)
        ts = ts / (xp.linalg.norm(ts) + _EPS)
        t0 = (xyz[1] - xyz[0]) / (ds[0] + _EPS)
        th0 = xp.arctan2(xp.linalg.norm(xp.cross(ts, t0)), xp.sum(ts * t0))
        km0 = th0 / (ds[0] + _EPS)
        K = K + wi[0] * (km0 * radius[0]) ** 2 * ds[0]
    if t_end is not None:
        te = xp.asarray(t_end)
        te = te / (xp.linalg.norm(te) + _EPS)
        t1 = (xyz[-1] - xyz[-2]) / (ds[-1] + _EPS)
        th1 = xp.arctan2(xp.linalg.norm(xp.cross(t1, te)), xp.sum(t1 * te))
        km1 = th1 / (ds[-1] + _EPS)
        K = K + wi[-1] * (km1 * radius[-1]) ** 2 * ds[-1]
    return K


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
    xyz = xp.asarray(xyz)
    radius = xp.asarray(radius)
    if radius.ndim == 0:
        radius = xp.full((xyz.shape[0],), radius)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    t = (xyz[1:] - xyz[:-1]) / (ds[:, None] + _EPS)
    dl = 0.5 * (ds[:-1] + ds[1:])
    wi = xp.ones((xyz.shape[0],), dtype=xyz.dtype) if w is None else xp.asarray(w)

    # Interior: centered chords for γ', Δt/Δℓ for γ'' — O(h) (first-order) on uneven samples
    d1 = (xyz[2:] - xyz[:-2]) / ((ds[:-1] + ds[1:])[:, None] + _EPS)
    # d2 ≈ γ'': difference of successive unit chord tangents t_i=(x_{i+1}-x_i)/Δs_i,
    # divided by local spacing Δℓ=½(Δs_{i-1}+Δs_i)
    d2 = (t[1:] - t[:-1]) / (dl[:, None] + _EPS)
    kappa = xp.linalg.norm(xp.cross(d1, d2), axis=-1) / (xp.linalg.norm(d1, axis=-1) ** 3 + _EPS)
    K = xp.sum(wi[1:-1] * (kappa * radius[1:-1]) ** 2 * dl)

    # Terminals: one-sided (forward / backward) using port unit tangents
    if t_start is not None:
        ts = xp.asarray(t_start)
        ts = ts / (xp.linalg.norm(ts) + _EPS)
        d2_0 = (t[0] - ts) / (ds[0] + _EPS)
        km0 = xp.linalg.norm(xp.cross(ts, d2_0)) / (xp.linalg.norm(ts) ** 3 + _EPS)
        K = K + wi[0] * (km0 * radius[0]) ** 2 * ds[0]
    if t_end is not None:
        te = xp.asarray(t_end)
        te = te / (xp.linalg.norm(te) + _EPS)
        d2_1 = (te - t[-1]) / (ds[-1] + _EPS)
        km1 = xp.linalg.norm(xp.cross(te, d2_1)) / (xp.linalg.norm(te) ** 3 + _EPS)
        K = K + wi[-1] * (km1 * radius[-1]) ** 2 * ds[-1]
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
    ts = xp.asarray(ts)
    n = int(ts.shape[0])
    if n < 3:
        raise ValueError(f"k_hermite needs len(ts)>=3, got {n}")

    xyz = eval_hermite_ts(p0, v0, p1, v1, ts, backend=backend)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    dl = 0.5 * (ds[:-1] + ds[1:])
    t_chord = (xyz[1:] - xyz[:-1]) / (ds[:, None] + _EPS)

    radius = xp.asarray(radius)
    if radius.ndim == 0:
        r = xp.full((n,), radius)
    elif radius.shape == (2,):
        r = (1.0 - ts) * radius[0] + ts * radius[1]
    else:
        r = radius

    wi = xp.ones((n,), dtype=xyz.dtype) if w is None else xp.asarray(w)
    K = 0.0
    for i in range(1, n - 1):
        kappa = eval_hermite_kappa(p0, v0, p1, v1, ts[i], backend=backend)
        K = K + wi[i] * (kappa * r[i]) ** 2 * dl[i - 1]

    # Optional port clamps: one-sided FD (do not replace analytic κ at ends — Hermite
    # already has κ(0),κ(1) from v0,v1; use these when port direction ≠ segment v)
    if t_start is not None:
        tp = xp.asarray(t_start)
        tp = tp / (xp.linalg.norm(tp) + _EPS)
        d2_0 = (t_chord[0] - tp) / (ds[0] + _EPS)
        km0 = xp.linalg.norm(xp.cross(tp, d2_0)) / (xp.linalg.norm(tp) ** 3 + _EPS)
        K = K + wi[0] * (km0 * r[0]) ** 2 * ds[0]
    if t_end is not None:
        tp = xp.asarray(t_end)
        tp = tp / (xp.linalg.norm(tp) + _EPS)
        d2_1 = (tp - t_chord[-1]) / (ds[-1] + _EPS)
        km1 = xp.linalg.norm(xp.cross(tp, d2_1)) / (xp.linalg.norm(tp) ** 3 + _EPS)
        K = K + wi[-1] * (km1 * r[-1]) ** 2 * ds[-1]
    return K


def graph_length(graph: OptimizeGraph, backend: str = "numpy"):
    """Σ length over Hermite segments; each seg_id once (usage walk for port clamps).

    Sample count from ``HermiteSegment.ts_count`` (set by assemble / sampling upstream).
    """
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
            total = total + path_length(xyz, backend=backend)
    return total


def graph_curvature(graph: OptimizeGraph, backend: str = "numpy"):
    """Σ k_bend over Hermite segments; each seg_id once (usage walk for port clamps).

    Sample count from ``HermiteSegment.ts_count``. Seen segments are skipped for
    cost but still sampled so the next new segment can prepend ``prev[-2:]`` and
    pick up the C¹ joint turn (avoid zero-length concat of ``prev[-1]`` alone).
    """
    xp = _backend(backend)
    total = xp.asarray(0.0)
    seen: set[str] = set()
    for path in graph.usage_paths:
        nseg = len(path.segments)
        prev_xyz = prev_r = None
        for i, (sid, rev) in enumerate(zip(path.segments, path.reverse)):
            seg = graph.segments[sid]
            nu, nv = graph.nodes[seg.u], graph.nodes[seg.v]
            if rev:
                p0, v0 = nv.pos, -nv.v  # type: ignore[operator]
                p1, v1 = nu.pos, -nu.v  # type: ignore[operator]
                r0, r1 = float(nv.radius), float(nu.radius)
            else:
                p0, v0 = nu.pos, nu.v
                p1, v1 = nv.pos, nv.v
                r0, r1 = float(nu.radius), float(nv.radius)
            if i == 0:
                v0 = path.v_start
            if i == nseg - 1:
                v1 = path.v_end
            n = max(2, int(seg.ts_count))
            xyz = sample_hermite(p0, v0, p1, v1, n=n, backend=backend)
            ts = xp.linspace(0.0, 1.0, n)
            radius = (1.0 - ts) * r0 + ts * r1

            if sid not in seen:
                seen.add(sid)
                t_end = path.v_end if i == nseg - 1 else None
                if i == 0 or prev_xyz is None:
                    total = total + k_bend(
                        xyz,
                        radius,
                        t_start=path.v_start if i == 0 else None,
                        t_end=t_end,
                        backend=backend,
                    )
                else:
                    # joint + current interior; do not rescore previous interiors
                    xyz_c = xp.concatenate([prev_xyz[-2:], xyz[1:]], axis=0)
                    r_c = xp.concatenate([prev_r[-2:], radius[1:]], axis=0)  # type: ignore
                    total = total + k_bend(
                        xyz_c, r_c, t_start=None, t_end=t_end, backend=backend
                    )

            prev_xyz, prev_r = xyz, radius
    return total

