"""Helpers for planar Y: junction-split geometry, Adam refine, 2D draw.

Paper figures are exported by ``y_shape_shared_point.py`` (shared-J vs split).
This module is imported for ``make_y_shape_2d`` / ``optimize_y`` / draw utils.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

# headless export (no Qt / display)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# repo root on path when run as a script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]  # .../MAPF_Pipeline
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.pipeline import run_optimize
from pipe_smooth.vis_bench.conditions import _assemble, _polyline

# paper-friendly colors (distinct usages; dark ports)
_USAGE_STYLE = {
    ("A", "C"): dict(color="#c0392b", zorder=3, label=r"$A\!\to\!C$"),
    ("C", "A"): dict(color="#c0392b", zorder=3, label=r"$A\!\to\!C$"),
    ("B", "C"): dict(color="#2980b9", zorder=3, label=r"$B\!\to\!C$"),
    ("C", "B"): dict(color="#2980b9", zorder=3, label=r"$B\!\to\!C$"),
    ("A", "B"): dict(color="#27ae60", zorder=4, label=r"$A\!\to\!B$"),
    ("B", "A"): dict(color="#27ae60", zorder=4, label=r"$A\!\to\!B$"),
}
_TUBE_FACE = "#bdc3c7"
_PORT_FACE = "#2c3e50"


def make_y_shape_2d(
    count: int = 5,
    resolution: float = 1.0,
    radius: float = 0.45,
):
    """Planar Y: A/B on the left face ``x=0``, C on ``x=max``, fork interior.

    Ports sit on exactly one domain face (required by ``build_junctions``).
    Same tree→junction→Hermite path as ``vis_bench`` examples.
    Usages ``(A,C), (B,C), (A,B)`` so the A↔B fairing conflict is explicit.
    """
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    sp, r = float(resolution), float(radius)
    n_cells = int(count) - 1
    # box large enough that A/B are not on y-edges (would be corners)
    size_x = max(3 * n_cells, 6)
    size_y = max(2 * n_cells + 2, 6)
    size_z = 4
    domain = {
        "resolution": sp,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }
    sx, sy, sz = size_x * sp, size_y * sp, size_z * sp
    z = 0.5 * sz  # mid-plane; not on z faces
    A = [0.0, sy - 2.0 * sp, z]
    B = [0.0, 2.0 * sp, z]
    C = [sx, 0.5 * sy, z]
    F = [0.35 * sx, 0.5 * sy, z]
    terminals = [
        {"id": "A", "pos": A, "radius": r},
        {"id": "B", "pos": B, "radius": r},
        {"id": "C", "pos": C, "radius": r},
    ]
    polylines = [
        _polyline(A, F, max(n_cells, 1)),
        _polyline(B, F, max(n_cells, 1)),
        _polyline(F, C, max(n_cells, 1)),
    ]
    usages = [("A", "C"), ("B", "C"), ("A", "B")]
    return _assemble("y_shape", terminals, polylines, usages, domain, r)


def _xy(xyz: np.ndarray) -> np.ndarray:
    return np.asarray(xyz, float).reshape(-1, 3)[:, :2]


def _draw_tube(ax, xy: np.ndarray, radius: float, *, face=_TUBE_FACE, alpha=0.55):
    """Approximate a constant-radius tube in 2D with sample disks + centerline."""
    if len(xy) < 2:
        return
    r = float(radius)
    step = max(1, len(xy) // 48)
    for p in xy[::step]:
        ax.add_patch(
            Circle(
                (float(p[0]), float(p[1])),
                r,
                facecolor=face,
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
        )
    # end caps
    for p in (xy[0], xy[-1]):
        ax.add_patch(
            Circle(
                (float(p[0]), float(p[1])),
                r,
                facecolor=face,
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
        )


def _draw_usage(ax, xy: np.ndarray, style: dict, *, lw=2.4):
    if len(xy) < 2:
        return
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=style["color"],
        lw=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=style.get("zorder", 3),
    )


def _port_labels(g: OptimizeGraph) -> dict[str, np.ndarray]:
    """Map terminal id → position from usage path ends / port nodes."""
    # Prefer usage endpoints: first/last sample of each usage path.
    out: dict[str, np.ndarray] = {}
    for path in g.usage_paths:
        samp = g.sample_usage_path(path, resolution=max(0.25, float(np.mean(g.radius))))
        xy = _xy(samp["xyz"])
        if len(xy) == 0:
            continue
        a, b = path.usage
        out.setdefault(str(a), xy[0].copy())
        out.setdefault(str(b), xy[-1].copy())
    # fallback: fixed port nodes
    for i, node in enumerate(g.nodes):
        if node.role == "port" and node.fixed_p:
            # unnamed ports — skip if already have A/B/C
            pass
    return out


def draw_y(
    ax,
    g: OptimizeGraph,
    *,
    title: str,
    show_usages: tuple[tuple[str, str], ...] | None = None,
    sample_res: float | None = None,
):
    """Top-down (x–y) draw of one graph: gray tube + colored usage centerlines."""
    g.refresh(backend="numpy", store=True)
    res = float(sample_res if sample_res is not None else max(0.2, 0.35 * float(np.mean(g.radius))))

    # shared physical tube from all segments (once)
    segs_xy = []
    for s in range(len(g.segments)):
        xyz = np.asarray(g.seg_xyz[s], float)
        segs_xy.append(_xy(xyz))
        _draw_tube(ax, _xy(xyz), float(np.mean(g.seg_radius[s])))

    # usage overlays
    wanted = set(show_usages) if show_usages is not None else None
    legend_handles: list[Line2D] = []
    seen_label: set[str] = set()
    for path in g.usage_paths:
        key = (str(path.usage[0]), str(path.usage[1]))
        key_rev = (key[1], key[0])
        if wanted is not None and key not in wanted and key_rev not in wanted:
            continue
        style = _USAGE_STYLE.get(key) or _USAGE_STYLE.get(key_rev) or dict(
            color="#7f8c8d", zorder=3, label=f"{key[0]}→{key[1]}"
        )
        samp = g.sample_usage_path(path, resolution=res)
        xy = _xy(samp["xyz"])
        _draw_usage(ax, xy, style)
        lab = style["label"]
        if lab not in seen_label:
            seen_label.add(lab)
            legend_handles.append(
                Line2D([0], [0], color=style["color"], lw=2.4, label=lab)
            )

    # ports
    ports = _port_labels(g)
    for name, p in ports.items():
        ax.add_patch(
            Circle(
                (float(p[0]), float(p[1])),
                0.12,
                facecolor=_PORT_FACE,
                edgecolor="white",
                lw=0.8,
                zorder=6,
            )
        )
        # label offset away from fork (left for A/B, right for C)
        dx = -0.35 if name in ("A", "B") else 0.28
        dy = 0.28 if name == "A" else (-0.35 if name == "B" else 0.05)
        ax.text(
            float(p[0]) + dx,
            float(p[1]) + dy,
            name,
            fontsize=12,
            fontweight="bold",
            color=_PORT_FACE,
            ha="center",
            va="center",
            zorder=7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, alpha=0.25, lw=0.6)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92, fontsize=9)
    # tight limits from geometry
    all_xy = np.concatenate(segs_xy, axis=0) if segs_xy else np.zeros((1, 2))
    rmax = float(np.max(g.radius)) if len(g.radius) else 0.5
    pad = 1.2 * rmax + 0.4
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)


def optimize_y(
    g: OptimizeGraph,
    *,
    steps: int = 400,
    lr: float = 0.05,
) -> OptimizeGraph:
    """Adam refine; no obstacles — focus on path / junction geometry."""
    g2 = copy.deepcopy(g)
    w = LossWeights(
        alpha=1.0,
        beta=2.0,
        mu_obs=0.0,
        mu_grp=0.0,
        lam=0.15,
        eta=10.0,
        mu_v=0.0,
    )
    out = _HERE / "_y_shape_opt_tmp"
    out.mkdir(parents=True, exist_ok=True)
    run_optimize(
        [g2],
        obstacles=[],
        method="adam",
        clear_kind="ignore",
        union_kind="chamfer",
        weights=w,
        margin=0.0,
        steps=int(steps),
        p_max_obs=64,
        p_max_grp=64,
        p_max_union=512,
        refresh_every=10,
        out_dir=out,
        record_state=False,
        verbose=True,
        lr=float(lr),
    )
    g2.refresh(backend="numpy", store=True)
    return g2


if __name__ == "__main__":
    print("Helpers only. Export paper figures with y_shape_shared_point.py")
