"""Paper figure: shared connect point (cond1) vs local junction split (cond2).

Left panel builds a Hermite Y where A/B/C arms share **one** node index ``J``
(by construction). After Adam, three usage centerlines still meet at that
single point — so ``A→B`` stays sharp while ``A→C`` / ``B→C`` can fair.

Right panel uses the production trunk-share / junction-split assemble path
(same as ``vis_bench``), so ``A→B`` can leave the fork.

Run:
  MPLBACKEND=Agg python pipe_smooth/paper/latex/figures/y_shape_shared_point.py

Writes:
  y_shape_shared_vs_split.pdf / .png
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

from pipe_smooth.geometry.assemble import (
    HermiteNode,
    HermiteSegment,
    OptimizeGraph,
    OptimizeUsagePath,
)
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.pipeline import run_optimize

# reuse helpers from sibling script (figures/ is not a Python package)
_spec = importlib.util.spec_from_file_location(
    "Y_shape_draw", _HERE / "Y_shape_draw.py"
)
assert _spec and _spec.loader
_yd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_yd)

_PORT_FACE = _yd._PORT_FACE
_USAGE_STYLE = _yd._USAGE_STYLE
_draw_tube = _yd._draw_tube
_draw_usage = _yd._draw_usage
_port_labels = _yd._port_labels
_xy = _yd._xy
make_y_shape_2d = _yd.make_y_shape_2d
optimize_y = _yd.optimize_y

OUT = _HERE


def _chain_points(a, b, n_seg: int) -> np.ndarray:
    """``n_seg`` segments → ``n_seg+1`` points from a to b (inclusive)."""
    a = np.asarray(a, float).reshape(3)
    b = np.asarray(b, float).reshape(3)
    t = np.linspace(0.0, 1.0, n_seg + 1)
    return (1.0 - t)[:, None] * a + t[:, None] * b


def make_shared_point_y(
    n_seg: int = 4,
    radius: float = 0.45,
    *,
    size_x: float = 12.0,
    size_y: float = 10.0,
    z: float = 2.0,
) -> OptimizeGraph:
    """Y-graph with a **single shared** junction node index for all three arms.

    Node layout (indices):
      A-arm:  A = 0, ..., J
      C-arm:  J, ..., C
      B-arm:  B, ..., J   (B is a separate port; last node is the same J)

    Usages A→C / B→C / A→B all pass through that one ``J``.
    """
    if n_seg < 2:
        raise ValueError(f"n_seg must be >= 2, got {n_seg}")
    r = float(radius)
    A = np.array([0.0, size_y - 2.0, z], float)
    B = np.array([0.0, 2.0, z], float)
    C = np.array([size_x, 0.5 * size_y, z], float)
    J = np.array([0.35 * size_x, 0.5 * size_y, z], float)

    arm_aj = _chain_points(A, J, n_seg)  # n_seg+1, last is J
    arm_jc = _chain_points(J, C, n_seg)  # first is J
    arm_bj = _chain_points(B, J, n_seg)  # last is J

    # unique nodes: A-arm without last J, then J once, then C-arm without first J,
    # then B-arm without last J
    pos_list = [arm_aj[i] for i in range(n_seg)]  # A .. before J
    i_j = len(pos_list)
    pos_list.append(J.copy())
    pos_list.extend(arm_jc[i] for i in range(1, n_seg + 1))  # after J .. C
    i_c = len(pos_list) - 1
    i_b0 = len(pos_list)
    pos_list.extend(arm_bj[i] for i in range(n_seg))  # B .. before J
    # B-arm connects last of these to shared J (no duplicate J node)

    pos = np.asarray(pos_list, float)
    N = len(pos)
    i_a, i_b = 0, i_b0
    assert i_c > i_j

    nodes = [
        HermiteNode(
            fixed_p=(i in (i_a, i_b, i_c)),
            fixed_v=(i in (i_a, i_b, i_c)),
            role="port" if i in (i_a, i_b, i_c) else ("junction" if i == i_j else "trunk"),
        )
        for i in range(N)
    ]

    segs: list[HermiteSegment] = []
    edge_segs: dict[str, list[int]] = {"e_aj": [], "e_jc": [], "e_bj": []}

    def add_seg(edge: str, u: int, v: int) -> int:
        sid = len(segs)
        segs.append(HermiteSegment(edge=edge, kind="trunk", u=u, v=v))
        edge_segs[edge].append(sid)
        return sid

    # A → J
    for k in range(n_seg - 1):
        add_seg("e_aj", k, k + 1)
    add_seg("e_aj", n_seg - 1, i_j)
    # J → C
    add_seg("e_jc", i_j, i_j + 1)
    for k in range(i_j + 1, i_c):
        add_seg("e_jc", k, k + 1)
    # B → J
    for k in range(i_b, i_b + n_seg - 1):
        add_seg("e_bj", k, k + 1)
    add_seg("e_bj", i_b + n_seg - 1, i_j)

    S = len(segs)
    vel = np.zeros((N, 3), float)
    for seg in segs:
        d = pos[seg.v] - pos[seg.u]
        L = float(np.linalg.norm(d))
        tang = d / (L + 1e-12)
        if not nodes[seg.u].fixed_v and np.linalg.norm(vel[seg.u]) < 1e-12:
            vel[seg.u] = tang * L
        if not nodes[seg.v].fixed_v and np.linalg.norm(vel[seg.v]) < 1e-12:
            vel[seg.v] = tang * L
    # shared J: average outgoing directions (still one point)
    d_ac = C - A
    d_bc = C - B
    t_j = 0.5 * (d_ac / (np.linalg.norm(d_ac) + 1e-12) + d_bc / (np.linalg.norm(d_bc) + 1e-12))
    t_j = t_j / (np.linalg.norm(t_j) + 1e-12)
    vel[i_j] = t_j * float(np.linalg.norm(J - A) / n_seg)

    v_a = (J - A)
    v_a = v_a / (np.linalg.norm(v_a) + 1e-12) * (np.linalg.norm(J - A) / n_seg)
    v_b = (J - B)
    v_b = v_b / (np.linalg.norm(v_b) + 1e-12) * (np.linalg.norm(J - B) / n_seg)
    v_c = (C - J)
    v_c = v_c / (np.linalg.norm(v_c) + 1e-12) * (np.linalg.norm(C - J) / n_seg)
    vel[i_a], vel[i_b], vel[i_c] = v_a, v_b, v_c

    segs_aj = edge_segs["e_aj"]
    segs_jc = edge_segs["e_jc"]
    segs_bj = edge_segs["e_bj"]

    usage_paths = [
        OptimizeUsagePath(
            usage=("A", "C"),
            edges=["e_aj", "e_jc"],
            edges_reverse=[False, False],
            segments=segs_aj + segs_jc,
            reverse=[False] * (len(segs_aj) + len(segs_jc)),
            v_start=v_a.copy(),
            v_end=v_c.copy(),
        ),
        OptimizeUsagePath(
            usage=("B", "C"),
            edges=["e_bj", "e_jc"],
            edges_reverse=[False, False],
            segments=segs_bj + segs_jc,
            reverse=[False] * (len(segs_bj) + len(segs_jc)),
            v_start=v_b.copy(),
            v_end=v_c.copy(),
        ),
        # A → B: walk A→J then reverse B→J
        OptimizeUsagePath(
            usage=("A", "B"),
            edges=["e_aj", "e_bj"],
            edges_reverse=[False, True],
            segments=segs_aj + list(reversed(segs_bj)),
            reverse=[False] * len(segs_aj) + [True] * len(segs_bj),
            v_start=v_a.copy(),
            v_end=-v_b.copy(),
        ),
    ]

    g = OptimizeGraph(
        id="y_shared_point",
        nodes=nodes,
        segments=segs,
        pos=pos,
        v=vel,
        radius=np.full(N, r),
        u_sign=np.ones(S),
        v_sign=np.ones(S),
        edge_segs=edge_segs,
        usage_paths=usage_paths,
    )
    g.check_usage_tangents_colinear()
    g = g.bind_ends_tables()
    g.refresh(backend="numpy", store=True)
    # stash J index for drawing
    g._paper_j_idx = i_j  # type: ignore[attr-defined]
    return g


def _find_shared_junction_xy(g: OptimizeGraph) -> np.ndarray | None:
    """If an explicit J index was stored, use it; else degree≥3 node."""
    if hasattr(g, "_paper_j_idx"):
        return _xy(g.pos[int(g._paper_j_idx)]).reshape(2)  # type: ignore[attr-defined]
    deg = np.zeros(len(g.nodes), dtype=int)
    for seg in g.segments:
        deg[seg.u] += 1
        deg[seg.v] += 1
    cands = np.where(deg >= 3)[0]
    if len(cands) == 0:
        return None
    return _xy(g.pos[int(cands[0])]).reshape(2)


def draw_panel(
    ax,
    g: OptimizeGraph,
    *,
    title: str,
    mark_j: bool,
    j_label: str = "J",
):
    """Same visual language as Y_shape_draw, plus optional shared-J marker."""
    g.refresh(backend="numpy", store=True)
    res = max(0.2, 0.35 * float(np.mean(g.radius)))
    segs_xy = []
    for s in range(len(g.segments)):
        xy = _xy(np.asarray(g.seg_xyz[s], float))
        segs_xy.append(xy)
        _draw_tube(ax, xy, float(np.mean(g.seg_radius[s])))

    legend_handles: list[Line2D] = []
    seen: set[str] = set()
    for path in g.usage_paths:
        key = (str(path.usage[0]), str(path.usage[1]))
        style = _USAGE_STYLE.get(key) or dict(
            color="#7f8c8d", zorder=3, label=f"{key[0]}→{key[1]}"
        )
        samp = g.sample_usage_path(path, resolution=res)
        _draw_usage(ax, _xy(samp["xyz"]), style)
        if style["label"] not in seen:
            seen.add(style["label"])
            legend_handles.append(
                Line2D([0], [0], color=style["color"], lw=2.4, label=style["label"])
            )

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

    if mark_j:
        jxy = _find_shared_junction_xy(g)
        if jxy is not None:
            ax.plot(
                jxy[0],
                jxy[1],
                marker="o",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="#e67e22",
                markeredgewidth=2.0,
                zorder=8,
            )
            ax.text(
                float(jxy[0]) + 0.35,
                float(jxy[1]) + 0.35,
                j_label,
                fontsize=12,
                fontweight="bold",
                color="#e67e22",
                zorder=9,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="#e67e22",
                    markeredgewidth=2.0,
                    markersize=8,
                    linestyle="None",
                    label="shared $J$",
                )
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True, alpha=0.25, lw=0.6)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92, fontsize=9)
    all_xy = np.concatenate(segs_xy, axis=0) if segs_xy else np.zeros((1, 2))
    rmax = float(np.max(g.radius)) if len(g.radius) else 0.5
    pad = 1.2 * rmax + 0.4
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)


def optimize_shared(g: OptimizeGraph, *, steps: int = 400, lr: float = 0.05) -> OptimizeGraph:
    g2 = copy.deepcopy(g)
    if hasattr(g, "_paper_j_idx"):
        g2._paper_j_idx = g._paper_j_idx  # type: ignore[attr-defined]
    w = LossWeights(
        alpha=1.0,
        beta=2.0,
        mu_obs=0.0,
        mu_grp=0.0,
        lam=0.0,  # no union — only one physical junction point
        eta=10.0,
        mu_v=0.0,
    )
    out = OUT / "_y_shared_opt_tmp"
    out.mkdir(parents=True, exist_ok=True)
    run_optimize(
        [g2],
        obstacles=[],
        method="adam",
        clear_kind="ignore",
        union_kind="ignore",
        weights=w,
        margin=0.0,
        steps=int(steps),
        p_max_union=64,
        refresh_every=10,
        out_dir=out,
        record_state=False,
        verbose=True,
        lr=float(lr),
    )
    g2.refresh(backend="numpy", store=True)
    return g2


def main():
    # --- cond1: forced shared J ---
    g_share0 = make_shared_point_y(n_seg=4, radius=0.45)
    print(
        f"shared-J graph: nodes={len(g_share0.nodes)} segs={len(g_share0.segments)} "
        f"J_idx={g_share0._paper_j_idx}"  # type: ignore[attr-defined]
    )
    g_share = optimize_shared(g_share0, steps=400, lr=0.05)
    j_after = g_share.pos[int(g_share._paper_j_idx)]  # type: ignore[attr-defined]
    print(f"shared J after opt: {j_after}")

    # --- cond2: production junction-split ---
    g_split0, _, _ = make_y_shape_2d(count=5, resolution=1.0, radius=0.45)
    g_split = optimize_y(g_split0, steps=400, lr=0.05)

    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)
    draw_panel(
        axes[0],
        g_share,
        title=r"(a) Shared $J$: $A\!\to\!B$ sharp at one point",
        mark_j=True,
    )
    draw_panel(
        axes[1],
        g_split,
        title=r"(b) Junction split: $A\!\to\!B$ can leave the fork",
        mark_j=False,
    )
    fig.savefig(OUT / "y_shape_shared_vs_split.pdf", dpi=200)
    fig.savefig(OUT / "y_shape_shared_vs_split.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / 'y_shape_shared_vs_split.pdf'}")


if __name__ == "__main__":
    main()
