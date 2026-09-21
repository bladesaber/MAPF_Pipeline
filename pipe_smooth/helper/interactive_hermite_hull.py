"""Interactive cubic Hermite — full + subdivided Bézier convex hulls.

Sliders: pos0 / pos1 / tag0 / tag1 / σ / n (sub-pieces via bezier_on_interval).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.widgets import Slider
import numpy as np

from pipe_smooth.geometry.hermite import (
    bezier_on_interval,
    eval_hermite_ts,
    hermite_bezier_controls,
    hermite_length,
    unit,
)

# ---------------------------------------------------------------------------
N_DRAW = 200
TANGENT_DRAW = 0.45
N_MAX = 16  # max sub-hull patches

SLIDER_SPECS = [
    ("pos0x", -1.0, 3.0, 0.0),
    ("pos0y", -2.0, 2.0, 0.0),
    ("pos1x", -1.0, 3.0, 2.0),
    ("pos1y", -2.0, 2.0, 0.0),
    ("tag0°", -180.0, 180.0, 40.0),
    ("tag1°", -180.0, 180.0, -40.0),
    ("σ0", 0.2, 5.0, 2.0),
    ("σ1", 0.2, 5.0, 2.0),
    ("n", 1.0, float(N_MAX), 4.0),
]


def _params_from_sliders(sliders: dict[str, Slider]):
    p0 = np.array([sliders["pos0x"].val, sliders["pos0y"].val, 0.0])
    p1 = np.array([sliders["pos1x"].val, sliders["pos1y"].val, 0.0])
    a0 = np.radians(sliders["tag0°"].val)
    a1 = np.radians(sliders["tag1°"].val)
    v0 = unit(np.array([np.cos(a0), np.sin(a0), 0.0]))
    v1 = unit(np.array([np.cos(a1), np.sin(a1), 0.0]))
    s0 = float(sliders["σ0"].val)
    s1 = float(sliders["σ1"].val)
    n = max(1, int(round(sliders["n"].val)))
    return p0, s0 * v0, p1, s1 * v1, s0, s1, n


def convex_hull_2d(pts_xy: np.ndarray) -> np.ndarray:
    """Andrew monotone chain. pts (n,2) → hull CCW (h,2)."""
    pts = np.asarray(pts_xy, dtype=float)
    pts = np.unique(np.round(pts, decimals=12), axis=0)
    if pts.shape[0] <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _set_hull_patch(patch: Polygon, hull: np.ndarray, visible: bool) -> None:
    if not visible or hull.shape[0] < 2:
        patch.set_visible(False)
        return
    if hull.shape[0] == 2:
        patch.set_xy(np.vstack([hull, hull[0]]))
    else:
        patch.set_xy(hull)
    patch.set_visible(True)


def main() -> None:
    fig = plt.figure(figsize=(10, 9.0))
    ax = fig.add_axes([0.08, 0.42, 0.84, 0.52])  # type: ignore

    sliders: dict[str, Slider] = {}
    y0, h, gap = 0.34, 0.022, 0.006
    for i, (name, vmin, vmax, vinit) in enumerate(SLIDER_SPECS):
        ax_s = fig.add_axes([0.12, y0 - i * (h + gap), 0.78, h])  # type: ignore
        fmt = "%.0f" if name == "n" else "%.2f"
        sliders[name] = Slider(ax_s, name, vmin, vmax, valinit=vinit, valfmt=fmt)

    ts_draw = np.linspace(0.0, 1.0, N_DRAW)
    cmap = plt.get_cmap("tab10")

    (line_curve,) = ax.plot([], [], "k-", lw=2.4, label="Hermite γ(t)", zorder=5)
    (line_ctrl,) = ax.plot(
        [],
        [],
        "o--",
        color="0.35",
        lw=1.2,
        ms=6,
        label="full Bézier C0..C3",
        zorder=6,
        alpha=0.7,
    )
    # full hull (faint outline)
    full_hull = Polygon(
        np.zeros((3, 2)),
        closed=True,
        facecolor="none",
        edgecolor="0.4",
        lw=1.5,
        ls="--",
        label=r"full CH$\{C_i\}$",
        zorder=2,
    )
    ax.add_patch(full_hull)

    # sub-hulls
    sub_patches: list[Polygon] = []
    sub_ctrl_lines = []
    for k in range(N_MAX):
        color = cmap(k % 10)
        patch = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.28,
            lw=1.4,
            zorder=3,
            label=r"sub CH$_k$" if k == 0 else None,
        )
        ax.add_patch(patch)
        sub_patches.append(patch)
        (ln,) = ax.plot([], [], "-", color=color, lw=1.0, alpha=0.85, zorder=4)
        sub_ctrl_lines.append(ln)

    (line_chord,) = ax.plot([], [], ":", color="0.55", lw=1, label="chord", zorder=2)
    scat_ends = ax.scatter([0.0, 2.0], [0.0, 0.0], c=["C0", "C3"], s=70, zorder=7)

    def _make_tangent_arrow(color: str, label: str) -> FancyArrowPatch:
        arr = FancyArrowPatch(
            (0.0, 0.0),
            (TANGENT_DRAW, 0.0),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2.0,
            color=color,
            zorder=8,
            label=label,
        )
        ax.add_patch(arr)
        return arr

    arrow0 = _make_tangent_arrow("C0", "tag0 (γ')")
    arrow1 = _make_tangent_arrow("C3", "tag1 (γ')")
    # tip markers / text near arrow heads
    tag0_txt = ax.text(0.0, 0.0, "tag0", color="C0", fontsize=9, ha="left", va="bottom", zorder=9)
    tag1_txt = ax.text(0.0, 0.0, "tag1", color="C3", fontsize=9, ha="left", va="bottom", zorder=9)

    title = ax.set_title("")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    info = fig.text(0.08, 0.015, "", family="monospace", fontsize=8, va="bottom")

    def _set_tangent_arrow(arr: FancyArrowPatch, origin, vel, txt) -> None:
        """Draw arrow at origin along unit(vel), fixed visual length TANGENT_DRAW."""
        u = unit(np.asarray(vel, dtype=float))
        tip = (float(origin[0] + TANGENT_DRAW * u[0]), float(origin[1] + TANGENT_DRAW * u[1]))
        base = (float(origin[0]), float(origin[1]))
        arr.set_positions(base, tip)
        txt.set_position((tip[0] + 0.04, tip[1] + 0.04))

    def redraw(_=None) -> None:
        p0, v0, p1, v1, s0, s1, n = _params_from_sliders(sliders)

        xyz = eval_hermite_ts(p0, v0, p1, v1, ts_draw)
        C = hermite_bezier_controls(p0, v0, p1, v1)
        L = float(hermite_length(p0, v0, p1, v1))

        line_curve.set_data(xyz[:, 0], xyz[:, 1])
        line_ctrl.set_data(C[:, 0], C[:, 1])
        line_chord.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        scat_ends.set_offsets(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]))
        _set_hull_patch(full_hull, convex_hull_2d(C[:, :2]), True)  # type: ignore

        all_ctrl = [C]
        for k in range(N_MAX):
            if k < n:
                ta, tb = k / n, (k + 1) / n
                Ck = np.asarray(bezier_on_interval(C, ta, tb))
                all_ctrl.append(Ck)
                _set_hull_patch(sub_patches[k], convex_hull_2d(Ck[:, :2]), True)
                sub_ctrl_lines[k].set_data(Ck[:, 0], Ck[:, 1])
                sub_ctrl_lines[k].set_visible(True)
            else:
                sub_patches[k].set_visible(False)
                sub_ctrl_lines[k].set_visible(False)

        _set_tangent_arrow(arrow0, p0, v0, tag0_txt)
        _set_tangent_arrow(arrow1, p1, v1, tag1_txt)

        pad = 0.45
        stack = np.vstack([xyz] + all_ctrl)
        # include arrow tips in view
        u0, u1 = unit(v0), unit(v1)
        tips = np.array(
            [
                [p0[0] + TANGENT_DRAW * u0[0], p0[1] + TANGENT_DRAW * u0[1], 0.0],
                [p1[0] + TANGENT_DRAW * u1[0], p1[1] + TANGENT_DRAW * u1[1], 0.0],
            ]
        )
        stack = np.vstack([stack, tips])
        ax.set_xlim(float(stack[:, 0].min()) - pad, float(stack[:, 0].max()) + pad)
        ax.set_ylim(float(stack[:, 1].min()) - pad, float(stack[:, 1].max()) + pad)

        title.set_text(
            rf"Hermite ⊂ ∪$_k$ CH{{C$^{{(k)}}$}}   n={n}   "
            rf"σ0={s0:.2f}, σ1={s1:.2f}, L≈{L:.3f}"
        )
        info.set_text(
            f"intervals [k/n,(k+1)/n] via bezier_on_interval; "
            f"arrows = unit tag0/tag1; full C0..C3 dashed; colored = sub hulls"
        )
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(redraw)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
