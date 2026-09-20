"""Interactive Hermite + 5 curvature costs — drag sliders to control ends."""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pipe_smooth.costs.path import k_bend, k_bend_approx, k_fd, k_hermite, k_menger
from pipe_smooth.geometry.hermite import eval_hermite_kappa, eval_hermite_ts, hermite_length, unit

# ---------------------------------------------------------------------------
# Defaults / ranges (xy plane; z fixed at 0)
# ---------------------------------------------------------------------------
RADIUS = 1.0
N_SAMPLES = 48
TANGENT_DRAW = 0.45

METHODS = ("k_bend", "k_bend_approx", "k_menger", "k_fd", "k_hermite")
COLORS = {
    "k_bend": "C0",
    "k_bend_approx": "C1",
    "k_menger": "C2",
    "k_fd": "C3",
    "k_hermite": "C4",
}

# slider: (label, vmin, vmax, vinit)
SLIDER_SPECS = [
    ("p1x", -1.0, 3.0, 0.0),
    ("p1y", -2.0, 2.0, 0.0),
    ("p2x", -1.0, 3.0, 2.0),
    ("p2y", -2.0, 2.0, 0.0),
    ("t1_ang°", -180.0, 180.0, 30.0),
    ("t2_ang°", -180.0, 180.0, -30.0),
    ("σ1", 0.2, 5.0, 2.0),
    ("σ2", 0.2, 5.0, 2.0),
]


def _params_from_sliders(sliders: dict[str, Slider]):
    p0 = np.array([sliders["p1x"].val, sliders["p1y"].val, 0.0])
    p1 = np.array([sliders["p2x"].val, sliders["p2y"].val, 0.0])
    a1 = np.radians(sliders["t1_ang°"].val)
    a2 = np.radians(sliders["t2_ang°"].val)
    v0 = unit(np.array([np.cos(a1), np.sin(a1), 0.0]))
    v1 = unit(np.array([np.cos(a2), np.sin(a2), 0.0]))
    s0 = float(sliders["σ1"].val)
    s1 = float(sliders["σ2"].val)
    return p0, v0, s0, p1, v1, s1


def _eval_five(p0, v0, s0, p1, v1, s1, xyz, ts) -> dict[str, float]:
    r = RADIUS
    return {
        "k_bend": float(k_bend(xyz, r)),
        "k_bend_approx": float(k_bend_approx(xyz, r)),
        "k_menger": float(k_menger(xyz, r)),
        "k_fd": float(k_fd(xyz, r)),
        "k_hermite": float(k_hermite(p0, v0, s0, p1, v1, s1, r, ts)),
    }


def main() -> None:
    fig = plt.figure(figsize=(12, 8.5))
    ax_curve = fig.add_axes([0.07, 0.42, 0.52, 0.52])  # type: ignore
    ax_bar = fig.add_axes([0.66, 0.42, 0.30, 0.52])  # type: ignore

    # slider axes stacked at bottom
    sliders: dict[str, Slider] = {}
    n = len(SLIDER_SPECS)
    y0, h, gap = 0.32, 0.025, 0.008
    for i, (name, vmin, vmax, vinit) in enumerate(SLIDER_SPECS):
        ax_s = fig.add_axes([0.12, y0 - i * (h + gap), 0.78, h])  # type: ignore
        sliders[name] = Slider(ax_s, name, vmin, vmax, valinit=vinit, valfmt="%.2f")

    ts = np.linspace(0.0, 1.0, max(N_SAMPLES, 3))
    ts_draw = np.linspace(0.0, 1.0, 200)

    # artists to update
    (line_curve,) = ax_curve.plot([], [], "k-", lw=2.2, label="Hermite γ(t)")
    (pts_samp,) = ax_curve.plot([], [], "o", ms=3.5, color="0.45", label="samples")
    (line_chord,) = ax_curve.plot([], [], ":", color="0.6", lw=1, label="chord")
    scat_ends = ax_curve.scatter([0.0, 2.0], [0.0, 0.0], c=["C0", "C3"], s=60, zorder=5)
    arrow1 = ax_curve.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.8),
    )
    arrow2 = ax_curve.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="C3", lw=1.8),
    )
    title_curve = ax_curve.set_title("")
    ax_curve.set_aspect("equal", adjustable="datalim")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=8, loc="upper right")
    ax_curve.set_xlabel("x")
    ax_curve.set_ylabel("y")

    bars = ax_bar.bar(METHODS, np.zeros(len(METHODS)), color=[COLORS[m] for m in METHODS], edgecolor="k", lw=0.6)
    ax_bar.set_ylabel("K")
    ax_bar.tick_params(axis="x", rotation=18)
    ax_bar.grid(True, axis="y", alpha=0.3)
    bar_texts = [
        ax_bar.text(0, 0, "", ha="center", va="bottom", fontsize=8) for _ in METHODS
    ]
    title_bar = ax_bar.set_title("Five curvature costs")
    info = fig.text(0.07, 0.01, "", family="monospace", fontsize=8, va="bottom")

    def redraw(_=None) -> None:
        p0, v0, s0, p1, v1, s1 = _params_from_sliders(sliders)
        xyz = eval_hermite_ts(p0, v0, s0, p1, v1, s1, ts)
        xyz_d = eval_hermite_ts(p0, v0, s0, p1, v1, s1, ts_draw)
        kappa_s = np.array(
            [float(eval_hermite_kappa(p0, v0, s0, p1, v1, s1, t)) for t in ts]
        )
        vals = _eval_five(p0, v0, s0, p1, v1, s1, xyz, ts)
        L = float(hermite_length(p0, v0, s0, p1, v1, s1))

        line_curve.set_data(xyz_d[:, 0], xyz_d[:, 1])
        pts_samp.set_data(xyz[:, 0], xyz[:, 1])
        line_chord.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        scat_ends.set_offsets(np.array([[p0[0], p0[1]], [p1[0], p1[1]]]))
        arrow1.xy = (p0[0] + TANGENT_DRAW * v0[0], p0[1] + TANGENT_DRAW * v0[1])
        arrow1.set_position((p0[0], p0[1]))
        arrow2.xy = (p1[0] + TANGENT_DRAW * v1[0], p1[1] + TANGENT_DRAW * v1[1])
        arrow2.set_position((p1[0], p1[1]))

        pad = 0.4
        xs = np.concatenate([xyz_d[:, 0], [p0[0], p1[0]]])
        ys = np.concatenate([xyz_d[:, 1], [p0[1], p1[1]]])
        ax_curve.set_xlim(xs.min() - pad, xs.max() + pad)
        ax_curve.set_ylim(ys.min() - pad, ys.max() + pad)
        title_curve.set_text(
            f"σ1={s0:.2f}, σ2={s1:.2f}, L≈{L:.3f}  |  "
            f"κ∈[{kappa_s.min():.3g}, {kappa_s.max():.3g}]"
        )

        heights = np.array([vals[m] for m in METHODS])
        for bar, h, txt, m in zip(bars, heights, bar_texts, METHODS):
            bar.set_height(h)
            txt.set_position((bar.get_x() + bar.get_width() / 2, h))
            txt.set_text(f"{h:.4g}")
        ymax = max(float(heights.max()) * 1.15, 1e-6)
        ax_bar.set_ylim(0.0, ymax)
        title_bar.set_text(f"Five costs (r={RADIUS})")

        base = vals["k_bend"] + 1e-15
        lines = [
            f"{m:14s} K={vals[m]:10.6f}  /k_bend={vals[m] / base:7.4f}" for m in METHODS
        ]
        info.set_text("  |  ".join(lines))
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(redraw)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
