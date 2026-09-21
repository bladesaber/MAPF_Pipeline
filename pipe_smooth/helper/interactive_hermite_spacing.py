"""Investigate ||v0|| / ||v1|| vs adjacent sample spacing on a Hermite curve.

Fixed: endpoints, unit tangent directions, parameter samples `ts`.
Sliders: only start/end tangent magnitudes σ0, σ1 (v = σ · û).
Shows Δs_i = ||x_{i+1} - x_i|| between consecutive waypoints.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pipe_smooth.geometry.hermite import eval_hermite_ts, hermite_length, unit

# ---------------------------------------------------------------------------
# Fixed geometry & sampling (edit here if needed)
# ---------------------------------------------------------------------------
POS0 = np.array([0.0, 0.0, 0.0])
POS1 = np.array([2.0, 0.0, 0.0])
DIR0 = unit(np.array([1.0, 0.4, 0.0]))   # fixed start direction
DIR1 = unit(np.array([1.0, -0.4, 0.0]))  # fixed end direction
TS = np.linspace(0.0, 1.0, 17)          # fixed parameter samples

SIGMA0_INIT = 2.0
SIGMA1_INIT = 2.0
SIGMA_MIN, SIGMA_MAX = 0.05, 6.0
# ---------------------------------------------------------------------------


def main() -> None:
    fig = plt.figure(figsize=(11, 6.5))
    ax_curve = fig.add_axes([0.07, 0.28, 0.50, 0.62])  # type: ignore
    ax_ds = fig.add_axes([0.64, 0.28, 0.32, 0.62])  # type: ignore

    ax_s0 = fig.add_axes([0.12, 0.14, 0.78, 0.04])  # type: ignore
    ax_s1 = fig.add_axes([0.12, 0.07, 0.78, 0.04])  # type: ignore
    s_sigma0 = Slider(ax_s0, "σ0 = ||v0||", SIGMA_MIN, SIGMA_MAX, valinit=SIGMA0_INIT, valfmt="%.2f")
    s_sigma1 = Slider(ax_s1, "σ1 = ||v1||", SIGMA_MIN, SIGMA_MAX, valinit=SIGMA1_INIT, valfmt="%.2f")

    ts_dense = np.linspace(0.0, 1.0, 200)
    mid_t = 0.5 * (TS[:-1] + TS[1:])

    (line_dense,) = ax_curve.plot([], [], "k-", lw=1.8, alpha=0.55, label="Hermite γ(t)")
    (line_samp,) = ax_curve.plot([], [], "o-", color="C0", lw=1.2, ms=6, label="waypoints @ fixed ts")
    scat_ends = ax_curve.scatter(
        [POS0[0], POS1[0]], [POS0[1], POS1[1]], c=["C2", "C3"], s=55, zorder=5
    )
    title_c = ax_curve.set_title("")
    ax_curve.set_aspect("equal", adjustable="datalim")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=8, loc="best")
    ax_curve.set_xlabel("x")
    ax_curve.set_ylabel("y")

    (stem_ds,) = ax_ds.plot([], [], "o-", color="C1", lw=1.6, ms=5, label=r"$\Delta s_i$")
    mean_line = ax_ds.axhline(0.0, color="0.4", ls="--", lw=1, label="mean Δs")
    title_d = ax_ds.set_title(r"Adjacent spacing $\Delta s_i=\|x_{i+1}-x_i\|$")
    ax_ds.set_xlabel(r"midpoint $t$ of segment $i$")
    ax_ds.set_ylabel(r"$\Delta s_i$")
    ax_ds.grid(True, alpha=0.3)
    ax_ds.legend(fontsize=8, loc="best")

    info = fig.text(0.07, 0.01, "", family="monospace", fontsize=8, va="bottom")

    def redraw(_=None) -> None:
        s0 = float(s_sigma0.val)
        s1 = float(s_sigma1.val)
        v0 = s0 * DIR0
        v1 = s1 * DIR1

        xyz = eval_hermite_ts(POS0, v0, POS1, v1, TS)
        dense = eval_hermite_ts(POS0, v0, POS1, v1, ts_dense)
        ds = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
        L_poly = float(np.sum(ds))
        L_gl = float(hermite_length(POS0, v0, POS1, v1))

        line_dense.set_data(dense[:, 0], dense[:, 1])
        line_samp.set_data(xyz[:, 0], xyz[:, 1])
        scat_ends.set_offsets(np.array([[POS0[0], POS0[1]], [POS1[0], POS1[1]]]))

        pad = 0.35
        ax_curve.set_xlim(float(dense[:, 0].min()) - pad, float(dense[:, 0].max()) + pad)
        ax_curve.set_ylim(float(dense[:, 1].min()) - pad, float(dense[:, 1].max()) + pad)
        title_c.set_text(
            rf"fixed ts (N={len(TS)}), dirs fixed  |  σ0={s0:.2f}, σ1={s1:.2f}"
        )

        stem_ds.set_data(mid_t, ds)
        mean_line.set_ydata([float(ds.mean()), float(ds.mean())])
        ax_ds.set_xlim(0.0, 1.0)
        ymax = max(float(ds.max()) * 1.15, 1e-6)
        ax_ds.set_ylim(0.0, ymax)
        title_d.set_text(
            rf"$\Delta s$: min={ds.min():.3g}, max={ds.max():.3g}, "
            rf"max/min={ds.max() / max(ds.min(), 1e-15):.2f}"
        )

        info.set_text(
            f"ΣΔs (polyline)={L_poly:.4f}  |  ∫||γ'||dt (GL4)={L_gl:.4f}  |  "
            f"Δs = [{', '.join(f'{d:.3f}' for d in ds)}]"
        )
        fig.canvas.draw_idle()

    s_sigma0.on_changed(redraw)
    s_sigma1.on_changed(redraw)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
