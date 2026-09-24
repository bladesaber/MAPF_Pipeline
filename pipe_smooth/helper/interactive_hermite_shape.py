"""Investigate cubic Hermite curve shape with planar endpoints on y=0.

p0 = [0, 0], p1 = [x, 0]; sliders for x, tangent angles, and magnitudes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pipe_smooth.geometry.hermite import eval_hermite_ts

fig = plt.figure(figsize=(10, 6.5))
ax = fig.add_axes([0.08, 0.42, 0.88, 0.52])  # type: ignore

sx = Slider(fig.add_axes([0.12, 0.30, 0.78, 0.03]), "x = p1.x", 0.1, 10.0, valinit=2.0, valfmt="%.2f")  # type: ignore
sa0 = Slider(fig.add_axes([0.12, 0.24, 0.78, 0.03]), "θ0 (deg)", 0.0, 180.0, valinit=30.0, valfmt="%.1f")  # type: ignore
sa1 = Slider(fig.add_axes([0.12, 0.18, 0.78, 0.03]), "θ1 (deg)", 0.0, 180.0, valinit=150.0, valfmt="%.1f")  # type: ignore
# log10 magnitude: -4 .. 4  →  1e-4 .. 1e4
sm0 = Slider(fig.add_axes([0.12, 0.12, 0.78, 0.03]), "log10||v0||", -4.0, 4.0, valinit=0.3, valfmt="%.2f")  # type: ignore
sm1 = Slider(fig.add_axes([0.12, 0.06, 0.78, 0.03]), "log10||v1||", -4.0, 4.0, valinit=0.3, valfmt="%.2f")  # type: ignore

(line,) = ax.plot([], [], "k-", lw=2)
(arrow0,) = ax.plot([], [], "C2-", lw=1.5)
(arrow1,) = ax.plot([], [], "C3-", lw=1.5)
scat = ax.scatter([0.0, 2.0], [0.0, 0.0], c=["C2", "C3"], s=60, zorder=5)
info = fig.text(0.08, 0.01, "", family="monospace", fontsize=9, va="bottom")
ts = np.linspace(0.0, 1.0, 200)


def redraw(_=None) -> None:
    x = float(sx.val)
    a0, a1 = np.deg2rad(sa0.val), np.deg2rad(sa1.val)
    s0, s1 = 10.0 ** sm0.val, 10.0 ** sm1.val
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([x, 0.0, 0.0])
    v0 = s0 * np.array([np.cos(a0), np.sin(a0), 0.0])
    v1 = s1 * np.array([np.cos(a1), np.sin(a1), 0.0])

    xyz = eval_hermite_ts(p0, v0, p1, v1, ts)
    line.set_data(xyz[:, 0], xyz[:, 1])
    # tangent arrows (scaled for display)
    scale = 0.25 * max(x, 1.0)
    u0, u1 = v0[:2] / s0, v1[:2] / s1
    arrow0.set_data([0, scale * u0[0]], [0, scale * u0[1]])
    arrow1.set_data([x, x + scale * u1[0]], [0, scale * u1[1]])
    scat.set_offsets(np.array([[0.0, 0.0], [x, 0.0]]))

    pad = 0.4 * max(x, 1.0)
    ax.set_xlim(float(xyz[:, 0].min()) - pad, float(xyz[:, 0].max()) + pad)
    ax.set_ylim(float(xyz[:, 1].min()) - pad, float(xyz[:, 1].max()) + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_title("cubic Hermite  |  p0=[0,0], p1=[x,0]")
    info.set_text(f"||v0||={s0:.4g}  ||v1||={s1:.4g}  |  θ0={sa0.val:.1f}°  θ1={sa1.val:.1f}°  |  x={x:.2f}")
    fig.canvas.draw_idle()


for s in (sx, sa0, sa1, sm0, sm1):
    s.on_changed(redraw)
redraw()
plt.show()
