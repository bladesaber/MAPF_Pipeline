#!/usr/bin/env python3
"""Plot a 2D heuristic heatmap from pipe_dump_heuristic JSON.

Input schema (main_heuristic.cpp → heuristic_q{id}.json under output_dir):
  instance_id, map_id, query_id, heuristic, size [sx,sy,sz],
  goals [[x,y,z], …], cells [{x,y,z,h}, …]  (free cells only)

Blocked / missing cells are blank. Goals marked with ×.
For size_z > 1, pass --z to pick a slice (default 0).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_heuristic(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("size", "cells", "goals", "query_id", "heuristic"):
        if key not in data:
            raise SystemExit(f"{path}: missing field {key!r}")
    if len(data["size"]) < 2:
        raise SystemExit(f"{path}: size must be [sx,sy] or [sx,sy,sz]")
    return data


def grid_from_cells(
    size: list[int], cells: list[dict], z: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, mask) with shape (sy, sx); mask True where h is defined."""
    sx, sy = int(size[0]), int(size[1])
    H = np.full((sy, sx), np.nan, dtype=np.float64)
    for c in cells:
        if int(c.get("z", 0)) != z:
            continue
        x, y = int(c["x"]), int(c["y"])
        if 0 <= x < sx and 0 <= y < sy:
            h = float(c["h"])
            if math.isfinite(h):
                H[y, x] = h
    mask = np.isfinite(H)
    return H, mask


def plot_heatmap(data: dict, z: int, out: Path | None, show: bool) -> None:
    size = data["size"]
    sz = int(size[2]) if len(size) >= 3 else 1
    if z < 0 or z >= sz:
        raise SystemExit(f"--z {z} out of range for size_z={sz}")

    H, mask = grid_from_cells(size, data["cells"], z)
    sx, sy = int(size[0]), int(size[1])

    fig, ax = plt.subplots(figsize=(8, 7))
    # Obstacles / non-free: light gray underlay
    ax.imshow(
        np.ones((sy, sx)),
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=(-0.5, sx - 0.5, sy - 0.5, -0.5),
        interpolation="nearest",
    )
    if mask.any():
        masked = np.ma.array(H, mask=~mask)
        im = ax.imshow(
            masked,
            cmap="viridis",
            origin="upper",
            extent=(-0.5, sx - 0.5, sy - 0.5, -0.5),
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("h")
    else:
        ax.text(0.5, 0.5, "no free cells", transform=ax.transAxes, ha="center")

    gx, gy = [], []
    for g in data["goals"]:
        # goals are [x,y,z] (C++ dump); skip other z-slices
        if len(g) >= 3 and int(g[2]) != z:
            continue
        gx.append(int(g[0]))
        gy.append(int(g[1]))
    if gx:
        ax.scatter(
            gx,
            gy,
            s=80,
            c="red",
            marker="x",
            linewidths=2,
            label="goal",
            zorder=5,
        )
        ax.legend(loc="upper right")

    map_id = data.get("map_id", "?")
    qid = data["query_id"]
    name = data["heuristic"]
    title = f"{map_id}  q{qid}  ·  {name}"
    if sz > 1:
        title += f"  ·  z={z}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, sx - 0.5)
    ax.set_ylim(sy - 0.5, -0.5)
    ax.set_aspect("equal")
    fig.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    if show or out is None:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="heuristic_q{id}.json from pipe_dump_heuristic",
    )
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="output image (png/pdf/…); default = show interactively",
    )
    ap.add_argument(
        "--z",
        type=int,
        default=0,
        help="z-slice for size_z>1 (default 0)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="also open a window when --out is set",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"not found: {args.input}")

    data = load_heuristic(args.input)
    plot_heatmap(data, z=args.z, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
