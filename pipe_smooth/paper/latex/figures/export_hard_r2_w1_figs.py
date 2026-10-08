"""Temporary: export paper figures from hard_r2_w1/adam/capsule runs.

Usage (conda env deploy_vllm):
  cd MAPF_Pipeline
  conda activate deploy_vllm
  python pipe_smooth/paper/latex/figures/export_hard_r2_w1_figs.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipe_smooth.collision.obstacles import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.pipeline import load_obstacles_file

RUN_ROOT = _ROOT / "pipe_smooth/runs/hard_r2_w1/adam/capsule"
DATA_ROOT = _ROOT / "pipe_smooth/data/hard_r2"
OUT_DIR = _HERE

# instances: showcase + near-median κr
INSTANCES = ("instance_0", "instance_1")
# single-group paper figure (clearest junctions on instance_0)
SHOWCASE_GROUP = "g2"
GROUP_COLORS = [
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#2c3e50",
    "#f39c12",
]


def _run_dir(name: str) -> Path:
    return RUN_ROOT / name


def _env_path(name: str) -> Path:
    # instance_0 -> instance_0.yaml
    return DATA_ROOT / f"{name}.yaml"


def load_coarse_polylines(env_path: Path) -> list[tuple[str, np.ndarray, float]]:
    """Return list of (group_id, xyz (N,3), radius) from discrete paths."""
    env = yaml.safe_load(env_path.read_text())
    out = []
    for g in env.get("paths") or []:
        gid = str(g["id"])
        for p in g.get("paths") or []:
            xyz = np.asarray(p["xyz"], dtype=float)
            r = float(p.get("radius", env.get("meta", {}).get("radius", 2.0)))
            if len(xyz) >= 2:
                out.append((gid, xyz, r))
    return out


def load_opt_polylines(run_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return list of (group_id, xyz (N,3), radius (N,)) from OptimizeGraph dumps."""
    out = []
    for path in sorted(run_dir.glob("graph_*.json")):
        g = OptimizeGraph.load(path)
        _, xyz, radius = g.refresh(backend="numpy", store=True)
        xyz = np.asarray(xyz, dtype=float)
        radius = np.asarray(radius, dtype=float)
        for s in range(xyz.shape[0]):
            out.append((g.id, xyz[s], radius[s]))
    return out


def _add_obstacles(plotter: pv.Plotter, obstacles, opacity: float = 0.12) -> None:
    for o in obstacles:
        if isinstance(o, SphereObs):
            mesh = pv.Sphere(radius=float(o.r), center=np.asarray(o.c, float))
        elif isinstance(o, CapsuleObs):
            a, b = np.asarray(o.a, float), np.asarray(o.b, float)
            mesh = pv.Line(a, b).tube(radius=float(o.r))
        elif isinstance(o, AabbObs):
            mn = np.asarray(o.mn, float)
            mx = np.asarray(o.mx, float)
            center = 0.5 * (mn + mx)
            mesh = pv.Cube(center=center, x_length=mx[0] - mn[0], y_length=mx[1] - mn[1], z_length=mx[2] - mn[2])
        else:
            continue
        plotter.add_mesh(mesh, color="#95a5a6", opacity=opacity, smooth_shading=True)


def _add_tubes(plotter: pv.Plotter, curves, *, tube_scale: float = 0.85) -> None:
    gids = sorted({gid for gid, *_ in curves})
    cmap = {gid: GROUP_COLORS[i % len(GROUP_COLORS)] for i, gid in enumerate(gids)}
    for gid, xyz, rad in curves:
        xyz = np.asarray(xyz, float)
        if xyz.ndim == 1:
            continue
        if len(xyz) < 2:
            continue
        line = pv.lines_from_points(xyz, close=False)
        r = np.asarray(rad, float)
        radius = float(r) if (r.ndim == 0 or r.size == 1) else float(np.mean(r))
        mesh = line.tube(radius=radius * tube_scale, n_sides=36, capping=True)
        plotter.add_mesh(
            mesh,
            color=cmap[gid],
            smooth_shading=True,
            specular=0.35,
            ambient=0.18,
            diffuse=0.75,
        )


def _curves_bbox_center(curves) -> np.ndarray:
    pts = []
    for _, xyz, _ in curves:
        xyz = np.asarray(xyz, float)
        if xyz.ndim == 2 and len(xyz) >= 1:
            pts.append(xyz)
    if not pts:
        return np.array([24.0, 24.0, 24.0], float)
    all_xyz = np.concatenate(pts, axis=0)
    return 0.5 * (all_xyz.min(axis=0) + all_xyz.max(axis=0))


def _curves_extent(curves) -> float:
    pts = []
    for _, xyz, _ in curves:
        xyz = np.asarray(xyz, float)
        if xyz.ndim == 2 and len(xyz) >= 1:
            pts.append(xyz)
    if not pts:
        return 70.0
    all_xyz = np.concatenate(pts, axis=0)
    return float(np.linalg.norm(all_xyz.max(axis=0) - all_xyz.min(axis=0)))


def _camera_iso(plotter: pv.Plotter, center=(24.0, 24.0, 24.0), dist: float | None = None) -> None:
    c = np.asarray(center, float)
    d = float(dist) if dist is not None else 70.0
    # isometric-ish offset scaled to content size
    offset = np.array([1.0, -1.0, 0.78], float)
    offset = offset / np.linalg.norm(offset) * d
    plotter.camera_position = [c + offset, c, (0.0, 0.0, 1.0)]


def _filter_group(curves, gid: str):
    return [(g, xyz, r) for g, xyz, r in curves if str(g) == str(gid)]


def render_scene(
    out_png: Path,
    curves,
    obstacles,
    *,
    window_size=(3840, 3000),
    frame_curves=None,
    show_obstacles: bool = False,
    camera_scale: float = 1.35,
) -> None:
    """High-res offscreen shot; captions live in LaTeX, not on the image.

    If frame_curves is set, camera is framed on that subset (e.g. one group).
    Obstacles are off by default (packing reads clearer without them).
    camera_scale > 1 pulls the camera farther from the content bbox.
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    if show_obstacles and obstacles:
        _add_obstacles(plotter, obstacles, opacity=0.10)
    _add_tubes(plotter, curves)
    plotter.add_axes(line_width=4, labels_off=False)
    frame = frame_curves if frame_curves is not None else curves
    center = _curves_bbox_center(frame)
    extent = _curves_extent(frame)
    _camera_iso(plotter, center=center, dist=max(55.0, float(camera_scale) * extent))
    plotter.show(screenshot=str(out_png), return_img=False)
    plotter.close()
    print(f"wrote {out_png} ({window_size[0]}x{window_size[1]})")


def export_j_curve(run_dir: Path, out_pdf: Path) -> None:
    rec = pickle.load((run_dir / "recorder.pkl").open("rb"))
    steps = [h["step"] for h in rec.history]
    J = [h["J"] for h in rec.history]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(steps, J, color="#2c3e50", lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$J$")
    ax.set_title(f"{run_dir.name}: objective")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def export_instance(name: str, *, showcase_gid: str = SHOWCASE_GROUP) -> None:
    run_dir = _run_dir(name)
    env_path = _env_path(name)
    obs = load_obstacles_file(run_dir / "obstacles.json", clear_kind="capsule")
    coarse = load_coarse_polylines(env_path)
    coarse_curves = [(gid, xyz, r) for gid, xyz, r in coarse]
    opt_curves = load_opt_polylines(run_dir)

    # all-groups overview (no obstacles; farther camera for full packing)
    before_png = OUT_DIR / f"hard_r2_w1_{name}_before.png"
    after_png = OUT_DIR / f"hard_r2_w1_{name}_after.png"
    render_scene(before_png, coarse_curves, obs, show_obstacles=False, camera_scale=1.45)
    render_scene(after_png, opt_curves, obs, show_obstacles=False, camera_scale=1.45)

    # single-group compare: no obstacles; slightly closer than overview
    g_before = _filter_group(coarse_curves, showcase_gid)
    g_after = _filter_group(opt_curves, showcase_gid)
    if not g_before or not g_after:
        raise RuntimeError(f"showcase group {showcase_gid!r} missing in {name}")
    g_before_png = OUT_DIR / f"hard_r2_w1_{name}_{showcase_gid}_before.png"
    g_after_png = OUT_DIR / f"hard_r2_w1_{name}_{showcase_gid}_after.png"
    render_scene(
        g_before_png,
        g_before,
        obs,
        frame_curves=g_before,
        show_obstacles=False,
        camera_scale=1.30,
    )
    render_scene(
        g_after_png,
        g_after,
        obs,
        frame_curves=g_after,
        show_obstacles=False,
        camera_scale=1.30,
    )

    export_j_curve(run_dir, OUT_DIR / f"hard_r2_w1_{name}_J.pdf")


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    for name in INSTANCES:
        export_instance(name)
    # also dump a tiny manifest for latex include
    manifest = {
        "run_root": str(RUN_ROOT),
        "instances": list(INSTANCES),
        "showcase_group": SHOWCASE_GROUP,
        "files": sorted(p.name for p in OUT_DIR.glob("hard_r2_w1_*")),
    }
    (OUT_DIR / "hard_r2_w1_export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done", manifest["files"])


if __name__ == "__main__":
    main()
