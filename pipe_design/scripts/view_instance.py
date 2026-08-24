#!/usr/bin/env python3
"""Browse pipe_design benchmark YAML instances (PyVistaQt + PyQt5).

Usage (conda env deploy_vllm):
  python view_instance.py
  python view_instance.py /path/to/folder
  python view_instance.py /path/to/instance_0.yaml --no-tube

Depends (use ONE Qt stack — conda recommended):
  conda install -c conda-forge pyqt
  pip install pyvista pyvistaqt pyyaml
  # do NOT pip install PyQt5 / PyQt5-Qt5 alongside conda pyqt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

# Conda pyqt plugins live under $CONDA_PREFIX/plugins; point Qt there if needed.
_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    _plat = Path(_conda) / "plugins" / "platforms"
    if _plat.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_plat))

import numpy as np
import pyvista as pv
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonDataModel import vtkPlane

COLORS = [
    (0.90, 0.30, 0.25),
    (0.20, 0.55, 0.85),
    (0.25, 0.75, 0.40),
    (0.95, 0.65, 0.15),
    (0.65, 0.35, 0.85),
    (0.15, 0.75, 0.75),
    (0.85, 0.40, 0.65),
    (0.55, 0.55, 0.20),
]

OBS_COLOR = "gray"
OBS_EDGE_COLOR = "black"
OBS_OPACITY = 0.40


def cell_centers(cells: list, res: float) -> np.ndarray:
    """Path cells → Nx3 world centers at (c + 0.5) * res."""
    return np.array(
        [[(c[0] + 0.5) * res, (c[1] + 0.5) * res, (c[2] + 0.5) * res] for c in cells],
        dtype=float,
    )


def draw_pipe(
    plotter: Any,
    cells: list,
    radius: float,
    color: tuple[float, float, float],
    res: float,
    use_tube: bool,
) -> list:
    """Draw one witness path as tube or centerline. Returns actors."""
    if len(cells) < 2:
        return []
    line = pv.lines_from_points(cell_centers(cells, res))
    mesh = line if not use_tube else line.tube(radius=radius, n_sides=12)
    return [
        plotter.add_mesh(mesh, color=color, opacity=0.9, smooth_shading=True),
    ]


def draw_terminals(
    plotter: Any,
    terminals: list,
    default_radius: float,
    color: tuple[float, float, float],
) -> list:
    """Draw terminal spheres. Returns actors."""
    actors = []
    for t in terminals:
        pos = t["pos"]
        r = float(t.get("radius", default_radius))
        sphere = pv.Sphere(radius=r * 1.2, center=(pos[0], pos[1], pos[2]))
        actors.append(plotter.add_mesh(sphere, color=color))
    return actors


def draw_obstacles(plotter: Any, obstacles: dict) -> tuple[list, list]:
    """Draw spheres / capsules / AABBs with edges. Returns (actors, solids for slice)."""
    actors: list = []
    solids: list = []
    for s in obstacles.get("spheres") or []:
        c, r = s["center"], float(s["radius"])
        solids.append(pv.Sphere(radius=r, center=(c[0], c[1], c[2])))
    for cap in obstacles.get("capsules") or []:
        a, b, r = cap["a"], cap["b"], float(cap["radius"])
        pts = np.array([[a[0], a[1], a[2]], [b[0], b[1], b[2]]], dtype=float)
        solids.append(pv.lines_from_points(pts).tube(radius=r, n_sides=12))
    for box in obstacles.get("aabbs") or []:
        mn, mx = box["mn"], box["mx"]
        solids.append(pv.Box(bounds=(mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])))
    for mesh in solids:
        actors.append(
            plotter.add_mesh(
                mesh, color=OBS_COLOR, opacity=OBS_OPACITY, smooth_shading=True
            )
        )
        actors.append(
            plotter.add_mesh(
                mesh.extract_feature_edges(),
                color=OBS_EDGE_COLOR,
                line_width=2,
            )
        )
    return actors, solids


def draw_domain(plotter: Any, sx: int, sy: int, sz: int, res: float) -> Any:
    """Full domain wireframe outline."""
    return plotter.add_mesh(
        pv.Box(bounds=(0, sx * res, 0, sy * res, 0, sz * res)).extract_feature_edges(),
        color="black",
        line_width=1,
    )


def draw_shell(
    plotter: Any, sx: int, sy: int, sz: int, shell_m: int, res: float
) -> Any | None:
    """Core planning box inside shell; None if shell unused / too thick."""
    if not (shell_m > 0 and sx > 2 * shell_m and sy > 2 * shell_m and sz > 2 * shell_m):
        return None
    return plotter.add_mesh(
        pv.Box(
            bounds=(
                shell_m * res,
                (sx - shell_m) * res,
                shell_m * res,
                (sy - shell_m) * res,
                shell_m * res,
                (sz - shell_m) * res,
            )
        ).extract_feature_edges(),
        color="dimgray",
        line_width=2,
    )


def obstacle_status_bits(meta: dict, obstacles: dict) -> str:
    """Short obstacle summary for the status bar."""
    otype = meta.get("obstacle_type", "none")
    n_aabb = len(obstacles.get("aabbs") or [])
    n_sph = len(obstacles.get("spheres") or [])
    n_cap = len(obstacles.get("capsules") or [])
    bits = [f"obs={otype}", f"aabb={n_aabb}"]
    if n_sph:
        bits.append(f"sph={n_sph}")
    if n_cap:
        bits.append(f"cap={n_cap}")
    obst = meta.get("obstacle") or {}
    if "density_target" in obst:
        bits.append(f"ρ={float(obst['density_target']):.2f}")
    if "dilate_steps" in meta:
        bits.append(f"k={meta['dilate_steps']}")
    return "  ".join(bits)


class MainWindow(QMainWindow):
    def __init__(self, use_tube: bool = True) -> None:
        super().__init__()
        self.use_tube = use_tube
        self.files: list[Path] = []
        self.index = 0
        self.setWindowTitle("pipe_design benchmark viewer")
        self.resize(1100, 750)

        file_menu = self.menuBar().addMenu("&File")  # type: ignore
        open_act = file_menu.addAction("Open &Folder…")  # type: ignore
        open_act.triggered.connect(self.open_folder)  # type: ignore
        quit_act = file_menu.addAction("&Quit")  # type: ignore
        quit_act.triggered.connect(self.close)  # type: ignore

        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_instance)
        self.next_btn.clicked.connect(self.next_instance)

        self.shell_cb = QCheckBox("Shell")
        self.shell_cb.setChecked(True)
        self.shell_cb.setEnabled(False)
        self.shell_cb.stateChanged.connect(self._on_shell_toggled)

        self.obs_cb = QCheckBox("Obstacles")
        self.obs_cb.setChecked(True)
        self.obs_cb.stateChanged.connect(self._on_obstacles_toggled)

        self.clip_cb = QCheckBox("Clip")
        self.clip_cb.setChecked(False)
        self.clip_cb.stateChanged.connect(self._on_clip_toggled)

        self.slice_cb = QCheckBox("Slice")
        self.slice_cb.setChecked(False)
        self.slice_cb.stateChanged.connect(self._on_slice_toggled)

        self.clip_axis = QComboBox()
        self.clip_axis.addItems(["X", "Y", "Z"])
        self.clip_axis.setFixedWidth(48)
        self.clip_axis.currentIndexChanged.connect(self._sync_cut)

        self.clip_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.clip_slider.setRange(0, 1000)
        self.clip_slider.setValue(500)
        self.clip_slider.setFixedWidth(140)
        self.clip_slider.valueChanged.connect(self._sync_cut)

        self.clip_pos_lbl = QLabel("—")
        self.clip_pos_lbl.setMinimumWidth(56)

        self._shell_actor = None
        self._obstacle_actors: list = []
        self._clip_actors: list = []
        self._clip_extents = (0.0, 0.0, 0.0)  # world size along x,y,z
        self._clip_plane = vtkPlane()
        self._slice_mesh: pv.PolyData | None = None

        self.status = QLabel("File → Open Folder…")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore

        top = QHBoxLayout()
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        top.addWidget(self.shell_cb)
        top.addWidget(self.obs_cb)
        top.addWidget(self.clip_cb)
        top.addWidget(self.slice_cb)
        top.addWidget(self.clip_axis)
        top.addWidget(self.clip_slider)
        top.addWidget(self.clip_pos_lbl)
        top.addWidget(self.status, stretch=1)

        self.plotter = QtInteractor(self)  # type: ignore
        self.group_box = QVBoxLayout()
        self.group_box.addWidget(QLabel("Groups"))
        self.group_box.addStretch(1)
        side = QWidget()
        side.setLayout(self.group_box)
        side.setFixedWidth(140)

        mid = QHBoxLayout()
        mid.addWidget(self.plotter.interactor, stretch=1)
        mid.addWidget(side)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(mid, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._set_nav_enabled(False)

    def _set_nav_enabled(self, on: bool) -> None:
        self.prev_btn.setEnabled(on)
        self.next_btn.setEnabled(on)

    def _set_actors_visible(self, actors: list, visible: bool) -> None:
        for a in actors:
            a.SetVisibility(visible)
        self.plotter.render()

    def _on_shell_toggled(self, state: int) -> None:
        if self._shell_actor is None:
            return
        self._shell_actor.SetVisibility(state == Qt.Checked)  # type: ignore
        self.plotter.render()

    def _on_obstacles_toggled(self, state: int) -> None:
        self._set_actors_visible(self._obstacle_actors, state == Qt.Checked)  # type: ignore

    def _on_clip_toggled(self, state: int) -> None:
        if state == Qt.Checked and self.slice_cb.isChecked():  # type: ignore
            self.slice_cb.blockSignals(True)
            self.slice_cb.setChecked(False)
            self.slice_cb.blockSignals(False)
        self._sync_cut()

    def _on_slice_toggled(self, state: int) -> None:
        if state == Qt.Checked and self.clip_cb.isChecked():  # type: ignore
            self.clip_cb.blockSignals(True)
            self.clip_cb.setChecked(False)
            self.clip_cb.blockSignals(False)
        self._sync_cut()

    def _sync_cut(self, *_args: object) -> None:
        self._sync_clip()
        self._sync_slice()

    def _plane_from_controls(self) -> tuple[list[float], list[float]]:
        """Shared origin/normal from axis combo + slider; updates position label."""
        ex, ey, ez = self._clip_extents
        axis = self.clip_axis.currentIndex()  # 0=X,1=Y,2=Z
        t = self.clip_slider.value() / 1000.0
        sizes = (ex, ey, ez)
        origin = [ex * 0.5, ey * 0.5, ez * 0.5]
        origin[axis] = t * sizes[axis]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        self.clip_pos_lbl.setText(f"{'XYZ'[axis]}={origin[axis]:.1f}")
        return origin, normal

    def _sync_slice(self, *_args: object) -> None:
        """One slice on merged solids (vtkCutter via PolyData.slice); shares axis/slider."""
        origin, normal = self._plane_from_controls()
        # Drop previous slice actor (do not add_mesh(empty) — PyVista rejects 0-point meshes).
        self.plotter.remove_actor("_slice", reset_camera=False)

        if (
            not self.slice_cb.isChecked()
            or self._slice_mesh is None
            or self._slice_mesh.n_points == 0
        ):
            self.plotter.render()
            return

        slc = self._slice_mesh.slice(normal=tuple(normal), origin=tuple(origin))
        if slc.n_points > 0:
            self.plotter.add_mesh(
                slc,
                color="crimson",
                line_width=3,
                name="_slice",
                reset_camera=False,
            )
        self.plotter.render()

    def _sync_clip(self, *_args: object) -> None:
        """Axis-aligned clip plane on all content actors (keep +normal side)."""
        origin, normal = self._plane_from_controls()
        self._clip_plane.SetOrigin(*origin)
        self._clip_plane.SetNormal(*normal)

        enabled = self.clip_cb.isChecked()
        for a in self._clip_actors:
            mapper = a.GetMapper()
            if mapper is None:
                continue
            mapper.RemoveAllClippingPlanes()
            if enabled:
                mapper.AddClippingPlane(self._clip_plane)
        self.plotter.render()

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select instance folder")
        if not d:
            return
        self.set_folder(Path(d))

    def set_folder(self, folder: Path, select: Path | None = None) -> None:
        files = sorted(folder.rglob("*.yaml"))
        if not files:
            self.status.setText(f"No .yaml under {folder}")
            self.files = []
            self._set_nav_enabled(False)
            return
        self.files = files
        self.index = 0
        if select is not None:
            select = select.resolve()
            for i, p in enumerate(files):
                if p.resolve() == select:
                    self.index = i
                    break
        self._set_nav_enabled(True)
        self.show_current()

    def prev_instance(self) -> None:
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self.show_current()

    def next_instance(self) -> None:
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self.show_current()

    def _clear_group_checkboxes(self) -> None:
        while self.group_box.count() > 2:
            item = self.group_box.takeAt(1)
            w = item.widget()  # type: ignore
            if w is not None:
                w.deleteLater()

    def show_current(self) -> None:
        path = self.files[self.index]
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.plotter.clear()
        self.plotter.add_axes()
        self._clear_group_checkboxes()
        self._shell_actor = None
        self._obstacle_actors = []
        self._clip_actors = []
        self._slice_mesh = None
        slice_solids: list = []

        grid = data["grid"]
        sx, sy, sz = int(grid["size_x"]), int(grid["size_y"]), int(grid["size_z"])
        meta = data.get("meta") or {}
        radius = float(meta.get("radius", 0.5))
        res = float(grid.get("resolution", meta.get("resolution", 1.0)))
        shell_margin = float(meta.get("shell_margin", 0.0))
        if "shell_cells" in meta:
            shell_m = int(meta["shell_cells"])
        else:
            shell_m = int(math.ceil((shell_margin + radius) / res))
        groups = data.get("groups") or []
        solution = {g["id"]: g for g in (data.get("solution") or [])}
        obstacles = data.get("obstacles") or {}
        self._clip_extents = (sx * res, sy * res, sz * res)

        self._clip_actors.append(draw_domain(self.plotter, sx, sy, sz, res))

        self._shell_actor = draw_shell(self.plotter, sx, sy, sz, shell_m, res)
        has_shell = self._shell_actor is not None
        self.shell_cb.setEnabled(has_shell)
        if has_shell:
            self._shell_actor.SetVisibility(self.shell_cb.isChecked())  # type: ignore
            self._clip_actors.append(self._shell_actor)

        self._obstacle_actors, obs_solids = draw_obstacles(self.plotter, obstacles)
        slice_solids.extend(obs_solids)
        self._set_actors_visible(self._obstacle_actors, self.obs_cb.isChecked())
        self._clip_actors.extend(self._obstacle_actors)

        for gi, g in enumerate(groups):
            gid = g["id"]
            color = COLORS[gi % len(COLORS)]
            actors: list = []

            for path_node in solution.get(gid, {}).get("paths") or []:
                cells = path_node.get("cells") or []
                if len(cells) >= 2 and self.use_tube:
                    tube = pv.lines_from_points(cell_centers(cells, res)).tube(
                        radius=radius, n_sides=12
                    )
                    slice_solids.append(tube)
                actors.extend(
                    draw_pipe(
                        self.plotter, cells, radius, color, res, self.use_tube
                    )
                )

            actors.extend(
                draw_terminals(
                    self.plotter, g.get("terminals") or [], radius, color
                )
            )
            self._clip_actors.extend(actors)

            cb = QCheckBox(str(gid))
            cb.setChecked(True)

            def on_group(state, acts=actors):
                vis = state == Qt.Checked  # type: ignore
                for a in acts:
                    a.SetVisibility(vis)
                self.plotter.render()

            cb.stateChanged.connect(on_group)
            self.group_box.insertWidget(self.group_box.count() - 1, cb)

        if slice_solids:
            self._slice_mesh = pv.merge(slice_solids)

        self.plotter.show_bounds(
            bounds=(0, sx * res, 0, sy * res, 0, sz * res),
            location="outer",
            color="gray",
        )
        self.status.setText(
            f"[{self.index + 1}/{len(self.files)}] {path}  |  "
            f"{meta.get('generator', '?')}  groups={len(groups)}  "
            f"r={radius}  shell_m={shell_m}  |  "
            f"{obstacle_status_bits(meta, obstacles)}"
        )
        self.setWindowTitle(f"bench: {path.name}")
        self._sync_cut()
        self.plotter.reset_camera()
        self.plotter.render()


def main() -> int:
    ap = argparse.ArgumentParser(description="Browse pipe_design benchmark YAML")
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help="optional folder or yaml file to open at start",
    )
    ap.add_argument("--no-tube", action="store_true", help="centerlines only")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(use_tube=not args.no_tube)
    win.show()

    if args.path:
        p = Path(args.path)
        if p.is_file():
            win.set_folder(p.parent, select=p)
        elif p.is_dir():
            win.set_folder(p)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
