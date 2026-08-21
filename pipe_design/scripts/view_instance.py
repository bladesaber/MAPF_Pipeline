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
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

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
        self._shell_actor = None
        self.status = QLabel("File → Open Folder…")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore

        top = QHBoxLayout()
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        top.addWidget(self.shell_cb)
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

    def _on_shell_toggled(self, state: int) -> None:
        if self._shell_actor is None:
            return
        self._shell_actor.SetVisibility(state == Qt.Checked)  # type: ignore
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

    def show_current(self) -> None:
        path = self.files[self.index]
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.plotter.clear()
        self.plotter.add_axes()

        # Clear old group checkboxes (keep title at 0, stretch at end).
        while self.group_box.count() > 2:
            item = self.group_box.takeAt(1)
            w = item.widget()  # type: ignore
            if w is not None:
                w.deleteLater()

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

        # Full domain outline.
        self.plotter.add_mesh(
            pv.Box(bounds=(0, sx * res, 0, sy * res, 0, sz * res)).extract_feature_edges(),
            color="black",
            line_width=1,
        )
        # Core planning box (inside shell); empty when shell_m == 0.
        self._shell_actor = None
        has_shell = (
            shell_m > 0 and sx > 2 * shell_m and sy > 2 * shell_m and sz > 2 * shell_m
        )
        self.shell_cb.setEnabled(has_shell)
        if has_shell:
            self._shell_actor = self.plotter.add_mesh(
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
            self._shell_actor.SetVisibility(self.shell_cb.isChecked())

        for gi, g in enumerate(groups):
            gid = g["id"]
            color = COLORS[gi % len(COLORS)]
            actors = []

            for path_node in solution.get(gid, {}).get("paths") or []:
                cells = path_node.get("cells") or []
                if len(cells) < 2:
                    continue
                pts = np.array(
                    [
                        [(c[0] + 0.5) * res, (c[1] + 0.5) * res, (c[2] + 0.5) * res]
                        for c in cells
                    ],
                    dtype=float,
                )
                line = pv.lines_from_points(pts)
                mesh = line if not self.use_tube else line.tube(radius=radius, n_sides=12)
                actors.append(
                    self.plotter.add_mesh(
                        mesh, color=color, opacity=0.9, smooth_shading=True
                    )
                )

            for t in g.get("terminals") or []:
                pos = t["pos"]
                r = float(t.get("radius", radius))
                sphere = pv.Sphere(radius=r * 1.2, center=(pos[0], pos[1], pos[2]))
                actors.append(self.plotter.add_mesh(sphere, color=color))

            cb = QCheckBox(str(gid))
            cb.setChecked(True)

            def on_group(state, acts=actors):
                vis = state == Qt.Checked  # type: ignore
                for a in acts:
                    a.SetVisibility(vis)
                self.plotter.render()

            cb.stateChanged.connect(on_group)
            self.group_box.insertWidget(self.group_box.count() - 1, cb)

        for s in obstacles.get("spheres") or []:
            c, r = s["center"], float(s["radius"])
            self.plotter.add_mesh(
                pv.Sphere(radius=r, center=(c[0], c[1], c[2])),
                color="gray",
                opacity=0.45,
            )
        for cap in obstacles.get("capsules") or []:
            a, b, r = cap["a"], cap["b"], float(cap["radius"])
            pts = np.array([[a[0], a[1], a[2]], [b[0], b[1], b[2]]], dtype=float)
            self.plotter.add_mesh(
                pv.lines_from_points(pts).tube(radius=r, n_sides=12),
                color="gray",
                opacity=0.45,
            )

        self.plotter.show_bounds(
            bounds=(0, sx * res, 0, sy * res, 0, sz * res),
            location="outer",
            color="gray",
        )
        self.status.setText(
            f"[{self.index + 1}/{len(self.files)}] {path}  |  "
            f"{meta.get('generator', '?')}  groups={len(groups)}  "
            f"r={radius}  shell_m={shell_m} (margin={shell_margin})"
        )
        self.setWindowTitle(f"bench: {path.name}")
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
