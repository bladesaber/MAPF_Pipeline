#!/usr/bin/env python3
"""Browse pipe_run_design DesignReport JSONL (PyVistaQt + PyQt5).

Usage (conda env deploy_vllm):
  python view_report.py
  python view_report.py /path/to/design_reports.jsonl
  python view_report.py /path/to/design_reports.jsonl --witness

Loads each report's solver paths; environment from report.instance_path YAML.
Optional witness = YAML constructive `solution` overlay.
View → Stats (or Stats button) opens an auxiliary window for this JSONL batch
(config, success rate, time / cost_ratio five-number summary, cost_ratio hist).
Clip / Slice share axis + slider (same UX as view_instance.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    _plat = Path(_conda) / "plugins" / "platforms"
    if _plat.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_plat))

import numpy as np
import pyvista as pv
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # type: ignore
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
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


def witness_color(
    rgb: tuple[float, float, float], blend: float = 0.55
) -> tuple[float, float, float]:
    """Same hue family as solver, lightened toward white for overlay contrast."""
    return tuple(c + (1.0 - c) * blend for c in rgb)  # type: ignore[return-value]


def cell_centers(cells: list, res: float) -> np.ndarray:
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
    opacity: float = 0.9,
) -> tuple[list, list]:
    tube_actors: list = []
    point_actors: list = []
    if not cells:
        return tube_actors, point_actors
    pts = cell_centers(cells, res)
    if len(cells) >= 2:
        mesh = pv.lines_from_points(pts).tube(radius=radius, n_sides=12)
        tube_actors.append(
            plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True)
        )
    point_actors.append(
        plotter.add_mesh(
            pv.PolyData(pts),
            color=color,
            point_size=10,
            render_points_as_spheres=True,
            opacity=opacity,
        )
    )
    return tube_actors, point_actors


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


def _five(vals: list[float]) -> str:
    """min / 25% / 50% / 75% / max (or — if empty)."""
    if not vals:
        return "—"
    a = np.asarray(vals, dtype=float)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return (
        f"min={q[0]:.3g}  25%={q[1]:.3g}  50%={q[2]:.3g}  "
        f"75%={q[3]:.3g}  max={q[4]:.3g}  (n={len(vals)})"
    )


class StatsWindow(QDialog):
    """Auxiliary stats for the currently loaded JSONL (single-algo batch)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch stats")
        self.resize(520, 640)
        self.setWindowFlags(self.windowFlags() | Qt.Window)  # type: ignore

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(280)

        self.fig = Figure(figsize=(5.0, 2.8), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)

        lay = QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addWidget(self.canvas)

    def refresh(self, reports: list[dict]) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if not reports:
            self.text.setPlainText("No reports loaded.")
            ax.set_title("cost_ratio (no data)")
            self.canvas.draw_idle()
            return

        r0 = reports[0]
        n = len(reports)
        n_ok = sum(1 for r in reports if r.get("success"))
        fails = Counter(
            (r.get("fail_reason") or "(empty)") for r in reports if not r.get("success")
        )

        budget = r0.get("budget_mode", "?")
        if budget == "time":
            budget_s = f"time  max_time_ms={r0.get('max_time_ms')}"
        else:
            budget_s = f"iterations  max_iterations={r0.get('max_iterations')}"

        kw = r0.get("algorithm_kwargs") or {}
        lines = [
            "=== base config (from first report) ===",
            f"algorithm:          {r0.get('algorithm')}",
            f"topology:           {r0.get('topology')}",
            f"low_model:          {r0.get('low_model')}",
            f"heuristic:          {r0.get('heuristic')}",
            f"action_model:       {r0.get('action_model')}",
            f"conflict_analyzer:  {r0.get('conflict_analyzer') or '(none)'}",
            f"algorithm_kwargs:   {kw}",
            f"budget:             {budget_s}",
            "",
            "=== outcome ===",
            f"N:                  {n}",
            f"success:            {n_ok}  ({100.0 * n_ok / n:.1f}%)",
            f"fail:               {n - n_ok}",
        ]
        if fails:
            lines.append("fail_reason:")
            for k, v in sorted(fails.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"  {k}: {v}")

        times_all = [float(r.get("total_time_ms") or 0) for r in reports]
        times_ok = [
            float(r.get("total_time_ms") or 0) for r in reports if r.get("success")
        ]
        ratios = [
            float(r["cost_ratio"])
            for r in reports
            if r.get("success")
            and r.get("witness_compared")
            and r.get("cost_ratio") is not None
        ]

        lines += [
            "",
            "=== total_time_ms (ms) ===",
            f"all runs:     {_five(times_all)}",
            f"success only: {_five(times_ok)}",
            "",
            "=== cost_ratio (success + witness_compared) ===",
            _five(ratios),
        ]
        self.text.setPlainText("\n".join(lines))

        if ratios:
            ax.hist(
                ratios,
                bins=min(20, max(5, len(ratios) // 2)),
                color="#348ABD",
                edgecolor="white",
            )
            ax.set_xlabel("cost_ratio")
            ax.set_ylabel("count")
            ax.set_title(f"cost_ratio (n={len(ratios)})")
        else:
            ax.set_title("cost_ratio (no success+witness data)")
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, use_tube: bool = True, show_witness: bool = False) -> None:
        super().__init__()
        self.use_tube = use_tube
        self.reports: list[dict] = []
        self.index = 0
        self._stats: StatsWindow | None = None
        self.setWindowTitle("pipe_design DesignReport viewer")
        self.resize(1100, 750)

        file_menu = self.menuBar().addMenu("&File")  # type: ignore
        open_act = file_menu.addAction("Open &JSONL…")  # type: ignore
        open_act.triggered.connect(self.open_jsonl)  # type: ignore
        quit_act = file_menu.addAction("&Quit")  # type: ignore
        quit_act.triggered.connect(self.close)  # type: ignore

        view_menu = self.menuBar().addMenu("&View")  # type: ignore
        stats_act = view_menu.addAction("&Stats…")  # type: ignore
        stats_act.triggered.connect(self.open_stats)  # type: ignore

        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_report)
        self.next_btn.clicked.connect(self.next_report)

        self.stats_btn = QPushButton("Stats")
        self.stats_btn.clicked.connect(self.open_stats)

        self.tube_cb = QCheckBox("Tube")
        self.tube_cb.setChecked(use_tube)
        self.tube_cb.stateChanged.connect(self._on_tube_toggled)

        self.obs_cb = QCheckBox("Obstacles")
        self.obs_cb.setChecked(True)
        self.obs_cb.stateChanged.connect(self._on_obstacles_toggled)

        self.witness_cb = QCheckBox("Witness")
        self.witness_cb.setChecked(show_witness)
        self.witness_cb.setToolTip("Overlay YAML constructive solution as reference")
        self.witness_cb.stateChanged.connect(self._on_witness_toggled)

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

        self._obstacle_actors: list = []
        self._clip_actors: list = []
        self._solver_groups: list[tuple[list, list, QCheckBox]] = []
        self._witness_actors: list = []  # (tubes, points) per group
        self._clip_extents = (0.0, 0.0, 0.0)
        self._clip_plane = vtkPlane()
        self._slice_mesh: pv.PolyData | None = None

        self.status = QLabel("File → Open JSONL…")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore

        top = QHBoxLayout()
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        top.addWidget(self.stats_btn)
        top.addWidget(self.tube_cb)
        top.addWidget(self.obs_cb)
        top.addWidget(self.witness_cb)
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

    def open_stats(self) -> None:
        if self._stats is None:
            self._stats = StatsWindow(self)
        self._stats.refresh(self.reports)
        self._stats.show()
        self._stats.raise_()
        self._stats.activateWindow()

    def _set_nav_enabled(self, on: bool) -> None:
        self.prev_btn.setEnabled(on)
        self.next_btn.setEnabled(on)

    def _on_tube_toggled(self, state: int) -> None:
        self.use_tube = state == Qt.Checked  # type: ignore
        use_tube = self.use_tube
        for tube_acts, point_acts, cb in self._solver_groups:
            on = cb.isChecked()
            for a in tube_acts:
                a.SetVisibility(on and use_tube)
            for a in point_acts:
                a.SetVisibility(on and not use_tube)
        self._set_witness_visibility(self.witness_cb.isChecked())
        self.plotter.render()

    def _on_obstacles_toggled(self, state: int) -> None:
        for a in self._obstacle_actors:
            a.SetVisibility(state == Qt.Checked)  # type: ignore
        self.plotter.render()

    def _on_witness_toggled(self, state: int) -> None:
        self._set_witness_visibility(state == Qt.Checked)  # type: ignore
        self.plotter.render()

    def _set_witness_visibility(self, on: bool) -> None:
        use_tube = self.tube_cb.isChecked()
        for tubes, points in self._witness_actors:
            for a in tubes:
                a.SetVisibility(on and use_tube)
            for a in points:
                a.SetVisibility(on and not use_tube)

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
        origin, normal = self._plane_from_controls()
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

    def open_jsonl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select design_reports.jsonl", "", "JSONL (*.jsonl);;All (*)"
        )
        if path:
            self.load_jsonl(Path(path))

    def load_jsonl(self, path: Path) -> None:
        reports: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reports.append(json.loads(line))
        if not reports:
            self.status.setText(f"No reports in {path}")
            self.reports = []
            self._set_nav_enabled(False)
            if self._stats is not None and self._stats.isVisible():
                self._stats.refresh([])
            return
        self.reports = reports
        self.index = 0
        self._set_nav_enabled(True)
        self.setWindowTitle(f"report: {path.name}")
        self.show_current()
        if self._stats is not None and self._stats.isVisible():
            self._stats.refresh(self.reports)

    def prev_report(self) -> None:
        if not self.reports:
            return
        self.index = (self.index - 1) % len(self.reports)
        self.show_current()

    def next_report(self) -> None:
        if not self.reports:
            return
        self.index = (self.index + 1) % len(self.reports)
        self.show_current()

    def _clear_group_checkboxes(self) -> None:
        while self.group_box.count() > 2:
            item = self.group_box.takeAt(1)
            w = item.widget()  # type: ignore
            if w is not None:
                w.deleteLater()

    def show_current(self) -> None:
        report = self.reports[self.index]
        inst_path = Path(report.get("instance_path") or "")
        if not inst_path.is_file():
            self.status.setText(
                f"[{self.index + 1}/{len(self.reports)}] missing instance_path: {inst_path}"
            )
            return

        with open(inst_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.plotter.clear()
        self.plotter.add_axes()
        self._clear_group_checkboxes()
        self._obstacle_actors = []
        self._clip_actors = []
        self._solver_groups = []
        self._witness_actors = []
        self._slice_mesh = None
        slice_solids: list = []

        grid = data["grid"]
        sx, sy, sz = int(grid["size_x"]), int(grid["size_y"]), int(grid["size_z"])
        meta = data.get("meta") or {}
        radius = float(meta.get("radius", 0.5))
        res = float(grid.get("resolution", meta.get("resolution", 1.0)))
        obstacles = data.get("obstacles") or {}
        witness = {g["id"]: g for g in (data.get("solution") or [])}
        self._clip_extents = (sx * res, sy * res, sz * res)

        domain = self.plotter.add_mesh(
            pv.Box(bounds=(0, sx * res, 0, sy * res, 0, sz * res)).extract_feature_edges(),
            color="black",
            line_width=1,
        )
        self._clip_actors.append(domain)

        self._obstacle_actors, obs_solids = draw_obstacles(self.plotter, obstacles)
        slice_solids.extend(obs_solids)
        for a in self._obstacle_actors:
            a.SetVisibility(self.obs_cb.isChecked())
        self._clip_actors.extend(self._obstacle_actors)

        use_tube = self.tube_cb.isChecked()
        show_wit = self.witness_cb.isChecked()

        # Solver paths from report
        for gi, g in enumerate(report.get("groups") or []):
            gid = g.get("group_id") or g.get("id")
            color = COLORS[gi % len(COLORS)]
            tube_acts: list = []
            point_acts: list = []
            for task in g.get("tasks") or []:
                cells = task.get("cells") or []
                if len(cells) >= 2:
                    slice_solids.append(
                        pv.lines_from_points(cell_centers(cells, res)).tube(
                            radius=radius, n_sides=12
                        )
                    )
                t_acts, p_acts = draw_pipe(self.plotter, cells, radius, color, res)
                tube_acts.extend(t_acts)
                point_acts.extend(p_acts)

            self._clip_actors.extend(tube_acts)
            self._clip_actors.extend(point_acts)

            cb = QCheckBox(str(gid))
            cb.setChecked(True)
            self._solver_groups.append((tube_acts, point_acts, cb))

            def on_group(state, tubes=tube_acts, points=point_acts):
                on = state == Qt.Checked  # type: ignore
                show_tube = self.tube_cb.isChecked()
                for a in tubes:
                    a.SetVisibility(on and show_tube)
                for a in points:
                    a.SetVisibility(on and not show_tube)
                self.plotter.render()

            cb.stateChanged.connect(on_group)
            self.group_box.insertWidget(self.group_box.count() - 1, cb)
            for a in tube_acts:
                a.SetVisibility(use_tube)
            for a in point_acts:
                a.SetVisibility(not use_tube)

        # Witness overlay: lightened group hue + thinner / more transparent
        for gi, g in enumerate(report.get("groups") or []):
            gid = g.get("group_id") or g.get("id")
            color = witness_color(COLORS[gi % len(COLORS)])
            tubes: list = []
            points: list = []
            wit_r = radius * 0.55
            for path_node in (witness.get(gid) or {}).get("paths") or []:
                cells = path_node.get("cells") or []
                if len(cells) >= 2:
                    slice_solids.append(
                        pv.lines_from_points(cell_centers(cells, res)).tube(
                            radius=wit_r, n_sides=12
                        )
                    )
                t_acts, p_acts = draw_pipe(
                    self.plotter, cells, wit_r, color, res, opacity=0.4
                )
                tubes.extend(t_acts)
                points.extend(p_acts)
            self._clip_actors.extend(tubes)
            self._clip_actors.extend(points)
            self._witness_actors.append((tubes, points))
        self._set_witness_visibility(show_wit)

        if slice_solids:
            self._slice_mesh = pv.merge(slice_solids)

        ok = report.get("success")
        status = report.get("status", "")
        algo = report.get("algorithm", "?")
        cost = report.get("total_cost", 0)
        ref = report.get("reference_cost")
        gap = report.get("cost_gap")
        bits = [
            f"[{self.index + 1}/{len(self.reports)}]",
            report.get("instance_id") or inst_path.name,
            f"algo={algo}",
            f"{status}",
            f"cost={cost}",
        ]
        if report.get("witness_compared") and ref is not None:
            bits.append(f"ref={ref}")
            if gap is not None:
                bits.append(f"gap={gap}")
        if not ok:
            bits.append(f"fail={report.get('fail_reason') or ''}")
        n_aabb = len(obstacles.get("aabbs") or [])
        n_sph = len(obstacles.get("spheres") or [])
        n_cap = len(obstacles.get("capsules") or [])
        bits.append(f"obs aabb={n_aabb} sph={n_sph} cap={n_cap}")
        self.status.setText("  |  ".join(str(b) for b in bits))

        self.plotter.show_bounds(
            bounds=(0, sx * res, 0, sy * res, 0, sz * res),
            location="outer",
            color="gray",
        )
        self._sync_cut()
        self.plotter.reset_camera()
        self.plotter.render()


def main() -> int:
    ap = argparse.ArgumentParser(description="Browse DesignReport JSONL from pipe_run_design")
    ap.add_argument("path", nargs="?", default=None, help="design_reports.jsonl")
    ap.add_argument("--points", action="store_true", help="start in point mode")
    ap.add_argument(
        "--witness",
        action="store_true",
        help="show YAML constructive solution as reference overlay",
    )
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(use_tube=not args.points, show_witness=args.witness)
    win.show()
    if args.path:
        win.load_jsonl(Path(args.path))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
