"""Optimize-bench viewer for fixed single-group test cases.

Left:  OptimizeGraph (viz: point / point+line / point+line+tube).
Mid:   LossWeights, ignore-obstacles, method/steps, progress, Start / Replay / Revert.
Right: collapsible report (clearance / path / solve) + J-curve plot.

Replay forces point mode and only updates node positions (no line/tube rebuild).

Skeleton ``graph_<id>.json`` lives in the case folder (topology fixed). Each Start
writes ``run_XXX/`` via ``pipeline.run_pipeline`` (``record_state=True``). Replay
``unpack``s recorded ``x``; Revert deletes the latest run and restores the previous
run (or the skeleton if none remain).

CLI
  python -m pipe_smooth.vis_bench [--save-dir DIR]

  --save-dir   root for cases (default: ``pipe_smooth/runs``) → ``DIR/elbow_90/``

Menu → Examples → Elbow 90… opens a dialog (count / resolution / radius), builds
an L-polyline (``count`` waypoints per leg, spacing ``resolution``), exports the
skeleton, and clears stale runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # type: ignore
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from pipe_smooth.geometry.assemble import (
    HermiteNode,
    HermiteSegment,
    OptimizeGraph,
    OptimizeUsagePath,
)
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.optimize.pack import PackLayout, unpack
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.pipeline import METHODS, run_pipeline

_CLEAR_KEYS = (
    "kind",
    "margin",
    "n_hit",
    "n_hit_obs",
    "n_hit_grp",
    "max_shortfall",
    "ok",
)

# method → {name: (kind, default, *extra)}  kind: float | int | bool | choice
_OPT_SPECS: dict[str, dict[str, tuple]] = {
    "adam": {
        "lr": ("float", 1e-3),
        "beta1": ("float", 0.9),
        "beta2": ("float", 0.999),
        "eps": ("float", 1e-8),
        "refresh_every": ("int", 10),
    },
    "adam_annealed": {
        "eta_max": ("float", 1e-3),
        "delta": ("float", 1e-2),
        "T0": ("float", 1e-3),
        "anneal": ("choice", "linear", ("linear", "exp")),
        "rho": ("float", 0.95),
        "beta1": ("float", 0.9),
        "beta2": ("float", 0.999),
        "eps": ("float", 1e-8),
        "key": ("int", 0),
        "refresh_every": ("int", 10),
    },
    "langevin": {
        "eta_max": ("float", 1e-3),
        "delta": ("float", 1e-2),
        "T0": ("float", 1e-3),
        "anneal": ("choice", "linear", ("linear", "exp")),
        "rho": ("float", 0.95),
        "eps": ("float", 1e-8),
        "key": ("int", 0),
        "refresh_every": ("int", 10),
    },
    "elastic_qp": {
        "trust": ("float", 0.1),
        "eps_band": ("float", 0.2),
        "upper_only": ("bool", True),
        "beta_v": ("float", 0.4),
        "w_slack": ("float", 1e3),
        "refresh_every": ("int", 5),
    },
    "interior_nlp": {
        "max_iter": ("int", 5),
        "eps_band": ("float", 0.2),
        "upper_only": ("bool", True),
        "beta_v": ("float", 0.4),
        "refresh_every": ("int", 1),
    },
}


def make_elbow90(count: int = 5, resolution: float = 1.0, radius: float = 0.5) -> OptimizeGraph:
    """L-path: ``count`` waypoints along +x, then ``count`` along +y (shared corner)."""
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    sp, r = float(resolution), float(radius)
    nodes: dict[str, HermiteNode] = {}
    chain: list[str] = []
    for i in range(count):
        nid = f"a{i}"
        chain.append(nid)
        nodes[nid] = HermiteNode(
            pos=np.array([i * sp, 0.0, 0.0]),
            radius=r,
            v=None,
            fixed_p=(i == 0),
            fixed_v=(i == 0),
            role="port" if i == 0 else "trunk",
        )
    corner = chain[-1]
    for j in range(1, count):
        nid = f"b{j}"
        chain.append(nid)
        is_end = j == count - 1
        nodes[nid] = HermiteNode(
            pos=np.array([(count - 1) * sp, j * sp, 0.0]),
            radius=r,
            v=None,
            fixed_p=is_end,
            fixed_v=is_end,
            role="port" if is_end else "trunk",
        )

    segments: dict[str, HermiteSegment] = {}
    seg_ids: list[str] = []
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        p0, p1 = nodes[u].pos, nodes[v].pos
        tang = p1 - p0
        L = float(np.linalg.norm(tang))
        tang = tang / (L + 1e-12)
        if not nodes[u].fixed_v:
            nodes[u].v = tang * L
        if not nodes[v].fixed_v:
            nodes[v].v = tang * L
        sid = f"s{i}"
        segments[sid] = HermiteSegment(id=sid, edge="e0", kind="trunk", u=u, v=v)
        seg_ids.append(sid)

    c = nodes[corner]
    if not c.fixed_v:
        c.v = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)) * sp

    g = OptimizeGraph(
        id="elbow_90",
        nodes=nodes,
        segments=segments,
        edge_segs={"e0": list(seg_ids)},
        junctions={},
        usage_paths=[
            OptimizeUsagePath(
                usage=("A", "B"),
                edges=["e0"],
                edges_reverse=[False],
                segments=seg_ids,
                reverse=[False] * len(seg_ids),
                v_start=np.array([sp, 0.0, 0.0]),
                v_end=np.array([0.0, sp, 0.0]),
            )
        ],
    )
    g.check_usage_tangents_colinear()
    return g


class _OptWorker(QThread):
    done = pyqtSignal(dict)
    fail = pyqtSignal(str)
    progress = pyqtSignal(int, int, float)  # step, total, J

    def __init__(self, graphs, obstacles, kwargs):
        super().__init__()
        self.graphs = graphs
        self.obstacles = obstacles
        self.kwargs = kwargs

    def run(self):
        total = int(self.kwargs.get("steps", 1))
        orig = Recorder.record

        def hooked(rec, step, J, x=None, **extra):
            orig(rec, step, J, x, **extra)
            self.progress.emit(int(step), total, float(J))

        Recorder.record = hooked  # type: ignore[method-assign]
        try:
            self.done.emit(run_pipeline(self.graphs, self.obstacles, **self.kwargs))
        except Exception as e:
            self.fail.emit(str(e))
        finally:
            Recorder.record = orig  # type: ignore[method-assign]


class BenchViewer(QMainWindow):
    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.case_dir: Path | None = None
        self.resolution = 0.1
        self.graphs: list[OptimizeGraph] = []
        self._last_run: Path | None = None
        self._worker: _OptWorker | None = None
        self._replay_xs: list = []
        self._replay_i = 0
        self._replay_lay: PackLayout | None = None
        self._node_actors: dict[tuple[str, str], object] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._replay_tick)

        self.setWindowTitle("optimize bench")
        self.resize(1500, 900)
        split = QSplitter()
        self.setCentralWidget(split)

        self.plotter = QtInteractor(self)  # type: ignore
        split.addWidget(self.plotter.interactor)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.addWidget(QLabel("LossWeights"))
        form = QFormLayout()
        self.weight_spins: dict[str, QDoubleSpinBox] = {}
        defaults = LossWeights()
        for f in fields(LossWeights):
            sp = QDoubleSpinBox()
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(4)
            sp.setSingleStep(0.1)
            sp.setValue(float(getattr(defaults, f.name)))
            form.addRow(f.name, sp)
            self.weight_spins[f.name] = sp
        mv.addLayout(form)

        self.ignore_obs = QCheckBox("ignore obstacles")
        self.ignore_obs.setChecked(True)
        mv.addWidget(self.ignore_obs)

        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("viz"))
        self.viz_mode = QComboBox()
        self.viz_mode.addItems(["point", "point+line", "point+line+tube"])
        self.viz_mode.setCurrentText("point+line")
        self.viz_mode.currentTextChanged.connect(self._on_viz_mode)
        row_v.addWidget(self.viz_mode)
        mv.addLayout(row_v)

        row = QHBoxLayout()
        row.addWidget(QLabel("method"))
        self.method = QComboBox()
        self.method.addItems(list(METHODS))
        self.method.setCurrentText("adam")
        self.method.currentTextChanged.connect(self._rebuild_opt_board)
        row.addWidget(self.method)
        row.addWidget(QLabel("steps"))
        self.steps = QSpinBox()
        self.steps.setRange(1, 100000)
        self.steps.setValue(50)
        row.addWidget(self.steps)
        mv.addLayout(row)

        mv.addWidget(QLabel("Optimizer"))
        self.opt_form = QFormLayout()
        self.opt_widgets: dict[str, QWidget] = {}
        mv.addLayout(self.opt_form)
        self._rebuild_opt_board(self.method.currentText())

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m")
        mv.addWidget(self.progress)
        self.progress_lbl = QLabel("")
        mv.addWidget(self.progress_lbl)

        self.btn_start = QPushButton("Start")
        self.btn_replay = QPushButton("Replay")
        self.btn_revert = QPushButton("Revert")
        self.btn_start.clicked.connect(self._start)
        self.btn_replay.clicked.connect(self._replay)
        self.btn_revert.clicked.connect(self._revert)
        mv.addWidget(self.btn_start)
        mv.addWidget(self.btn_replay)
        mv.addWidget(self.btn_revert)
        mv.addStretch(1)
        split.addWidget(mid)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Report"))
        self.report_box = QToolBox()
        clear_w = QWidget()
        self.clear_form = QFormLayout(clear_w)
        self.clear_labels = {k: QLabel("—") for k in _CLEAR_KEYS}
        for k, lab in self.clear_labels.items():
            self.clear_form.addRow(k, lab)
        self.report_box.addItem(clear_w, "Clearance")
        self.path_tree = QTreeWidget()
        self.path_tree.setHeaderLabels(["metric", "value"])
        self.path_tree.setColumnCount(2)
        self.report_box.addItem(self.path_tree, "Path")
        solve_w = QWidget()
        solve_v = QVBoxLayout(solve_w)
        self.solve_form = QFormLayout()
        self.solve_labels: dict[str, QLabel] = {}
        solve_v.addLayout(self.solve_form)
        self.j_fig = Figure(figsize=(3.2, 2.2), tight_layout=True)
        self.j_ax = self.j_fig.add_subplot(111)
        self.j_canvas = FigureCanvasQTAgg(self.j_fig)
        solve_v.addWidget(self.j_canvas)
        self.report_box.addItem(solve_w, "Solve")
        rv.addWidget(self.report_box)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)

        m = self.menuBar().addMenu("&Examples")
        act = m.addAction("Elbow 90…")
        act.triggered.connect(self._example_elbow90)

        self.plotter.add_axes()
        self._set_busy(True)
        self.btn_revert.setEnabled(False)

    def _rebuild_opt_board(self, method: str = ""):
        method = method or self.method.currentText()
        while self.opt_form.rowCount():
            self.opt_form.removeRow(0)
        self.opt_widgets = {}
        for name, spec in _OPT_SPECS.get(method, {}).items():
            kind, default = spec[0], spec[1]
            if kind == "float":
                w = QDoubleSpinBox()
                w.setRange(-1e12, 1e12)
                w.setDecimals(8)
                w.setSingleStep(max(abs(float(default)) * 0.1, 1e-6))
                w.setValue(float(default))
            elif kind == "int":
                w = QSpinBox()
                w.setRange(0, 10**9)
                w.setValue(int(default))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            else:  # choice
                w = QComboBox()
                w.addItems(list(spec[2]))
                w.setCurrentText(str(default))
            self.opt_form.addRow(name, w)
            self.opt_widgets[name] = w

    def _opt_kwargs(self) -> dict:
        out: dict = {}
        for name, w in self.opt_widgets.items():
            if isinstance(w, QDoubleSpinBox):
                out[name] = float(w.value())
            elif isinstance(w, QSpinBox):
                out[name] = int(w.value())
            elif isinstance(w, QCheckBox):
                out[name] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                out[name] = w.currentText()
        return out

    def _runs(self) -> list[Path]:
        if self.case_dir is None:
            return []
        return sorted(p for p in self.case_dir.glob("run_*") if p.is_dir())

    def _load_graphs_from(self, folder: Path):
        paths = sorted(folder.glob("graph_*.json"))
        if not paths:
            raise FileNotFoundError(f"no graph_*.json in {folder}")
        self.graphs = [OptimizeGraph.load(p) for p in paths]

    def _show_report(self, store_cfg: str | Path | None = None):
        """Load ``report`` / ``recorder`` from a run's ``store_cfg.json`` and refresh UI."""
        report: dict = {}
        j_curve = None
        if store_cfg is not None:
            cfg_path = Path(store_cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "store_cfg.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                rp = cfg.get("report")
                if rp and Path(rp).exists():
                    report = json.loads(Path(rp).read_text())
                rec_p = cfg.get("recorder")
                if rec_p and Path(rec_p).exists():
                    j_curve = Recorder.load(rec_p).J_curve()

        clear = report.get("clearance") or {}
        for k, lab in self.clear_labels.items():
            v = clear.get(k, "—")
            lab.setText(f"{v:.6g}" if isinstance(v, float) else str(v))

        self.path_tree.clear()
        path = report.get("path") or {}
        agg = path.get("aggregate") or {}
        root_agg = QTreeWidgetItem(["aggregate", ""])
        self.path_tree.addTopLevelItem(root_agg)
        for k in ("L", "kappa_max", "V"):
            QTreeWidgetItem(
                root_agg, [k, f"{float(agg[k]):.6g}" if k in agg else "—"]
            )
        for gid, gdat in (path.get("groups") or {}).items():
            gi = QTreeWidgetItem([f"group {gid}", ""])
            self.path_tree.addTopLevelItem(gi)
            for k in ("L", "kappa_max", "V"):
                if k in gdat:
                    QTreeWidgetItem(gi, [k, f"{float(gdat[k]):.6g}"])
            for ukey, udat in (gdat.get("usages") or {}).items():
                ui = QTreeWidgetItem(gi, [f"usage {ukey}", ""])
                for k in ("L", "kappa_max"):
                    if k in udat:
                        QTreeWidgetItem(ui, [k, f"{float(udat[k]):.6g}"])
        self.path_tree.expandToDepth(1)

        solve = report.get("solve") or {}
        while self.solve_form.rowCount():
            self.solve_form.removeRow(0)
        self.solve_labels.clear()
        for k, v in solve.items():
            lab = QLabel(f"{v:.6g}" if isinstance(v, float) else str(v))
            self.solve_labels[k] = lab
            self.solve_form.addRow(k, lab)

        self.j_ax.clear()
        if j_curve is not None and len(j_curve):
            self.j_ax.plot(np.arange(1, len(j_curve) + 1), j_curve, color="C0", lw=1.5)
            self.j_ax.set_xlabel("step")
            self.j_ax.set_ylabel("J")
            self.j_ax.set_title("J curve")
            self.j_ax.grid(True, alpha=0.3)
        else:
            self.j_ax.text(0.5, 0.5, "no J curve", ha="center", va="center")
            self.j_ax.set_axis_off()
        self.j_canvas.draw_idle()

    def _example_elbow90(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Elbow 90")
        form = QFormLayout(dlg)
        sp_count = QSpinBox()
        sp_count.setRange(2, 500)
        sp_count.setValue(5)
        sp_res = QDoubleSpinBox()
        sp_res.setRange(1e-4, 1e4)
        sp_res.setDecimals(4)
        sp_res.setValue(1.0)
        sp_rad = QDoubleSpinBox()
        sp_rad.setRange(1e-4, 1e3)
        sp_rad.setDecimals(4)
        sp_rad.setValue(0.5)
        form.addRow("count (per leg)", sp_count)
        form.addRow("resolution (spacing)", sp_res)
        form.addRow("radius", sp_rad)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return

        count = int(sp_count.value())
        self.resolution = float(sp_res.value())
        radius = float(sp_rad.value())
        self.case_dir = self.save_dir / "elbow_90"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        self.graphs = [make_elbow90(count, self.resolution, radius)]
        for g in self.graphs:
            g.export(self.case_dir / f"graph_{g.id}.json")
        self.setWindowTitle(f"optimize bench — elbow_90  (n={count})")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self.redraw()
        self.plotter.reset_camera()

    def _next_run_dir(self) -> Path:
        assert self.case_dir is not None
        n = 1
        while (self.case_dir / f"run_{n:03d}").exists():
            n += 1
        return self.case_dir / f"run_{n:03d}"

    def _weights(self) -> LossWeights:
        return LossWeights(
            **{k: float(sp.value()) for k, sp in self.weight_spins.items()}
        )

    def _on_viz_mode(self, _mode: str = ""):
        if self.graphs and not self._timer.isActive():
            self.redraw()

    def redraw(self):
        mode = self.viz_mode.currentText()
        self._node_actors = {}
        self.plotter.clear()
        self.plotter.add_axes()
        for g in self.graphs:
            if mode != "point":
                g.refresh(backend="numpy", resolution=max(self.resolution * 0.25, 1e-3))
                for seg in g.segments.values():
                    xyz = np.asarray(seg.xyz, float)
                    if xyz.ndim != 2 or xyz.shape[0] < 2:
                        continue
                    line = pv.lines_from_points(xyz)
                    self.plotter.add_mesh(line, color="blue", line_width=2)
                    if mode == "point+line+tube":
                        line["radius"] = np.asarray(seg.radius, float)
                        tube = line.tube(scalars="radius", absolute=True, n_sides=12)
                        self.plotter.add_mesh(tube, color="lightblue", opacity=0.35)
            for nid, n in g.nodes.items():
                actor = self.plotter.add_mesh(
                    pv.Sphere(radius=0.15, center=(0.0, 0.0, 0.0)),
                    color="red",
                    name=f"n:{g.id}:{nid}",
                )
                pos = np.asarray(n.pos, float)
                actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
                self._node_actors[(g.id, nid)] = actor
        self.plotter.render()

    def _update_points(self):
        for g in self.graphs:
            for nid, n in g.nodes.items():
                actor = self._node_actors.get((g.id, nid))
                if actor is None:
                    continue
                pos = np.asarray(n.pos, float)
                actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))  # type: ignore
        self.plotter.render()

    def _set_busy(self, busy: bool):
        on = (not busy) and bool(self.graphs)
        self.btn_start.setEnabled(on)
        self.btn_replay.setEnabled(on)
        self.btn_revert.setEnabled(on and bool(self._runs()))

    def _on_progress(self, step: int, total: int, J: float):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(min(step, total))
        self.progress_lbl.setText(f"step {step}/{total}  J={J:.6g}")

    def _start(self):
        if not self.graphs or self.case_dir is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._timer.stop()
        self.viz_mode.setEnabled(True)
        out = self._next_run_dir()
        clear_kind = "ignore" if self.ignore_obs.isChecked() else "capsule"
        steps = int(self.steps.value())
        kw = dict(
            method=self.method.currentText(),
            clear_kind=clear_kind,
            weights=self._weights(),
            steps=steps,
            out_dir=out,
            record_state=True,
            verbose=False,
            **self._opt_kwargs(),
        )
        self.progress.setMaximum(steps)
        self.progress.setValue(0)
        self.progress_lbl.setText("starting…")
        self._set_busy(True)
        self._worker = _OptWorker(self.graphs, [], kw)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_opt_done)
        self._worker.fail.connect(self._on_opt_fail)
        self._worker.start()

    def _on_opt_done(self, result: dict):
        self._last_run = Path(result["out_dir"])
        self._show_report(self._last_run / "store_cfg.json")
        self.progress.setValue(self.progress.maximum())
        self.progress_lbl.setText("done")
        self._set_busy(False)
        self.redraw()
        self.plotter.reset_camera()

    def _on_opt_fail(self, msg: str):
        self.progress_lbl.setText("failed")
        self._set_busy(False)
        QMessageBox.critical(self, "Optimize failed", msg)

    def _replay(self):
        runs = self._runs()
        run = self._last_run if self._last_run and self._last_run.exists() else (
            runs[-1] if runs else None
        )
        if run is None or not (run / "recorder.pkl").exists():
            QMessageBox.warning(self, "Replay", "No run with recorder.pkl yet.")
            return
        if not (run / "layout.json").exists():
            QMessageBox.warning(self, "Replay", "Missing layout.json in last run.")
            return
        assert self.case_dir is not None
        self._load_graphs_from(self.case_dir)
        rec = Recorder.load(run / "recorder.pkl")
        xs = rec.states()
        if not xs:
            QMessageBox.warning(
                self, "Replay", "Recorder has no x snapshots (record_state?)."
            )
            return
        self._replay_lay = PackLayout.load(run / "layout.json")
        self._replay_xs = xs
        self._replay_i = 0
        self.viz_mode.blockSignals(True)
        self.viz_mode.setCurrentText("point")
        self.viz_mode.blockSignals(False)
        self.viz_mode.setEnabled(False)
        self.redraw()
        self._timer.start(50)

    def _replay_tick(self):
        if self._replay_i >= len(self._replay_xs) or self._replay_lay is None:
            self._timer.stop()
            self.viz_mode.setEnabled(True)
            return
        unpack(self._replay_xs[self._replay_i], self.graphs, self._replay_lay)
        self._replay_i += 1
        self._update_points()

    def _revert(self):
        self._timer.stop()
        self.viz_mode.setEnabled(True)
        runs = self._runs()
        if not runs or self.case_dir is None:
            QMessageBox.warning(self, "Revert", "No optimized run to delete.")
            return
        current = runs[-1]
        shutil.rmtree(current)
        self._last_run = None
        remain = self._runs()
        try:
            if remain:
                self._last_run = remain[-1]
                self._load_graphs_from(remain[-1])
                self._show_report(remain[-1])
            else:
                self._load_graphs_from(self.case_dir)
                self._show_report(None)
        except Exception as e:
            QMessageBox.critical(self, "Revert failed", str(e))
            return
        self._set_busy(False)
        self.redraw()


def main(argv: list[str] | None = None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="pipe_smooth optimize bench viewer")
    parser.add_argument(
        "--save-dir",
        "-o",
        type=Path,
        default=root / "runs",
        help="root for cases (default: pipe_smooth/runs → runs/elbow_90/)",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        for c in (
            Path(sys.prefix) / "plugins" / "platforms",
            Path(sys.prefix) / "plugins",
        ):
            if (c / "libqxcb.so").exists() or (c / "platforms" / "libqxcb.so").exists():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
                    c if (c / "libqxcb.so").exists() else c / "platforms"
                )
                break

    app = QApplication([sys.argv[0]])
    w = BenchViewer(args.save_dir)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
