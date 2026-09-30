"""Optimize-bench viewer for fixed single-group test cases.

Left:  OptimizeGraph + orange sphere obstacles (viz mode / usage arrows).
Mid:   collapsible Optimize (weights + run) / Viz (mode + usage).
Right: collapsible report (clearance / path / solve) + J-curve plot.

Replay forces point mode and only updates node positions (no line/tube rebuild).

Skeleton ``graph_<id>.json`` (+ optional ``obstacles.json``) lives in the case
folder. Each Start writes ``run_XXX/`` via ``pipeline.run_pipeline``
(``record_state=True``). Replay ``unpack``s recorded ``x``; Revert deletes the
latest run and restores the previous run (or the skeleton if none remain).

CLI
  python -m pipe_smooth.vis_bench [--save-dir DIR]

  --save-dir   root for cases (default: ``pipe_smooth/runs``)

Menu → File → Load YAML… loads an instance env.
Menu → Examples → Elbow / S-curve / Spiral / Branch union / M-shape… builds a case and clears stale runs.
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
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # type: ignore
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
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

from pipe_smooth.collision import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights
from pipe_smooth.optimize.pack import PackLayout, unpack
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.pipeline import (
    METHODS,
    build_graphs,
    dump_obstacles,
    load_obstacles,
    load_obstacles_file,
    run_pipeline,
)
from pipe_smooth.vis_bench.conditions import (
    MAX_BRANCH,
    M_USAGE_PAIRS,
    make_branch_union,
    make_elbow90,
    make_m_shape,
    make_s_curve,
    make_spiral_entangle,
)

_CLEAR_KEYS = (
    "kind",
    "margin",
    "n_hit",
    "n_hit_obs",
    "n_hit_grp",
    "max_shortfall",
    "ok",
)

_GROUP_COLS = (
    (0.90, 0.35, 0.20),
    (0.20, 0.70, 0.35),
    (0.25, 0.45, 0.95),
    (0.90, 0.75, 0.15),
    (0.70, 0.30, 0.80),
    (0.10, 0.75, 0.75),
    (0.95, 0.50, 0.20),
    (0.55, 0.35, 0.20),
    (0.40, 0.80, 0.55),
    (0.85, 0.25, 0.55),
    (0.30, 0.55, 0.85),
    (0.75, 0.85, 0.25),
    (0.55, 0.20, 0.45),
    (0.15, 0.60, 0.55),
    (0.95, 0.65, 0.45),
    (0.45, 0.45, 0.75),
    (0.80, 0.40, 0.15),
    (0.35, 0.75, 0.20),
    (0.60, 0.25, 0.90),
    (0.20, 0.85, 0.85),
    (0.85, 0.55, 0.70),
    (0.50, 0.65, 0.30),
    (0.15, 0.40, 0.70),
    (0.70, 0.70, 0.40),
)

# method → {name: (kind, default, *extra)}  kind: float | int | bool | choice
_OPT_SPECS: dict[str, dict[str, tuple]] = {
    "adam": {
        "lr": ("float", 1e-3),
        "beta1": ("float", 0.9),
        "beta2": ("float", 0.999),
        "eps": ("float", 1e-8),
        "refresh_every": ("int", 5),
    },
    "adam_annealed": {
        "eta_max": ("float", 1e-2),
        "delta": ("float", 1e-2),
        "T0": ("float", 1e-3),
        "anneal": ("choice", "linear", ("linear", "exp")),
        "rho": ("float", 0.95),
        "beta1": ("float", 0.9),
        "beta2": ("float", 0.999),
        "eps": ("float", 1e-8),
        "key": ("int", 0),
        "refresh_every": ("int", 5),
    },
    "langevin": {
        "eta_max": ("float", 1e-2),
        "delta": ("float", 1e-2),
        "T0": ("float", 1e-3),
        "anneal": ("choice", "linear", ("linear", "exp")),
        "rho": ("float", 0.95),
        "eps": ("float", 1e-8),
        "key": ("int", 0),
        "refresh_every": ("int", 5),
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
        self.obstacles: list = []
        self._last_run: Path | None = None
        self._worker: _OptWorker | None = None
        self._replay_xs: list = []
        self._replay_i = 0
        self._replay_lay: PackLayout | None = None
        self._node_poly: dict[str, object] = {}  # gid → PolyData (batched points)
        self._group_actors: dict[str, list] = {}  # gid → actors
        self._usage_actors: list[tuple[str, object]] = []  # (usage key, actor)
        self._obs_actor = None
        self.show_obstacles = True
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
        self.mid_box = QToolBox()
        mv.addWidget(self.mid_box)

        opt_w = QWidget()
        ov = QVBoxLayout(opt_w)
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
        ov.addLayout(form)
        row_ck = QHBoxLayout()
        row_ck.addWidget(QLabel("clear_kind"))
        self.clear_kind = QComboBox()
        self.clear_kind.addItems(["capsule", "hull", "ignore"])
        self.clear_kind.setCurrentText("ignore")
        row_ck.addWidget(self.clear_kind)
        ov.addLayout(row_ck)
        row_uk = QHBoxLayout()
        row_uk.addWidget(QLabel("union_kind"))
        self.union_kind = QComboBox()
        self.union_kind.addItems(["chamfer", "ignore"])
        self.union_kind.setCurrentText("ignore")
        row_uk.addWidget(self.union_kind)
        ov.addLayout(row_uk)
        row_mg = QHBoxLayout()
        row_mg.addWidget(QLabel("margin"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1e3)
        self.margin.setDecimals(4)
        self.margin.setSingleStep(0.1)
        self.margin.setValue(0.0)
        row_mg.addWidget(self.margin)
        ov.addLayout(row_mg)
        for name, default in (
            ("p_max_obs", 128),
            ("p_max_grp", 128),
            ("p_max_union", 64),
        ):
            row_p = QHBoxLayout()
            row_p.addWidget(QLabel(name))
            sp = QSpinBox()
            sp.setRange(0, 10**7)
            sp.setValue(int(default))
            row_p.addWidget(sp)
            ov.addLayout(row_p)
            setattr(self, name, sp)
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
        ov.addLayout(row)
        self.opt_form = QFormLayout()
        self.opt_widgets: dict[str, QWidget] = {}
        ov.addLayout(self.opt_form)
        self._rebuild_opt_board(self.method.currentText())
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m")
        ov.addWidget(self.progress)
        self.progress_lbl = QLabel("")
        ov.addWidget(self.progress_lbl)
        self.btn_start = QPushButton("Start")
        self.btn_replay = QPushButton("Replay")
        self.btn_revert = QPushButton("Revert")
        self.btn_start.clicked.connect(self._start)
        self.btn_replay.clicked.connect(self._replay)
        self.btn_revert.clicked.connect(self._revert)
        ov.addWidget(self.btn_start)
        ov.addWidget(self.btn_replay)
        ov.addWidget(self.btn_revert)
        ov.addStretch(1)
        self.mid_box.addItem(opt_w, "Optimize")

        viz_w = QWidget()
        vv = QVBoxLayout(viz_w)
        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("mode"))
        self.viz_mode = QComboBox()
        self.viz_mode.addItems(["point", "point+line", "point+line+tube"])
        self.viz_mode.setCurrentText("point+line")
        self.viz_mode.currentTextChanged.connect(self._on_viz_mode)
        row_v.addWidget(self.viz_mode)
        vv.addLayout(row_v)
        vv.addWidget(QLabel("Groups"))
        self.group_box = QVBoxLayout()
        vv.addLayout(self.group_box)
        vv.addWidget(QLabel("Usage paths"))
        self.usage_box = QVBoxLayout()
        vv.addLayout(self.usage_box)
        vv.addStretch(1)
        self.mid_box.addItem(viz_w, "Viz")
        self.group_on: dict[str, bool] = {}
        self.usage_on: dict[str, bool] = {}
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

        mf = self.menuBar().addMenu("&File")
        mf.addAction("Load YAML…").triggered.connect(self._load_yaml)
        mvw = self.menuBar().addMenu("&View")
        act_obs = QAction("Obstacles", self)
        act_obs.setCheckable(True)
        act_obs.setChecked(True)
        act_obs.toggled.connect(self._toggle_obstacles)
        mvw.addAction(act_obs)
        m = self.menuBar().addMenu("&Examples")
        m.addAction("Elbow 90…").triggered.connect(self._example_elbow90)
        m.addAction("S-curve + 2 obs…").triggered.connect(self._example_s_curve)
        m.addAction("Spiral entangle…").triggered.connect(self._example_spiral_entangle)
        m.addAction("Branch union…").triggered.connect(self._example_branch_union)
        m.addAction("M-shape usages…").triggered.connect(self._example_m_shape)

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
        rec = None
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
                    rec = Recorder.load(rec_p)

        clear = report.get("clearance") or {}
        for k, lab in self.clear_labels.items():
            v = clear.get(k, "—")
            lab.setText(f"{v:.6g}" if isinstance(v, float) else str(v))

        self.path_tree.clear()
        path = report.get("path") or {}
        agg = path.get("aggregate") or {}
        root_agg = QTreeWidgetItem(["aggregate", ""])
        self.path_tree.addTopLevelItem(root_agg)
        for k in ("L", "kr_max", "V"):
            v = agg.get(k, agg.get("kappa_max") if k == "kr_max" else None)
            QTreeWidgetItem(
                root_agg, [k, f"{float(v):.6g}" if isinstance(v, (int, float)) else "—"]
            )
        for gid, gdat in (path.get("groups") or {}).items():
            gi = QTreeWidgetItem([f"group {gid}", ""])
            self.path_tree.addTopLevelItem(gi)
            for k in ("L", "kr_max", "V"):
                v = gdat.get(k, gdat.get("kappa_max") if k == "kr_max" else None)
                if isinstance(v, (int, float)):
                    QTreeWidgetItem(gi, [k, f"{float(v):.6g}"])
            for ukey, udat in (gdat.get("usages") or {}).items():
                ui = QTreeWidgetItem(gi, [f"usage {ukey}", ""])
                for k in ("L", "kr_max"):
                    v = udat.get(k, udat.get("kappa_max") if k == "kr_max" else None)
                    if isinstance(v, (int, float)):
                        QTreeWidgetItem(ui, [k, f"{float(v):.6g}"])
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
        hist = rec.history if rec is not None else []
        if hist:
            steps = np.arange(1, len(hist) + 1)
            self.j_ax.plot(steps, [h["J"] for h in hist], color="C0", lw=1.5, label="J")
            for k in ("L", "K", "space", "v", "union", "clear"):
                if k in hist[0]:
                    self.j_ax.plot(
                        steps, [float(h.get(k, 0.0)) for h in hist], lw=1.1, label=k
                    )
            self.j_ax.set_xlabel("step")
            self.j_ax.set_ylabel("cost")
            self.j_ax.set_title("J / sub-costs")
            self.j_ax.legend(fontsize=7, loc="best")
            self.j_ax.grid(True, alpha=0.3)
        else:
            self.j_ax.text(0.5, 0.5, "no J curve", ha="center", va="center")
            self.j_ax.set_axis_off()
        self.j_canvas.draw_idle()

    def _load_yaml(self):
        start = Path(__file__).resolve().parents[1] / "data" / "hard"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load YAML",
            str(start if start.is_dir() else self.save_dir),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        path = Path(path)
        try:
            env = yaml.safe_load(path.read_text()) or {}
            graphs = build_graphs(env)["graphs"]
            obs = load_obstacles(env, clear_kind="capsule")
        except Exception as e:
            QMessageBox.warning(self, "Load YAML", str(e))
            return

        self.resolution = float((env.get("grid") or {}).get("resolution", 1.0))
        self.case_dir = self.save_dir / path.stem
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        self.graphs = graphs
        self.obstacles = list(obs)
        self._write_obstacles()
        for g in self.graphs:
            g.export(self.case_dir / f"graph_{g.id}.json")
        self.setWindowTitle(f"optimize bench — {path.stem}")
        self.clear_kind.setCurrentText("capsule")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

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
        self.obstacles = []
        self._write_obstacles()
        for g in self.graphs:
            g.export(self.case_dir / f"graph_{g.id}.json")
        self.setWindowTitle(f"optimize bench — elbow_90  (n={count})")
        self.clear_kind.setCurrentText("ignore")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

    def _example_s_curve(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("S-curve + 2 obstacles")
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
        sp_obs0 = QDoubleSpinBox()
        sp_obs0.setRange(1e-4, 1e3)
        sp_obs0.setDecimals(4)
        sp_obs0.setValue(0.6)
        sp_obs1 = QDoubleSpinBox()
        sp_obs1.setRange(1e-4, 1e3)
        sp_obs1.setDecimals(4)
        sp_obs1.setValue(0.6)
        form.addRow("count (per leg)", sp_count)
        form.addRow("resolution (spacing)", sp_res)
        form.addRow("pipe radius", sp_rad)
        form.addRow("obs bay0 radius", sp_obs0)
        form.addRow("obs bay1 radius", sp_obs1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return

        count = int(sp_count.value())
        self.resolution = float(sp_res.value())
        radius = float(sp_rad.value())
        self.case_dir = self.save_dir / "s_curve"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        g, obs = make_s_curve(
            count,
            self.resolution,
            radius,
            obs_r0=float(sp_obs0.value()),
            obs_r1=float(sp_obs1.value()),
        )
        self.graphs = [g]
        self.obstacles = list(obs)
        self._write_obstacles()
        for gr in self.graphs:
            gr.export(self.case_dir / f"graph_{gr.id}.json")
        self.setWindowTitle(f"optimize bench — s_curve  (n={count})")
        self.clear_kind.setCurrentText("capsule")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

    def _example_spiral_entangle(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Spiral entangle (2 groups)")
        form = QFormLayout(dlg)
        sp_count = QSpinBox()
        sp_count.setRange(3, 500)
        sp_count.setValue(24)
        sp_turns = QDoubleSpinBox()
        sp_turns.setRange(0.25, 20.0)
        sp_turns.setDecimals(2)
        sp_turns.setValue(2.0)
        sp_hr = QDoubleSpinBox()
        sp_hr.setRange(1e-3, 1e3)
        sp_hr.setDecimals(4)
        sp_hr.setValue(2.0)
        sp_pitch = QDoubleSpinBox()
        sp_pitch.setRange(1e-3, 1e3)
        sp_pitch.setDecimals(4)
        sp_pitch.setValue(4.0)
        sp_rad = QDoubleSpinBox()
        sp_rad.setRange(1e-4, 1e3)
        sp_rad.setDecimals(4)
        sp_rad.setValue(0.6)
        form.addRow("count (samples)", sp_count)
        form.addRow("turns", sp_turns)
        form.addRow("helix radius", sp_hr)
        form.addRow("pitch (per turn)", sp_pitch)
        form.addRow("pipe radius", sp_rad)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return

        count = int(sp_count.value())
        self.case_dir = self.save_dir / "spiral_entangle"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        self.graphs = make_spiral_entangle(
            count,
            turns=float(sp_turns.value()),
            helix_r=float(sp_hr.value()),
            pitch=float(sp_pitch.value()),
            radius=float(sp_rad.value()),
        )
        n0 = self.graphs[0]
        self.resolution = float(np.linalg.norm(n0.pos[1] - n0.pos[0]))
        self.obstacles = []
        self._write_obstacles()
        for gr in self.graphs:
            gr.export(self.case_dir / f"graph_{gr.id}.json")
        self.setWindowTitle(f"optimize bench — spiral_entangle  (n={count})")
        self.clear_kind.setCurrentText("capsule")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

    def _example_branch_union(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Branch union")
        form = QFormLayout(dlg)
        sp_count = QSpinBox()
        sp_count.setRange(2, 500)
        sp_count.setValue(4)
        sp_nb = QSpinBox()
        sp_nb.setRange(2, MAX_BRANCH)
        sp_nb.setValue(3)
        sp_res = QDoubleSpinBox()
        sp_res.setRange(1e-4, 1e4)
        sp_res.setDecimals(4)
        sp_res.setValue(1.0)
        sp_rad = QDoubleSpinBox()
        sp_rad.setRange(1e-4, 1e3)
        sp_rad.setDecimals(4)
        sp_rad.setValue(0.5)
        sp_gap = QDoubleSpinBox()
        sp_gap.setRange(1e-4, 1e3)
        sp_gap.setDecimals(4)
        sp_gap.setValue(2.0)
        form.addRow("count (per leg)", sp_count)
        form.addRow(f"n_branch (≤{MAX_BRANCH})", sp_nb)
        form.addRow("resolution (spacing)", sp_res)
        form.addRow("pipe radius", sp_rad)
        form.addRow("fork_gap", sp_gap)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return

        count = int(sp_count.value())
        n_branch = int(sp_nb.value())
        self.resolution = float(sp_res.value())
        self.case_dir = self.save_dir / "branch_union"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        g, tree, part = make_branch_union(
            count,
            n_branch=n_branch,
            resolution=self.resolution,
            radius=float(sp_rad.value()),
            fork_gap=float(sp_gap.value()),
        )
        self.graphs = [g]
        self.obstacles = []
        self._write_obstacles()
        tree.export(self.case_dir / f"tree_{tree.id}.json")
        part.export(self.case_dir / f"partitioned_{part.id}.json")
        g.export(self.case_dir / f"graph_{g.id}.json")
        self.setWindowTitle(f"optimize bench — branch_union  (n={count}, b={n_branch})")
        self.clear_kind.setCurrentText("ignore")
        if "lam" in self.weight_spins:
            self.weight_spins["lam"].setValue(1.0)
        self.union_kind.setCurrentText("chamfer")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

    def _example_m_shape(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("M-shape usages")
        form = QFormLayout(dlg)
        sp_step = QSpinBox()
        sp_step.setRange(1, 500)
        sp_step.setValue(2)
        sp_height = QSpinBox()
        sp_height.setRange(1, 500)
        sp_height.setValue(2)
        sp_res = QDoubleSpinBox()
        sp_res.setRange(1e-4, 1e4)
        sp_res.setDecimals(4)
        sp_res.setValue(1.0)
        sp_rad = QDoubleSpinBox()
        sp_rad.setRange(1e-4, 1e3)
        sp_rad.setDecimals(4)
        sp_rad.setValue(0.5)
        form.addRow("step (cells A–B)", sp_step)
        form.addRow("height (cells)", sp_height)
        form.addRow("resolution", sp_res)
        form.addRow("pipe radius", sp_rad)
        default_on = {("A", "B"), ("B", "C"), ("A", "C")}
        usage_cbs: dict[tuple[str, str], QCheckBox] = {}
        for pair in M_USAGE_PAIRS:
            cb = QCheckBox(f"{pair[0]}→{pair[1]}")
            cb.setChecked(pair in default_on)
            usage_cbs[pair] = cb
            form.addRow(cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        selected = [p for p, cb in usage_cbs.items() if cb.isChecked()]
        if not selected:
            return

        step = int(sp_step.value())
        height = int(sp_height.value())
        self.resolution = float(sp_res.value())
        self.case_dir = self.save_dir / "m_shape"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        for p in self._runs():
            shutil.rmtree(p)
        self._last_run = None
        g, tree, part = make_m_shape(
            step=step,
            height=height,
            resolution=self.resolution,
            radius=float(sp_rad.value()),
            usages=selected,
        )
        self.graphs = [g]
        self.obstacles = []
        self._write_obstacles()
        tree.export(self.case_dir / f"tree_{tree.id}.json")
        part.export(self.case_dir / f"partitioned_{part.id}.json")
        g.export(self.case_dir / f"graph_{g.id}.json")
        self.setWindowTitle(
            f"optimize bench — m_shape  (step={step}, h={height}, usages={len(selected)})"
        )
        self.clear_kind.setCurrentText("ignore")
        self.union_kind.setCurrentText("chamfer")
        self._show_report(None)
        self.progress.setValue(0)
        self.progress_lbl.setText("")
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()
        self.plotter.reset_camera()

    def _write_obstacles(self):
        assert self.case_dir is not None
        dump_obstacles(self.obstacles, self.case_dir / "obstacles.json")

    def _load_obstacles(self):
        self.obstacles = []
        if self.case_dir is None:
            return
        path = self.case_dir / "obstacles.json"
        if path.exists():
            self.obstacles = load_obstacles_file(path, clear_kind="capsule")

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

    def _fill_usage_checks(self):
        while self.group_box.count():
            item = self.group_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        prev_g = dict(self.group_on)
        self.group_on = {}
        for g in self.graphs:
            gid = str(g.id)
            self.group_on[gid] = prev_g.get(gid, True)
            cb = QCheckBox(gid)
            cb.setChecked(self.group_on[gid])
            cb.toggled.connect(
                lambda on, k=gid: (
                    self.group_on.__setitem__(k, on),
                    self._sync_group_vis(),
                )
            )
            self.group_box.addWidget(cb)

        while self.usage_box.count():
            item = self.usage_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        prev = dict(self.usage_on)
        self.usage_on = {}
        for g in self.graphs:
            for up in g.usage_paths:
                key = f"{g.id}|{up.usage[0]}->{up.usage[1]}"
                self.usage_on[key] = prev.get(key, False)
                cb = QCheckBox(key)
                cb.setChecked(self.usage_on[key])
                cb.toggled.connect(
                    lambda on, k=key: (
                        self.usage_on.__setitem__(k, on),
                        self._sync_usage_vis(),
                    )
                )
                self.usage_box.addWidget(cb)

    def _sync_group_vis(self):
        for gid, actors in self._group_actors.items():
            vis = bool(self.group_on.get(gid, True))
            for a in actors:
                a.SetVisibility(vis)  # type: ignore
        self.plotter.render()

    def _sync_usage_vis(self):
        """Rebuild scene when usage toggles (overlays are only built when on)."""
        if self.graphs and not self._timer.isActive():
            self.redraw()

    def redraw(self):
        from pipe_smooth.geometry.assemble import HermiteSegment

        mode = self.viz_mode.currentText()
        self._node_poly = {}
        self._group_actors = {}
        self._usage_actors = []
        self.plotter.clear()
        self.plotter.add_axes()
        for gi, g in enumerate(self.graphs):
            gid = str(g.id)
            col = _GROUP_COLS[gi % len(_GROUP_COLS)]
            actors: list = []
            ends = np.asarray(g.hermite_ends(backend="numpy"), dtype=float)
            spacing = max(float(self.resolution) * 0.25, 1e-3)
            T = 5
            for s in range(len(g.segments)):
                T = max(
                    T,
                    HermiteSegment.ts_count_for(
                        ends[s, 0], ends[s, 1], ends[s, 2], ends[s, 3], spacing
                    ),
                )
            ends, xyz, rad = g.refresh(ts_count=T, backend="numpy", store=True)
            xyz = np.asarray(xyz, float)
            rad = np.asarray(rad, float)
            pos = np.asarray(g.pos, float)
            if mode != "point":
                lines, tubes = [], []
                for sid in range(len(g.segments)):
                    pts = xyz[sid]
                    if pts.ndim != 2 or pts.shape[0] < 2:
                        continue
                    line = pv.lines_from_points(pts)
                    lines.append(line)
                    if mode == "point+line+tube":
                        line["radius"] = rad[sid]
                        tubes.append(line.tube(scalars="radius", absolute=True, n_sides=8))
                if lines:
                    actors.append(
                        self.plotter.add_mesh(
                            pv.merge(lines), color=col, line_width=2, render=False
                        )
                    )
                if tubes:
                    actors.append(
                        self.plotter.add_mesh(
                            pv.merge(tubes), color=col, opacity=0.35, render=False
                        )
                    )
            pdata = pv.PolyData(pos.copy())
            actors.append(
                self.plotter.add_mesh(
                    pdata,
                    color=col,
                    point_size=5,
                    render_points_as_spheres=True,
                    render=False,
                )
            )
            self._node_poly[g.id] = pdata

            for up in g.usage_paths:
                key = f"{g.id}|{up.usage[0]}->{up.usage[1]}"
                if not self.usage_on.get(key, False):
                    continue
                chunks = []
                for sid, rev in zip(up.segments, up.reverse):
                    pts = xyz[sid]
                    if rev:
                        pts = pts[::-1]
                    if chunks:
                        pts = pts[1:]
                    chunks.append(pts)
                if not chunks:
                    continue
                path = np.concatenate(chunks, axis=0)
                if path.shape[0] < 2:
                    continue
                line_actor = self.plotter.add_mesh(
                    pv.lines_from_points(path),
                    color=col,
                    line_width=5,
                    name=f"u:{key}",
                    render=False,
                )
                actors.append(line_actor)
                self._usage_actors.append((key, line_actor))
                step = max(1, (path.shape[0] - 1) // 4)
                tip = max(float(self.resolution) * 0.4, 0.12)
                ai = 0
                for i in range(0, path.shape[0] - 1, step):
                    d = path[i + 1] - path[i]
                    if float(np.linalg.norm(d)) < 1e-9:
                        continue
                    arr = self.plotter.add_mesh(
                        pv.Arrow(
                            start=path[i],
                            direction=d,
                            tip_length=0.35,
                            tip_radius=0.12,
                            shaft_radius=0.04,
                            scale=tip,
                        ),
                        color=col,
                        name=f"u:{key}:a{ai}",
                        render=False,
                    )
                    actors.append(arr)
                    self._usage_actors.append((key, arr))
                    ai += 1

            vis = bool(self.group_on.get(gid, True))
            for a in actors:
                a.SetVisibility(vis)  # type: ignore
            self._group_actors[gid] = actors

        obs_meshes = []
        for o in self.obstacles:
            if isinstance(o, SphereObs):
                obs_meshes.append(pv.Sphere(radius=float(o.r), center=np.asarray(o.c, float)))
            elif isinstance(o, CapsuleObs):
                a, b = np.asarray(o.a, float), np.asarray(o.b, float)
                obs_meshes.append(
                    pv.lines_from_points(np.vstack([a, b])).tube(radius=float(o.r))
                )
            elif isinstance(o, AabbObs):
                mn, mx = np.asarray(o.mn, float), np.asarray(o.mx, float)
                obs_meshes.append(
                    pv.Box(
                        bounds=(
                            float(mn[0]), float(mx[0]),
                            float(mn[1]), float(mx[1]),
                            float(mn[2]), float(mx[2]),
                        )
                    )
                )
        self._obs_actor = None
        if obs_meshes:
            self._obs_actor = self.plotter.add_mesh(
                pv.merge(obs_meshes), color="orange", opacity=0.45, render=False
            )
            self._obs_actor.SetVisibility(bool(self.show_obstacles))  # type: ignore
        self.plotter.render()

    def _toggle_obstacles(self, on: bool):
        self.show_obstacles = bool(on)
        if self._obs_actor is not None:
            self._obs_actor.SetVisibility(self.show_obstacles)  # type: ignore
            self.plotter.render()

    def _update_points(self):
        for g in self.graphs:
            pdata = self._node_poly.get(g.id)
            if pdata is None:
                continue
            pdata.points = np.asarray(g.pos, float)  # type: ignore[attr-defined]
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
        clear_kind = self.clear_kind.currentText()
        steps = int(self.steps.value())
        kw = dict(
            method=self.method.currentText(),
            clear_kind=clear_kind,
            union_kind=self.union_kind.currentText(),
            margin=float(self.margin.value()),
            p_max_obs=int(self.p_max_obs.value()),
            p_max_grp=int(self.p_max_grp.value()),
            p_max_union=int(self.p_max_union.value()),
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
        self._worker = _OptWorker(self.graphs, self.obstacles, kw)
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
                self._load_obstacles()
                self._show_report(remain[-1])
            else:
                self._load_graphs_from(self.case_dir)
                self._load_obstacles()
                self._show_report(None)
        except Exception as e:
            QMessageBox.critical(self, "Revert failed", str(e))
            return
        self._set_busy(False)
        self._fill_usage_checks()
        self.redraw()


def main(argv: list[str] | None = None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="pipe_smooth optimize bench viewer")
    parser.add_argument(
        "--save-dir",
        "-o",
        type=Path,
        default=root / "runs",
        help="root for cases (default: pipe_smooth/runs)",
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
