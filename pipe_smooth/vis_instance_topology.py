"""Interactive PyQt + PyVista viewer for pipe_smooth topology stages.

Inspect the three pipeline layers produced by ``build_graphs`` on a prepared
env YAML (same input as the optimize pipeline):

  GroupTree       — coarse undirected tree (edges + terminals / branch nodes)
  PartitionedTree — trunk / junction labeled edges after §6.2 partition
  OptimizeGraph   — densified Hermite nodes (red) and segments (blue)

UI
  File → Load env YAML…     load / reload an env; rebuilds trees, parts, graphs
  File → Load obstacles…    load spheres/capsules/aabbs JSON or env YAML
  View → GroupTree | …      switch which layer is drawn (full redraw)
  View → Point | Point+line | Point+tube   edge draw mode
  View → Obstacles          show / hide env obstacles (AABB / capsule / sphere)
  Clip bar                  axis-aligned clip plane (X/Y/Z + slider) + obstacle opacity
  Right toolbox             per-layer checkboxes; toggles actor visibility only
    Groups                  show / hide a whole pipe group
    Labeled edges           (partition) per ``gid|eid`` edge
    UsagePath / Optimize…   optional usage overlays (off by default)

  Partition also has a color mode: per labeled-edge hash, or trunk gray +
  junction colored by complex Z.

CLI
  python -m pipe_smooth.vis_topology [env.yaml] [--resolution R]

  env.yaml       optional path to prepared env; if omitted, use File → Load
  --resolution   Hermite sample spacing (world length) for OptimizeGraph
                 refresh / draw; smaller → denser blue curves (default: 0.1)

Requires a working Qt platform plugin (sets QT_QPA_PLATFORM_PLUGIN_PATH from
the active conda/prefix when needed).
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonDataModel import vtkPlane

from pipe_smooth.collision import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.pipeline import build_graphs, load_obstacles, load_obstacles_file


def _color(key: str) -> tuple[float, float, float]:
    """Stable RGB in [0,1] from string id."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.75, 0.95)
    return float(r), float(g), float(b)


class TopologyViewer(QMainWindow):
    def __init__(self, resolution: float = 0.1):
        super().__init__()
        self.setWindowTitle("pipe_smooth topology viewer")
        self.resize(1400, 900)

        self.resolution = float(resolution)
        self.trees = []
        self.parts = []
        self.graphs = []
        self.domain = None  # (sx, sy, sz) world extents, or None
        self.layer = "tree"  # tree | partition | optimize
        self.part_mode = "labeled"  # labeled | trunk_junc
        self.draw_mode = "point+line"  # point | point+line | point+tube
        self.show_tangents = False
        self.tangent_ratio = 2.0  # arrow length = ||stored v|| * ratio
        self.group_on: dict[str, bool] = {}
        self.part_edge_on: dict[str, bool] = {}  # "gid|eid"
        self.part_usage_on: dict[str, bool] = {}  # "gid|A->B"
        self.opt_usage_on: dict[str, bool] = {}
        self.opt_seg_on: dict[str, bool] = {}  # "gid|sid"
        self.obstacles: list = []
        self.show_obstacles = True
        self.obstacle_opacity = 0.35
        self._actors: dict[str, object] = {}
        self._obs_actors: list[object] = []
        self._domain_actor = None
        self._clip_lo = (0.0, 0.0, 0.0)
        self._clip_extents = (1.0, 1.0, 1.0)
        self._clip_plane = vtkPlane()

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        clip_bar = QHBoxLayout()
        self.clip_cb = QCheckBox("Clip")
        self.clip_cb.setChecked(False)
        self.clip_cb.stateChanged.connect(self._sync_clip)
        self.clip_axis = QComboBox()
        self.clip_axis.addItems(["X", "Y", "Z"])
        self.clip_axis.setFixedWidth(48)
        self.clip_axis.currentIndexChanged.connect(self._sync_clip)
        self.clip_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.clip_slider.setRange(0, 1000)
        self.clip_slider.setValue(500)
        self.clip_slider.setFixedWidth(160)
        self.clip_slider.valueChanged.connect(self._sync_clip)
        self.clip_pos_lbl = QLabel("—")
        self.clip_pos_lbl.setMinimumWidth(64)
        self.obs_opacity_slider = QSlider(Qt.Horizontal)  # type: ignore
        self.obs_opacity_slider.setRange(0, 100)
        self.obs_opacity_slider.setValue(int(round(self.obstacle_opacity * 100)))
        self.obs_opacity_slider.setFixedWidth(120)
        self.obs_opacity_slider.valueChanged.connect(self._set_obstacle_opacity)
        self.obs_opacity_lbl = QLabel(f"{self.obstacle_opacity:.2f}")
        self.obs_opacity_lbl.setMinimumWidth(36)
        clip_bar.addWidget(self.clip_cb)
        clip_bar.addWidget(QLabel("Axis"))
        clip_bar.addWidget(self.clip_axis)
        clip_bar.addWidget(self.clip_slider)
        clip_bar.addWidget(self.clip_pos_lbl)
        clip_bar.addWidget(QLabel("Obs α"))
        clip_bar.addWidget(self.obs_opacity_slider)
        clip_bar.addWidget(self.obs_opacity_lbl)
        clip_bar.addStretch(1)
        root.addLayout(clip_bar, 0)

        split = QSplitter()
        root.addWidget(split, 1)
        # Parent QtInteractor to a host inside the splitter — parenting to
        # QMainWindow leaves an orphan gray widget above the central layout.
        plot_host = QWidget()
        plot_lay = QVBoxLayout(plot_host)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(plot_host)  # type: ignore
        plot_lay.addWidget(self.plotter.interactor)
        split.addWidget(plot_host)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.toolbox = QToolBox()
        rv.addWidget(self.toolbox)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([1100, 300])

        self.box_tree = QWidget()
        self.box_part = QWidget()
        self.box_opt = QWidget()
        self.lay_tree = QVBoxLayout(self.box_tree)
        self.lay_part = QVBoxLayout(self.box_part)
        self.lay_opt = QVBoxLayout(self.box_opt)
        self.toolbox.addItem(self._wrap(self.box_tree), "GroupTree")
        self.toolbox.addItem(self._wrap(self.box_part), "PartitionedTree")
        self.toolbox.addItem(self._wrap(self.box_opt), "OptimizeGraph")

        # partition color mode
        self.lay_part.addWidget(QLabel("Color mode"))
        self.mode_group = QButtonGroup(self)
        for text, mode in (
            ("Labeled edges", "labeled"),
            ("Trunk + junction-by-Z", "trunk_junc"),
        ):
            rb = QRadioButton(text)
            rb.setChecked(mode == "labeled")
            self.mode_group.addButton(rb)
            rb.toggled.connect(lambda on, m=mode: self._set_part_mode(m) if on else None)
            self.lay_part.addWidget(rb)
        self.lay_part.addWidget(QLabel("Groups"))
        self.part_group_box = QVBoxLayout()
        self.lay_part.addLayout(self.part_group_box)
        self.lay_part.addWidget(QLabel("Labeled edges"))
        self.part_edge_box = QVBoxLayout()
        self.lay_part.addLayout(self.part_edge_box)
        self.lay_part.addWidget(QLabel("UsagePath"))
        self.part_usage_box = QVBoxLayout()
        self.lay_part.addLayout(self.part_usage_box)
        self.lay_part.addStretch(1)

        self.lay_tree.addWidget(QLabel("Groups"))
        self.tree_group_box = QVBoxLayout()
        self.lay_tree.addLayout(self.tree_group_box)
        self.lay_tree.addStretch(1)

        self.lay_opt.addWidget(QLabel("Groups"))
        self.opt_group_box = QVBoxLayout()
        self.lay_opt.addLayout(self.opt_group_box)
        self.lay_opt.addWidget(QLabel("HermiteSegments"))
        self.opt_seg_box = QVBoxLayout()
        self.lay_opt.addLayout(self.opt_seg_box)
        self.lay_opt.addWidget(QLabel("OptimizeUsagePath"))
        self.opt_usage_box = QVBoxLayout()
        self.lay_opt.addLayout(self.opt_usage_box)
        self.lay_opt.addStretch(1)

        self._build_menu()
        self.plotter.add_axes()
        self.plotter.show_grid()

    @staticmethod
    def _wrap(inner: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        return sc

    def _add_view_settings_menus(self):
        m_view = self.menuBar().addMenu("&View")
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self._view_acts: dict[str, QAction] = {}
        for name, key in (
            ("GroupTree", "tree"),
            ("PartitionedTree", "partition"),
            ("OptimizeGraph", "optimize"),
        ):
            act = QAction(name, self, checkable=True)
            act.setChecked(key == "tree")
            act.triggered.connect(lambda _=False, k=key: self._set_layer(k))
            grp.addAction(act)
            m_view.addAction(act)
            self._view_acts[key] = act
        m_view.addSeparator()
        mode_grp = QActionGroup(self)
        mode_grp.setExclusive(True)
        for name, key in (
            ("&Point", "point"),
            ("Point + &line", "point+line"),
            ("Point + t&ube", "point+tube"),
        ):
            act = QAction(name, self, checkable=True)
            act.setChecked(key == self.draw_mode)
            act.triggered.connect(lambda _=False, k=key: self._set_draw_mode(k))
            mode_grp.addAction(act)
            m_view.addAction(act)
        m_view.addSeparator()
        act_t = QAction("Show Hermite &tangents", self, checkable=True)
        act_t.setChecked(self.show_tangents)
        act_t.toggled.connect(self._set_show_tangents)
        m_view.addAction(act_t)
        act_o = QAction("&Obstacles", self, checkable=True)
        act_o.setChecked(self.show_obstacles)
        act_o.toggled.connect(self._set_show_obstacles)
        m_view.addAction(act_o)
        self._act_obstacles = act_o

        m_set = self.menuBar().addMenu("&Settings")
        m_set.addAction("Tangent scale &ratio…").triggered.connect(
            self._edit_tangent_ratio
        )

    def _set_draw_mode(self, mode: str):
        self.draw_mode = mode
        self.redraw()

    def _set_show_tangents(self, on: bool):
        self.show_tangents = bool(on)
        if self.layer == "optimize":
            self.redraw()

    def _set_show_obstacles(self, on: bool):
        self.show_obstacles = bool(on)
        for actor in self._obs_actors:
            actor.SetVisibility(self.show_obstacles)  # type: ignore
        self.plotter.render()

    def _set_obstacle_opacity(self, value: int):
        self.obstacle_opacity = float(value) / 100.0
        self.obs_opacity_lbl.setText(f"{self.obstacle_opacity:.2f}")
        for actor in self._obs_actors:
            actor.GetProperty().SetOpacity(self.obstacle_opacity)  # type: ignore
        self.plotter.render()

    def _edit_tangent_ratio(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Tangent scale",
            "display length = ||v|| × ratio",
            self.tangent_ratio,
            0.01,
            100.0,
            2,
        )
        if ok:
            self.tangent_ratio = float(val)
            if self.layer == "optimize" and self.show_tangents:
                self.redraw()

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        act_load = QAction("&Load env YAML…", self)
        act_load.triggered.connect(self.load_env)
        m_file.addAction(act_load)
        act_obs = QAction("Load &obstacles…", self)
        act_obs.triggered.connect(self.load_obstacles_dialog)
        m_file.addAction(act_obs)
        self._add_view_settings_menus()

    def _set_layer(self, layer: str):
        self.layer = layer
        self.toolbox.setCurrentIndex({"tree": 0, "partition": 1, "optimize": 2}[layer])
        self.redraw()

    def _set_part_mode(self, mode: str):
        self.part_mode = mode
        if self.layer == "partition":
            self.redraw()

    def _clear_layout(self, layout: QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _fill_checks(self, layout: QVBoxLayout, keys: list[str], store: dict):
        self._clear_layout(layout)
        for k in keys:
            store.setdefault(k, True)
            cb = QCheckBox(k)
            cb.setChecked(store[k])
            cb.toggled.connect(
                lambda on, key=k, s=store: (s.__setitem__(key, on), self._sync_vis())
            )
            layout.addWidget(cb)

    def _sync_vis(self):
        """Toggle actor visibility from checkbox state — no geometry rebuild."""
        for name, actor in self._actors.items():
            bits = name.split(":", 2)
            kind = bits[0]
            on = True
            if kind in ("tree", "tree_n", "tree_xyz"):
                on = self.group_on.get(bits[1], True)
            elif kind == "part":
                gid, eid = bits[1], bits[2]
                on = self.group_on.get(gid, True) and self.part_edge_on.get(
                    f"{gid}|{eid}", True
                )
            elif kind == "part_n":
                on = self.group_on.get(bits[1], True)
            elif kind == "part_u":
                key = name[len("part_u:") :]
                on = self.group_on.get(key.split("|", 1)[0], True) and self.part_usage_on.get(
                    key, False
                )
            elif kind == "opt":
                on = self.group_on.get(bits[1], True) and self.opt_seg_on.get(
                    f"{bits[1]}|{bits[2]}", True
                )
            elif kind == "opt_t":
                parts = name.split(":")
                gid, sid = parts[1], parts[2]
                on = (
                    self.show_tangents
                    and self.group_on.get(gid, True)
                    and self.opt_seg_on.get(f"{gid}|{sid}", True)
                )
            elif kind in ("opt_n", "opt_v"):
                on = self.group_on.get(bits[1], True)
            elif kind == "opt_u":
                key = name[len("opt_u:") :]
                on = self.group_on.get(key.split("|", 1)[0], True) and self.opt_usage_on.get(
                    key, False
                )
            actor.SetVisibility(bool(on))  # type: ignore
        self.plotter.render()

    def _bind_data(
        self,
        trees,
        parts,
        graphs,
        domain,
        title: str | None = None,
        obstacles=None,
    ):
        self.trees, self.parts, self.graphs, self.domain = trees, parts, graphs, domain
        self.obstacles = list(obstacles or [])
        gids = [t.id for t in self.trees]
        self.group_on = {g: True for g in gids}
        self.part_edge_on = {
            f"{p.id}|{e.id}": True
            for p in self.parts
            for e in list(p.trunk_edges) + list(p.junction_edges)
        }
        self.part_usage_on = {
            f"{p.id}|{up.usage[0]}->{up.usage[1]}": False
            for p in self.parts
            for up in p.usage_paths
        }
        self.opt_usage_on = {
            f"{g.id}|{up.usage[0]}->{up.usage[1]}": False
            for g in self.graphs
            for up in g.usage_paths
        }
        self.opt_seg_on = {
            f"{g.id}|{sid}": True for g in self.graphs for sid in g.segments
        }
        self._fill_checks(self.tree_group_box, gids, self.group_on)
        self._fill_checks(self.part_group_box, gids, self.group_on)
        self._fill_checks(self.opt_group_box, gids, self.group_on)
        self._fill_checks(
            self.part_edge_box, sorted(self.part_edge_on), self.part_edge_on
        )
        self._fill_checks(
            self.part_usage_box, sorted(self.part_usage_on), self.part_usage_on
        )
        self._fill_checks(self.opt_seg_box, sorted(self.opt_seg_on), self.opt_seg_on)
        self._fill_checks(
            self.opt_usage_box, sorted(self.opt_usage_on), self.opt_usage_on
        )
        if title:
            self.setWindowTitle(title)
        if self.domain is not None:
            self._clip_lo = (0.0, 0.0, 0.0)
            self._clip_extents = tuple(float(x) for x in self.domain)  # type: ignore
        self.redraw()
        self.plotter.reset_camera()

    def load_env(self):
        root = Path(__file__).resolve().parents[1]
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load prepared env",
            str(root / "data"),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            env = yaml.safe_load(Path(path).read_text())
            built = build_graphs(env)
            self._bind_data(
                built["trees"],
                built["parts"],
                built["graphs"],
                self._domain_from_env(env),
                title=f"topology — {Path(path).name}",
                obstacles=load_obstacles(env, "capsule"),
            )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def load_obstacles_dialog(self):
        root = Path(__file__).resolve().parents[1]
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load obstacles",
            str(root / "data"),
            "Obstacles (*.json *.yaml *.yml);;All (*)",
        )
        if not path:
            return
        try:
            self.obstacles = load_obstacles_file(path, "capsule")
            self.show_obstacles = True
            if getattr(self, "_act_obstacles", None) is not None:
                self._act_obstacles.blockSignals(True)
                self._act_obstacles.setChecked(True)
                self._act_obstacles.blockSignals(False)
            self.redraw()
        except Exception as e:
            QMessageBox.critical(self, "Load obstacles failed", str(e))

    def redraw(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self._actors = {}
        self._obs_actors = []
        self._domain_actor = None
        self._draw_domain()
        self._draw_obstacles()
        if not self.trees:
            self._refresh_clip_extents()
            self._sync_clip()
            self.plotter.render()
            return
        if self.layer == "tree":
            self._draw_trees()
        elif self.layer == "partition":
            self._draw_parts()
        else:
            self._draw_graphs()
        self._sync_vis()
        self._refresh_clip_extents()
        self._sync_clip()

    def _refresh_clip_extents(self) -> None:
        """Keep clip slider range aligned with domain, or scene bounds if no domain."""
        if self.domain is not None:
            self._clip_lo = (0.0, 0.0, 0.0)
            self._clip_extents = tuple(float(x) for x in self.domain)  # type: ignore
            return
        try:
            b = self.plotter.bounds
        except Exception:
            return
        if b is None or len(b) < 6:
            return
        self._clip_lo = (float(b[0]), float(b[2]), float(b[4]))
        self._clip_extents = (
            max(float(b[1] - b[0]), 1e-6),
            max(float(b[3] - b[2]), 1e-6),
            max(float(b[5] - b[4]), 1e-6),
        )

    def _plane_from_controls(self) -> tuple[list[float], list[float]]:
        """Origin/normal from axis + slider; updates position label."""
        ox, oy, oz = self._clip_lo
        ex, ey, ez = self._clip_extents
        axis = self.clip_axis.currentIndex()
        t = self.clip_slider.value() / 1000.0
        lo = (ox, oy, oz)
        sizes = (ex, ey, ez)
        origin = [ox + ex * 0.5, oy + ey * 0.5, oz + ez * 0.5]
        origin[axis] = lo[axis] + t * sizes[axis]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        self.clip_pos_lbl.setText(f"{'XYZ'[axis]}={origin[axis]:.1f}")
        return origin, normal

    def _sync_clip(self, *_args: object) -> None:
        """Axis-aligned clip on domain / topology / obstacle actors."""
        origin, normal = self._plane_from_controls()
        self._clip_plane.SetOrigin(*origin)
        self._clip_plane.SetNormal(*normal)
        enabled = self.clip_cb.isChecked()
        actors = list(self._actors.values()) + list(self._obs_actors)
        if self._domain_actor is not None:
            actors.append(self._domain_actor)
        for a in actors:
            mapper = a.GetMapper()  # type: ignore
            if mapper is None:
                continue
            mapper.RemoveAllClippingPlanes()
            if enabled:
                mapper.AddClippingPlane(self._clip_plane)
        self.plotter.render()

    def _draw_obstacles(self):
        """Orange translucent obstacle meshes; visibility from View → Obstacles."""
        for i, o in enumerate(self.obstacles):
            name = f"obs:{i}"
            if isinstance(o, SphereObs):
                mesh = pv.Sphere(radius=float(o.r), center=np.asarray(o.c, float))
            elif isinstance(o, CapsuleObs):
                a = np.asarray(o.a, float)
                b = np.asarray(o.b, float)
                line = pv.lines_from_points(np.vstack([a, b]))
                mesh = line.tube(radius=float(o.r))
            elif isinstance(o, AabbObs):
                mn = np.asarray(o.mn, float)
                mx = np.asarray(o.mx, float)
                mesh = pv.Box(
                    bounds=(float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2]))
                )
            else:
                continue
            actor = self.plotter.add_mesh(
                mesh, color="gray", opacity=float(self.obstacle_opacity), name=name
            )
            actor.SetVisibility(bool(self.show_obstacles))  # type: ignore
            self._obs_actors.append(actor)

    @staticmethod
    def _domain_from_env(env: dict):
        grid = env.get("grid") or {}
        if not grid:
            return None
        res = float(grid.get("resolution", 1.0))
        return (
            float(grid["size_x"]) * res,
            float(grid["size_y"]) * res,
            float(grid["size_z"]) * res,
        )

    def _draw_domain(self):
        self._domain_actor = None
        if self.domain is None:
            return
        sx, sy, sz = self.domain
        box = pv.Box(bounds=(0.0, sx, 0.0, sy, 0.0, sz))
        self._domain_actor = self.plotter.add_mesh(
            box, style="wireframe", color="white", line_width=2
        )
        self._clip_lo = (0.0, 0.0, 0.0)
        self._clip_extents = (float(sx), float(sy), float(sz))

    def _poly(self, xyz, color, width=3, name=None, radius=None):
        """Path as line / tube, or skip in point-only mode.

        ``radius``: float or (N,) matching ``xyz`` — used only for point+tube.
        """
        if self.draw_mode == "point":
            return
        xyz = np.asarray(xyz, float)
        if xyz.ndim != 2 or xyz.shape[0] < 2:
            return
        line = pv.lines_from_points(xyz)
        if self.draw_mode == "point+tube":
            r = np.asarray(radius, float).reshape(-1)
            if r.size == 1:
                mesh = line.tube(radius=float(r[0]))
            else:
                line["radius"] = r
                mesh = line.tube(scalars="radius", absolute=True)
            actor = self.plotter.add_mesh(mesh, color=color, name=name)
        else:
            actor = self.plotter.add_mesh(line, color=color, line_width=width, name=name)
        if name is not None:
            self._actors[name] = actor

    def _arrow(self, origin, vel, color, name):
        """Hermite end arrow; world length = ||vel|| * tangent_ratio."""
        vel = np.asarray(vel, float).reshape(3)
        n = float(np.linalg.norm(vel))
        if n < 1e-12:
            return
        actor = self.plotter.add_mesh(
            pv.Arrow(
                start=np.asarray(origin, float).reshape(3),
                direction=vel / n,
                tip_length=0.35,
                tip_radius=0.12,
                shaft_radius=0.04,
                scale=n * self.tangent_ratio,
            ),
            color=color,
            name=name,
        )
        self._actors[name] = actor

    def _draw_trees(self):
        for t in self.trees:
            for e in t.edges:
                self._poly(
                    e.xyz,
                    _color(f"{t.id}:{e.id}"),
                    width=3,
                    name=f"tree:{t.id}:{e.id}",
                    radius=e.radius,
                )
                for i, p in enumerate(e.xyz):
                    actor = self.plotter.add_mesh(
                        pv.Sphere(radius=0.06, center=p),
                        color="red",
                        name=f"tree_xyz:{t.id}:{e.id}:{i}",
                    )
                    self._actors[f"tree_xyz:{t.id}:{e.id}:{i}"] = actor
            for nid, n in t.nodes.items():
                actor = self.plotter.add_mesh(
                    pv.Sphere(
                        radius=0.5,
                        center=n.pos,
                    ),
                    color="blue" if n.terminal_id else "lightgray",
                    opacity=0.9,
                    name=f"tree_n:{t.id}:{nid}",
                )
                self._actors[f"tree_n:{t.id}:{nid}"] = actor

    def _edge_xyz(self, part, edge):
        return np.asarray([part.nodes[nid].pos for nid in edge.nodes], float)

    def _edge_radius(self, part, edge):
        return np.asarray([part.nodes[nid].radius for nid in edge.nodes], float)

    def _draw_parts(self):
        for p in self.parts:
            edge_to_z = {}
            for z in p.complexes:
                for eid in z.edges:
                    edge_to_z[eid] = z.id
            edges = list(p.trunk_edges) + list(p.junction_edges)
            by_id = {e.id: e for e in edges}

            for e in edges:
                xyz = self._edge_xyz(p, e)
                if self.part_mode == "labeled":
                    col = _color(f"{p.id}:{e.id}")
                elif e.kind == "trunk":
                    col = (0.55, 0.55, 0.55)
                else:
                    col = _color(f"{p.id}:{edge_to_z.get(e.id, e.id)}")
                self._poly(
                    xyz, col, width=3, name=f"part:{p.id}:{e.id}", radius=self._edge_radius(p, e)
                )

            for nid, n in p.nodes.items():
                if p.labels.get(nid) == "junction":
                    c = "orange"
                elif n.terminal_id:
                    c = "blue"
                else:
                    c = "dimgray"
                actor = self.plotter.add_mesh(
                    pv.Sphere(radius=0.12, center=n.pos),
                    color=c,
                    opacity=0.85,
                    name=f"part_n:{p.id}:{nid}",
                )
                self._actors[f"part_n:{p.id}:{nid}"] = actor

            for up in p.usage_paths:
                key = f"{p.id}|{up.usage[0]}->{up.usage[1]}"
                pts, rads = [], []
                for eid, rev in zip(up.edges, up.reverse):
                    e = by_id[eid]
                    xyz = self._edge_xyz(p, e)
                    rr = self._edge_radius(p, e)
                    if rev:
                        xyz, rr = xyz[::-1], rr[::-1]
                    if pts:
                        xyz, rr = xyz[1:], rr[1:]
                    pts.append(xyz)
                    rads.append(rr)
                if not pts:
                    continue
                self._poly(
                    np.concatenate(pts, axis=0),
                    _color(key),
                    width=6,
                    name=f"part_u:{key}",
                    radius=np.concatenate(rads, axis=0),
                )

    def _draw_graphs(self):
        from pipe_smooth.geometry.assemble import HermiteSegment

        for g in self.graphs:
            ends = np.asarray(g.hermite_ends(backend="numpy"), dtype=float)
            T = 5
            for s in range(len(g.segments)):
                T = max(
                    T,
                    HermiteSegment.ts_count_for(
                        ends[s, 0], ends[s, 1], ends[s, 2], ends[s, 3], self.resolution
                    ),
                )
            ends, xyz, rad = g.refresh(ts_count=T, backend="numpy", store=True)
            ends = np.asarray(ends, float)
            xyz = np.asarray(xyz, float)
            rad = np.asarray(rad, float)
            for sid in range(len(g.segments)):
                self._poly(
                    xyz[sid],
                    "blue",
                    width=3,
                    name=f"opt:{g.id}:{sid}",
                    radius=rad[sid],
                )
                if self.show_tangents:
                    p0, v0, p1, v1 = ends[sid]
                    self._arrow(p0, v0, "lime", f"opt_t:{g.id}:{sid}:u")
                    self._arrow(p1, v1, "orange", f"opt_t:{g.id}:{sid}:v")
            pos = np.asarray(g.pos, float)
            for nid in range(len(g.nodes)):
                actor = self.plotter.add_mesh(
                    pv.Sphere(radius=0.01, center=pos[nid]),
                    color="red",
                    name=f"opt_n:{g.id}:{nid}",
                )
                self._actors[f"opt_n:{g.id}:{nid}"] = actor

            for up in g.usage_paths:
                key = f"{g.id}|{up.usage[0]}->{up.usage[1]}"
                pts, rads = [], []
                for sid, rev in zip(up.segments, up.reverse):
                    p, rr = xyz[sid], rad[sid]
                    if rev:
                        p, rr = p[::-1], rr[::-1]
                    if pts:
                        p, rr = p[1:], rr[1:]
                    pts.append(p)
                    rads.append(rr)
                if not pts:
                    continue
                self._poly(
                    np.concatenate(pts, axis=0),
                    _color(key),
                    width=6,
                    name=f"opt_u:{key}",
                    radius=np.concatenate(rads, axis=0),
                )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="PyQt + PyVista viewer for pipe_smooth GroupTree / "
        "PartitionedTree / OptimizeGraph.",
    )
    parser.add_argument(
        "env",
        nargs="?",
        type=Path,
        help="prepared env YAML (optional; else File → Load env YAML…)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=0.1,
        help="Hermite sample spacing for OptimizeGraph draw (default: 0.1)",
    )
    args = parser.parse_args(argv)
    if args.resolution <= 0:
        parser.error("--resolution must be > 0")

    # Conda often ships libqxcb.so under $PREFIX/plugins/platforms, while
    # pip PyQt5 looks in site-packages/PyQt5/Qt5/plugins (may lack xcb).
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
    w = TopologyViewer(resolution=args.resolution)
    w.show()
    if args.env is not None:
        env = yaml.safe_load(args.env.read_text())
        built = build_graphs(env)
        w._bind_data(
            built["trees"],
            built["parts"],
            built["graphs"],
            TopologyViewer._domain_from_env(env),
            title=f"topology — {args.env.name}",
            obstacles=load_obstacles(env, "capsule"),
        )
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
