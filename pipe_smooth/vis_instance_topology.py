"""Interactive PyQt + PyVista viewer for pipe_smooth topology stages.

Inspect the three pipeline layers produced by ``build_graphs`` on a prepared
env YAML (same input as the optimize pipeline):

  GroupTree       — coarse undirected tree (edges + terminals / branch nodes)
  PartitionedTree — trunk / junction labeled edges after §6.2 partition
  OptimizeGraph   — densified Hermite nodes (red) and segments (blue)

UI
  File → Load env YAML…     load / reload an env; rebuilds trees, parts, graphs
  View → GroupTree | …      switch which layer is drawn (full redraw)
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
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from pipe_smooth.pipeline import build_graphs


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
        self.group_on: dict[str, bool] = {}
        self.part_edge_on: dict[str, bool] = {}  # "gid|eid"
        self.part_usage_on: dict[str, bool] = {}  # "gid|A->B"
        self.opt_usage_on: dict[str, bool] = {}
        self._actors: dict[str, object] = {}

        split = QSplitter()
        self.setCentralWidget(split)
        self.plotter = QtInteractor(self)  # type: ignore
        split.addWidget(self.plotter.interactor)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.toolbox = QToolBox()
        rv.addWidget(self.toolbox)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

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

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        act_load = QAction("&Load env YAML…", self)
        act_load.triggered.connect(self.load_env)
        m_file.addAction(act_load)

        m_view = self.menuBar().addMenu("&View")
        grp = QActionGroup(self)
        grp.setExclusive(True)
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
            elif kind in ("opt", "opt_n", "opt_v"):
                on = self.group_on.get(bits[1], True)
            elif kind == "opt_u":
                key = name[len("opt_u:") :]
                on = self.group_on.get(key.split("|", 1)[0], True) and self.opt_usage_on.get(
                    key, False
                )
            actor.SetVisibility(bool(on)) # type: ignore
        self.plotter.render()

    def _bind_data(self, trees, parts, graphs, domain, title: str | None = None):
        self.trees, self.parts, self.graphs, self.domain = trees, parts, graphs, domain
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
        self._fill_checks(self.tree_group_box, gids, self.group_on)
        self._fill_checks(self.part_group_box, gids, self.group_on)
        self._fill_checks(self.opt_group_box, gids, self.group_on)
        self._fill_checks(
            self.part_edge_box, sorted(self.part_edge_on), self.part_edge_on
        )
        self._fill_checks(
            self.part_usage_box, sorted(self.part_usage_on), self.part_usage_on
        )
        self._fill_checks(
            self.opt_usage_box, sorted(self.opt_usage_on), self.opt_usage_on
        )
        if title:
            self.setWindowTitle(title)
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
            )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def redraw(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self._actors = {}
        self._draw_domain()
        if not self.trees:
            self.plotter.render()
            return
        if self.layer == "tree":
            self._draw_trees()
        elif self.layer == "partition":
            self._draw_parts()
        else:
            self._draw_graphs()
        self._sync_vis()

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
        if self.domain is None:
            return
        sx, sy, sz = self.domain
        box = pv.Box(bounds=(0.0, sx, 0.0, sy, 0.0, sz))
        self.plotter.add_mesh(box, style="wireframe", color="white", line_width=2)

    def _poly(self, xyz, color, width=3, name=None):
        """Connected polyline (add_lines needs paired endpoints, not a chain)."""
        xyz = np.asarray(xyz, float)
        if xyz.ndim != 2 or xyz.shape[0] < 2:
            return
        actor = self.plotter.add_mesh(
            pv.lines_from_points(xyz), color=color, line_width=width, name=name
        )
        if name is not None:
            self._actors[name] = actor

    def _draw_trees(self):
        for t in self.trees:
            for e in t.edges:
                self._poly(
                    e.xyz, _color(f"{t.id}:{e.id}"), width=3, name=f"tree:{t.id}:{e.id}"
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
                        radius=max(0.15, 0.35 * float(n.radius)),
                        center=n.pos,
                    ),
                    color="blue" if n.terminal_id else "lightgray",
                    opacity=0.9,
                    name=f"tree_n:{t.id}:{nid}",
                )
                self._actors[f"tree_n:{t.id}:{nid}"] = actor

    def _edge_xyz(self, part, edge):
        return np.asarray([part.nodes[nid].pos for nid in edge.nodes], float)

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
                self._poly(xyz, col, width=3, name=f"part:{p.id}:{e.id}")

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
                pts = []
                for eid, rev in zip(up.edges, up.reverse):
                    e = by_id[eid]
                    xyz = self._edge_xyz(p, e)
                    if rev:
                        xyz = xyz[::-1]
                    if pts:
                        xyz = xyz[1:]
                    pts.append(xyz)
                if not pts:
                    continue
                self._poly(
                    np.concatenate(pts, axis=0),
                    _color(key),
                    width=6,
                    name=f"part_u:{key}",
                )

    def _draw_graphs(self):
        for g in self.graphs:
            # rebuild Hermite caches, then draw curves (not node chords)
            g.refresh(backend="numpy", resolution=self.resolution)
            for sid, seg in g.segments.items():
                self._poly(
                    np.asarray(seg.xyz, float),
                    "blue",
                    width=3,
                    name=f"opt:{g.id}:{sid}",
                )
            for nid, n in g.nodes.items():
                actor = self.plotter.add_mesh(
                    pv.Sphere(radius=0.1, center=n.pos),
                    color="red",
                    name=f"opt_n:{g.id}:{nid}",
                )
                self._actors[f"opt_n:{g.id}:{nid}"] = actor

            for up in g.usage_paths:
                key = f"{g.id}|{up.usage[0]}->{up.usage[1]}"
                pts = []
                for sid, rev in zip(up.segments, up.reverse):
                    xyz = np.asarray(g.segments[sid].xyz, float)
                    if rev:
                        xyz = xyz[::-1]
                    if pts:
                        xyz = xyz[1:]
                    pts.append(xyz)
                if not pts:
                    continue
                self._poly(
                    np.concatenate(pts, axis=0),
                    _color(key),
                    width=6,
                    name=f"opt_u:{key}",
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
        )
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
