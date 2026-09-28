"""PyQt + PyVista viewer for exported topology JSON dumps.

Loads one or more of:

  tree_*.json / type=GroupTree
  partitioned_*.json / type=PartitionedTree
  graph_*.json / type=OptimizeGraph

UI mirrors ``vis_instance_topology`` (layer view + toolbox toggles).
File → Load JSON… accepts multi-select.

CLI
  python -m pipe_smooth.vis_topology [file_or_dir ...] [--resolution R]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
)
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.topology.junction import PartitionedTree
from pipe_smooth.topology.tree import GroupTree
from pipe_smooth.vis_instance_topology import TopologyViewer


def _detect_load(path: Path):
    """Return ``(\"tree\"|\"partition\"|\"optimize\", obj)`` from JSON file."""
    path = Path(path)
    raw = json.loads(path.read_text())
    typ = str(raw.get("type") or "")
    name = path.name.lower()
    if typ == "GroupTree" or name.startswith("tree_"):
        return "tree", GroupTree.load(path)
    if typ == "PartitionedTree" or name.startswith("partitioned_"):
        return "partition", PartitionedTree.load(path)
    if typ == "OptimizeGraph" or name.startswith("graph_"):
        return "optimize", OptimizeGraph.load(path)
    raise ValueError(f"unknown topology JSON: {path} (type={typ!r})")


class TopologyJsonViewer(TopologyViewer):
    """Same drawing as env viewer; File menu loads JSON dumps instead of YAML."""

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction("&Load JSON…").triggered.connect(self.load_json)
        self._add_view_settings_menus()

    def redraw(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self._actors = {}
        self._draw_domain()
        if self.layer == "tree":
            self._draw_trees()
        elif self.layer == "partition":
            self._draw_parts()
        else:
            self._draw_graphs()
        self._sync_vis()

    def _bind_data(self, trees, parts, graphs, domain, title: str | None = None):
        self.trees, self.parts, self.graphs, self.domain = trees, parts, graphs, domain
        gids = list(
            dict.fromkeys(
                [t.id for t in self.trees]
                + [p.id for p in self.parts]
                + [g.id for g in self.graphs]
            )
        )
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
        self.redraw()
        self.plotter.reset_camera()

    def _ingest(self, paths: list[Path]):
        trees, parts, graphs = [], [], []
        for path in paths:
            kind, obj = _detect_load(path)
            if kind == "tree":
                trees.append(obj)
            elif kind == "partition":
                parts.append(obj)
            else:
                graphs.append(obj)
        if not (trees or parts or graphs):
            raise ValueError("no GroupTree / PartitionedTree / OptimizeGraph loaded")
        if graphs and not trees and not parts:
            layer = "optimize"
        elif parts and not trees:
            layer = "partition"
        else:
            layer = "tree"
        self.layer = layer
        if hasattr(self, "_view_acts") and layer in self._view_acts:
            self._view_acts[layer].setChecked(True)
        self.toolbox.setCurrentIndex({"tree": 0, "partition": 1, "optimize": 2}[layer])
        names = ", ".join(p.name for p in paths[:3])
        if len(paths) > 3:
            names += f" +{len(paths) - 3}"
        self._bind_data(trees, parts, graphs, None, title=f"topology — {names}")

    def load_json(self):
        root = Path(__file__).resolve().parent / "runs"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load topology JSON",
            str(root if root.is_dir() else Path.home()),
            "JSON (*.json)",
        )
        if not paths:
            return
        try:
            self._ingest([Path(p) for p in paths])
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="View exported GroupTree / PartitionedTree / OptimizeGraph JSON.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON file(s) and/or folder(s) with tree_/partitioned_/graph_ dumps",
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
    w = TopologyJsonViewer(resolution=args.resolution)
    w.show()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(
                sorted(
                    list(p.glob("tree_*.json"))
                    + list(p.glob("partitioned_*.json"))
                    + list(p.glob("graph_*.json"))
                )
            )
        elif p.is_file():
            files.append(p)
    if files:
        try:
            w._ingest(files)
        except Exception as e:
            QMessageBox.critical(w, "Load failed", str(e))

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
