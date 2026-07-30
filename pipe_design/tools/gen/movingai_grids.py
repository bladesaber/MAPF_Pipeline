#!/usr/bin/env python3
"""MovingAI 2D grids → SearchInstance JSON.

Format: https://movingai.com/benchmarks/formats.html

One map → one instance → one output directory:
  <output-dir>/<map>/instance.json
  SearchInstance.output_dir == <output-dir>/<map>/
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Passable terrain in MovingAI maps; all other chars become obstacles.
# (W is "not passable from terrain" — treat as blocked for binary grid search.)
PASSABLE = set(".GS")

GRIDS_ROOT = Path(__file__).resolve().parent.parent / "data" / "movingai" / "grids"


def load_map(path: Path) -> tuple[int, int, list[list[str]]]:
    """Return (width, height, rows) with rows[y][x]; (0,0) = upper-left."""
    lines = path.read_text().splitlines()
    if len(lines) < 4 or lines[0].strip() != "type octile":
        raise ValueError(f"bad map header: {path}")
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    if lines[3].strip() != "map":
        raise ValueError(f"missing 'map' line: {path}")
    rows = [list(line.rstrip("\n")) for line in lines[4 : 4 + height]]
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError(f"map size mismatch: {path}")
    return width, height, rows


def load_scen(path: Path) -> list[dict]:
    """Parse version-1 scenario lines → list of query dicts."""
    lines = path.read_text().splitlines()
    if not lines or not lines[0].lower().startswith("version"):
        raise ValueError(f"bad scen header: {path}")
    queries = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"bad scen line ({path}): {line}")
        queries.append(
            {
                "start_x": int(parts[4]),
                "start_y": int(parts[5]),
                "goal_x": int(parts[6]),
                "goal_y": int(parts[7]),
                "opt_cost": float(parts[8]),
            }
        )
    return queries


def parse_json_obj(text: str, flag: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{flag}: invalid JSON ({e})") from e
    if not isinstance(obj, dict):
        raise SystemExit(f"{flag}: expected a JSON object")
    return obj


def build_instance(
    suite: str,
    map_name: str,
    width: int,
    height: int,
    rows: list[list[str]],
    scen_queries: list[dict],
    cfg: dict,
) -> dict:
    """Build one SearchInstance dict. cfg['output_dir'] is this instance's dir."""
    blocked = [
        [x, y, 0]
        for y in range(height)
        for x in range(width)
        if rows[y][x] not in PASSABLE
    ]
    queries = [
        {
            "query_id": i,
            "start": [q["start_x"], q["start_y"], 0],
            "goal": [q["goal_x"], q["goal_y"], 0],
            "opt_cost": q["opt_cost"],
        }
        for i, q in enumerate(scen_queries)
    ]
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "instance_id": f"{suite}/{map_name}",
        "map_id": f"{suite}/{map_name}",
        "size": [width, height, 1],
        "resolution": 1.0,
        "clearance_cap": 1.0,
        "pipe_radius": 0.1,
        "action_model": cfg["action_model"],
        "action_model_kwargs": cfg["action_model_kwargs"],
        "heuristic": cfg["heuristic"],
        "heuristic_kwargs": cfg["heuristic_kwargs"],
        "heap": cfg["heap"],
        "heap_kwargs": cfg["heap_kwargs"],
        "model": cfg["model"],
        "model_kwargs": cfg["model_kwargs"],
        "model_id": cfg.get("model_id") or "",
        "heap_id": cfg.get("heap_id") or "",
        "heuristic_id": cfg.get("heuristic_id") or "",
        "max_expansions": cfg["max_expansions"],
        "max_time_ms": cfg["max_time_ms"],
        "output_dir": cfg["output_dir"],
        "export_closed_set": cfg["export_closed_set"],
        "export_posthoc": cfg["export_posthoc"],
        "obstacles": (
            [{"mode": "sphere", "cell_radius": 0.1, "cells": blocked}] if blocked else []
        ),
        "queries": queries,
    }


def convert_one(
    suite_dir: Path, map_path: Path, limit: int | None, instance_dir: Path, cfg: dict
) -> Path:
    """Write one map → <instance_dir>/instance.json; output_dir == instance_dir."""
    map_name = map_path.stem
    scen_path = suite_dir / "scen" / f"{map_path.name}.scen"
    if not scen_path.is_file():
        raise FileNotFoundError(f"missing scen: {scen_path}")

    width, height, rows = load_map(map_path)
    scen_queries = load_scen(scen_path)
    if limit is not None:
        scen_queries = scen_queries[:limit]

    instance_dir = instance_dir.resolve()
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True)

    cfg_one = {**cfg, "output_dir": str(instance_dir)}
    inst = build_instance(
        suite=suite_dir.name,
        map_name=map_name,
        width=width,
        height=height,
        rows=rows,
        scen_queries=scen_queries,
        cfg=cfg_one,
    )
    out_path = instance_dir / "instance.json"
    out_path.write_text(json.dumps(inst, indent=2) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, help="suite under grids/, e.g. dao")
    ap.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=None,
        help="map stem or file name (repeatable); default = all maps in suite",
    )
    ap.add_argument("--limit", type=int, default=None, help="max scen queries per map")
    ap.add_argument(
        "--grids-root",
        type=Path,
        default=GRIDS_ROOT,
        help=f"grids root (default: {GRIDS_ROOT})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="root; creates <output-dir>/<map>/ per map "
        "(instance.json + SearchInstance.output_dir)",
    )
    ap.add_argument("--model", default="astar", help='e.g. "astar" | "directed_astar"')
    ap.add_argument(
        "--model-kwargs",
        default='{"heuristic_weight": 1.0}',
        help="JSON object for model_kwargs",
    )
    ap.add_argument(
        "--model-id",
        default="",
        help="preset id from search_presets/models.json (batch provenance)",
    )
    ap.add_argument(
        "--heuristic", default="octile", help='e.g. "octile" | "manhattan" | "dijkstra"'
    )
    ap.add_argument(
        "--heuristic-kwargs",
        default='{"step_cost": 1.0}',
        help="JSON object for heuristic_kwargs",
    )
    ap.add_argument(
        "--heuristic-id",
        default="",
        help="preset id from search_presets/heuristics.json (batch provenance)",
    )
    ap.add_argument(
        "--heap",
        default="LazyBinaryHeap",
        help='e.g. "LazyBinaryHeap" | "BinaryHeap" | "PairingHeap" | "BucketHeap"',
    )
    ap.add_argument(
        "--heap-kwargs",
        default="{}",
        help="JSON object for heap_kwargs",
    )
    ap.add_argument(
        "--heap-id",
        default="",
        help="preset id from search_presets/heaps.json (batch provenance)",
    )
    ap.add_argument(
        "--action-model",
        default="EightConnected",
        help='e.g. "EightConnected" | "SixConnected"',
    )
    ap.add_argument(
        "--action-model-kwargs",
        default='{"step_cost": 1.0, "turn_weight": 0.0}',
        help="JSON object for action_model_kwargs",
    )
    ap.add_argument(
        "--max-expansions",
        type=int,
        default=None,
        help="cap on expansions; omit / default = unlimited (JSON null)",
    )
    ap.add_argument(
        "--max-time-ms",
        type=float,
        default=0.0,
        help="cap on sum of step() time in ms; <=0 = unlimited (default 0)",
    )
    ap.add_argument(
        "--export-closed-set",
        action="store_true",
        help="write closed_set/ under each instance output_dir",
    )
    ap.add_argument(
        "--export-posthoc",
        action="store_true",
        help="write posthoc/ under each instance output_dir (requires size_z==1)",
    )
    args = ap.parse_args()

    cfg = {
        "model": args.model,
        "model_kwargs": parse_json_obj(args.model_kwargs, "--model-kwargs"),
        "model_id": args.model_id,
        "heuristic": args.heuristic,
        "heuristic_kwargs": parse_json_obj(args.heuristic_kwargs, "--heuristic-kwargs"),
        "heuristic_id": args.heuristic_id,
        "heap": args.heap,
        "heap_kwargs": parse_json_obj(args.heap_kwargs, "--heap-kwargs"),
        "heap_id": args.heap_id,
        "action_model": args.action_model,
        "action_model_kwargs": parse_json_obj(
            args.action_model_kwargs, "--action-model-kwargs"
        ),
        "max_expansions": args.max_expansions,
        "max_time_ms": args.max_time_ms,
        "export_closed_set": args.export_closed_set,
        "export_posthoc": args.export_posthoc,
    }

    suite_dir = args.grids_root / args.suite
    map_dir = suite_dir / "map"
    if not map_dir.is_dir():
        raise SystemExit(f"map dir not found: {map_dir}")

    if args.maps:
        map_paths = []
        for name in args.maps:
            p = Path(name)
            cand = map_dir / (p.name if p.suffix == ".map" else f"{p.name}.map")
            if not cand.is_file():
                raise SystemExit(f"map not found: {cand}")
            map_paths.append(cand)
    else:
        map_paths = sorted(map_dir.glob("*.map"))

    if not map_paths:
        raise SystemExit(f"no maps in {map_dir}")

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for mp in map_paths:
        # one map → one instance dir under --output-dir
        instance_dir = out_root / mp.stem
        out = convert_one(suite_dir, mp, args.limit, instance_dir, cfg)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
