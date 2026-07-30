#!/usr/bin/env python3
"""Batch MovingAI grids via subprocess → movingai_grids.py.

Sweep = cartesian product of entries in three preset JSON files:
  --models / --heaps / --heuristics  (each a JSON file)
  If omitted, uses search_presets/{models,heaps,heuristics}.json

One task = one (model, heap, heuristic) combo:
  <output-dir>/<model_id>__<heap_id>__<heuristic_id>/
Under each task, movingai_grids writes one subdirectory per map.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent
MOVINGAI = GEN_DIR / "movingai_grids.py"
PRESETS = GEN_DIR / "search_presets"
GRIDS_ROOT = GEN_DIR.parent / "data" / "movingai" / "grids"


def load_presets(path: Path, id_key: str) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"presets file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"bad presets file: {path}")
    out = []
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: preset {name!r} must be an object")
        if id_key not in entry or not entry[id_key]:
            raise SystemExit(f"{path}: preset {name!r} missing {id_key!r}")
        out.append(entry)
    return out


def resolve_maps(suite_dir: Path, maps: list[str] | None) -> list[str]:
    map_dir = suite_dir / "map"
    if not map_dir.is_dir():
        raise SystemExit(f"map dir not found: {map_dir}")
    if maps:
        stems = []
        for name in maps:
            p = Path(name)
            cand = map_dir / (p.name if p.suffix == ".map" else f"{p.name}.map")
            if not cand.is_file():
                raise SystemExit(f"map not found: {cand}")
            stems.append(cand.stem)
        return stems
    paths = sorted(map_dir.glob("*.map"))
    if not paths:
        raise SystemExit(f"no maps in {map_dir}")
    return [p.stem for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, help="suite under grids/, e.g. dao")
    ap.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=None,
        help="map stem (repeatable); default = all maps",
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
        help="root; one task dir per <model_id>__<heap_id>__<heuristic_id>/",
    )
    ap.add_argument(
        "--models",
        type=Path,
        default=None,
        help=f"models JSON file (default: {PRESETS / 'models.json'})",
    )
    ap.add_argument(
        "--heaps",
        type=Path,
        default=None,
        help=f"heaps JSON file (default: {PRESETS / 'heaps.json'})",
    )
    ap.add_argument(
        "--heuristics",
        type=Path,
        default=None,
        help=f"heuristics JSON file (default: {PRESETS / 'heuristics.json'})",
    )
    ap.add_argument("--action-model", default="EightConnected")
    ap.add_argument(
        "--action-model-kwargs",
        default='{"step_cost": 1.0, "turn_weight": 0.0}',
    )
    ap.add_argument("--max-expansions", type=int, default=None)
    ap.add_argument("--max-time-ms", type=float, default=0.0)
    ap.add_argument("--export-closed-set", action="store_true")
    ap.add_argument("--export-posthoc", action="store_true")
    args = ap.parse_args()

    if not MOVINGAI.is_file():
        raise SystemExit(f"missing {MOVINGAI}")

    models_path = args.models or (PRESETS / "models.json")
    heaps_path = args.heaps or (PRESETS / "heaps.json")
    heur_path = args.heuristics or (PRESETS / "heuristics.json")

    models = load_presets(models_path, "model_id")
    heaps = load_presets(heaps_path, "heap_id")
    heuristics = load_presets(heur_path, "heuristic_id")
    map_stems = resolve_maps(args.grids_root / args.suite, args.maps)

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for m, h, he in product(models, heaps, heuristics):
        task_dir = out_root / f"{m['model_id']}__{h['heap_id']}__{he['heuristic_id']}"

        cmd = [
            sys.executable,
            str(MOVINGAI),
            "--suite",
            args.suite,
            "--grids-root",
            str(args.grids_root),
            "--output-dir",
            str(task_dir),
            "--model",
            m["model"],
            "--model-kwargs",
            json.dumps(m.get("model_kwargs", {})),
            "--model-id",
            m["model_id"],
            "--heap",
            h["heap"],
            "--heap-kwargs",
            json.dumps(h.get("heap_kwargs", {})),
            "--heap-id",
            h["heap_id"],
            "--heuristic",
            he["heuristic"],
            "--heuristic-kwargs",
            json.dumps(he["heuristic_kwargs"]),
            "--heuristic-id",
            he["heuristic_id"],
            "--action-model",
            args.action_model,
            "--action-model-kwargs",
            args.action_model_kwargs,
            "--max-time-ms",
            str(args.max_time_ms),
        ]
        for map_stem in map_stems:
            cmd.extend(["--map", map_stem])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.max_expansions is not None:
            cmd.extend(["--max-expansions", str(args.max_expansions)])
        if args.export_closed_set:
            cmd.append("--export-closed-set")
        if args.export_posthoc:
            cmd.append("--export-posthoc")

        print("run:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
