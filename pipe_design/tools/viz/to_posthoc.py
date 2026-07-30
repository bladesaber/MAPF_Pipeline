#!/usr/bin/env python3
"""Convert pipe_run_low posthoc/trace_q*.json → Posthoc .trace.yaml (+ optional .grid).

Docs: https://posthoc.pathfinding.ai/docs/search-trace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# Default Pixel views (search-trace 1.4.0).
VIEWS = {
    "main": [
        {
            "$": "rect",
            "width": 1,
            "height": 1,
            "x": "${{ $.x }}",
            "y": "${{ $.y }}",
            "fill": "${{ colors.green }}",
            "$if": "${{ $.type == 'source' }}",
        },
        {
            "$": "rect",
            "width": 1,
            "height": 1,
            "x": "${{ $.x }}",
            "y": "${{ $.y }}",
            "fill": "${{ colors.blue }}",
            "$if": "${{ $.type == 'expand' }}",
            "clear": "end",
        },
        {
            "$": "rect",
            "width": 1,
            "height": 1,
            "x": "${{ $.x }}",
            "y": "${{ $.y }}",
            "fill": "${{ colors.red }}",
            "$if": "${{ $.type == 'end' }}",
        },
    ]
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--trace", type=Path, required=True, help="posthoc/trace_q*.json")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output .trace.yaml")
    ap.add_argument("--instance", type=Path, default=None, help="SearchInstance JSON for .grid")
    ap.add_argument("--grid-out", type=Path, default=None, help="output .grid path")
    ap.add_argument("--no-grid", action="store_true", help="skip .grid")
    args = ap.parse_args()

    data = json.loads(args.trace.read_text())
    out = args.out or args.trace.with_name(args.trace.stem + ".trace.yaml")

    doc = {
        "version": data.get("version", "1.4.0"),
        "views": VIEWS,
        "pivot": {"x": "${{ $.x + 0.5 }}", "y": "${{ $.y + 0.5 }}", "scale": 0.35},
        "events": data["events"],
    }
    header = (
        "# Posthoc search-trace from pipe_run_low export_posthoc\n"
        f"# instance_id={data.get('instance_id')} map_id={data.get('map_id')} "
        f"query_id={data.get('query_id')}\n"
    )
    out.write_text(header + yaml.dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {out}")

    if args.no_grid:
        return

    inst_path = args.instance or args.trace.parent.parent / "instance.json"
    if not inst_path.is_file():
        print(f"skip .grid: no instance at {inst_path}")
        return

    inst = json.loads(inst_path.read_text())
    sx, sy = int(inst["size"][0]), int(inst["size"][1])
    rows = [["." for _ in range(sx)] for _ in range(sy)]
    for obs in inst.get("obstacles") or []:
        if obs.get("mode") != "sphere":
            continue
        for cell in obs.get("cells") or []:
            x, y = int(cell[0]), int(cell[1])
            if 0 <= x < sx and 0 <= y < sy:
                rows[y][x] = "@"

    map_id = str(inst.get("map_id") or data.get("map_id") or "map")
    grid_out = args.grid_out or out.with_name(map_id.split("/")[-1] + ".grid")
    lines = ["type octile", f"height {sy}", f"width {sx}", "map", *("".join(r) for r in rows)]
    grid_out.write_text("\n".join(lines) + "\n")
    print(f"wrote {grid_out}")


if __name__ == "__main__":
    main()
