#!/usr/bin/env python3
"""Load pipe_design instance YAML(s) for pipe_smooth (keep obstacles).

Usages: per group, randomly sample undirected terminal pairs (no reverse
duplicates), then randomly orient each pair. Target count min(n+1, C(n,2));
every terminal used at least once. Requires n >= 2.

Examples:
  python example_load_env.py path/to/instance_0.yaml
  python example_load_env.py path/to/easy/ --seed 0
"""

from __future__ import annotations

import argparse
import random
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data"
DEFAULT_ENV = (
    ROOT.parents[0]
    / "pipe_smooth/data/bench/easy/instance_0.yaml"
)


def load_env(yaml_path: str | Path, seed: int | None = None) -> dict:
    """Read instance YAML. Return meta/grid/groups/paths/usages/obstacles."""
    src = Path(yaml_path).resolve()
    data = yaml.safe_load(src.read_text())
    meta = data.get("meta") or {}
    grid = data.get("grid") or {}
    res = float(grid.get("resolution", meta.get("resolution", 1.0)))
    usage_seed = meta.get("seed", 0) if seed is None else seed
    rng = random.Random(usage_seed)

    raw_obs = data.get("obstacles") or {}
    obstacles = {
        "spheres": list(raw_obs.get("spheres") or []),
        "capsules": list(raw_obs.get("capsules") or []),
        "aabbs": list(raw_obs.get("aabbs") or []),
    }

    groups_out = []
    for g in data.get("groups") or []:
        terms = [
            {
                "id": t["id"],
                "pos": [float(x) for x in t["pos"]],
                "radius": float(t.get("radius", meta.get("radius", 0.5))),
            }
            for t in g.get("terminals") or []
        ]
        groups_out.append({"id": g["id"], "terminals": terms})

    # TODO: add radius to solution paths in the future
    r_meta = float(meta.get("radius", 0.5))
    paths_out = []
    for s in data.get("solution") or []:
        group_paths = []
        for p in s.get("paths") or []:
            cells = p.get("cells") or []
            xyz = [
                [(c[0] + 0.5) * res, (c[1] + 0.5) * res, (c[2] + 0.5) * res]
                for c in cells
            ]
            group_paths.append(
                {
                    "cost": p.get("cost"),
                    "radius": float(p.get("radius", r_meta)),
                    "cells": cells,
                    "xyz": xyz,
                }
            )
        paths_out.append({"id": s["id"], "paths": group_paths})

    usages_out = []
    for g in groups_out:
        ids = [t["id"] for t in g["terminals"]]
        n = len(ids)
        if n < 2:
            raise ValueError(f"group {g['id']}: need n>=2 terminals, got {n}")
        pool = [list(p) for p in combinations(ids, 2)]  # undirected; no (b,a)
        k = min(n + 1, len(pool))
        pairs = None
        for _ in range(256):
            cand = rng.sample(pool, k)
            if set(ids) <= {a for e in cand for a in e}:
                pairs = cand
                break
        if pairs is None:
            raise RuntimeError(f"group {g['id']}: failed to sample covering usages")
        for e in pairs:
            if rng.random() < 0.5:
                e[0], e[1] = e[1], e[0]
        usages_out.append({"id": g["id"], "usages": pairs})

    return {
        "source": str(src),
        "meta": meta,
        "grid": grid,
        "obstacles": obstacles,
        "groups": groups_out,
        "paths": paths_out,
        "usages": usages_out,
        "usage_seed": usage_seed,
    }


def collect_yaml_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        if not files:
            raise FileNotFoundError(f"no .yaml/.yml under {path}")
        return files
    raise FileNotFoundError(path)


def save_env(env: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(env, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load pipe_design env → pipe_smooth/data YAML")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_ENV),
        help="one instance YAML or a folder of YAML files",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(OUT_DIR),
        help="output directory (default: pipe_smooth/data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for usages (default: instance meta.seed)",
    )
    args = parser.parse_args()

    inputs = collect_yaml_inputs(Path(args.input))
    out_dir = Path(args.out_dir)
    for src in inputs:
        env = load_env(src, seed=args.seed)
        dst = out_dir / src.name
        save_env(env, dst)
        print(
            f"wrote {dst}  groups={len(env['groups'])}  "
            f"obs=({len(env['obstacles']['spheres'])},"
            f"{len(env['obstacles']['capsules'])},"
            f"{len(env['obstacles']['aabbs'])})  "
            f"usages={[u['usages'] for u in env['usages']]}"
        )


if __name__ == "__main__":
    main()
