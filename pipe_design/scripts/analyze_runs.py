#!/usr/bin/env python3
"""Summarize DesignReport JSONL under pipe_design/data/runs.

Filename: {algo}_{path_gen}_{obstacle}_{level}.jsonl
  e.g. cbs_axis_ray_pillar_field_easy.jsonl

Usage:
  python analyze_runs.py
  python analyze_runs.py /path/to/runs
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ALGOS = ("cbs", "pbs", "pp")
LEVELS = ("easy", "middle", "hard")
PATH_GENS = ("axis_ray", "plane_slice")
OBSTACLES = ("pillar_field", "corridor_rack")


def parse_stem(stem: str) -> tuple[str, str, str, str] | None:
    for algo in ALGOS:
        if not stem.startswith(algo + "_"):
            continue
        rest = stem[len(algo) + 1 :]
        for path_gen in PATH_GENS:
            if not rest.startswith(path_gen + "_"):
                continue
            rest2 = rest[len(path_gen) + 1 :]
            for obst in OBSTACLES:
                if not rest2.startswith(obst + "_"):
                    continue
                level = rest2[len(obst) + 1 :]
                if level in LEVELS:
                    return algo, path_gen, obst, level
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data" / "runs"
    )
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # key -> reports
    batches: dict[tuple[str, str, str, str], list[dict]] = {}
    for path in sorted(root.glob("*.jsonl")):
        parsed = parse_stem(path.stem)
        if parsed is None:
            print(f"[skip] unrecognized name: {path.name}", file=sys.stderr)
            continue
        batches[parsed] = load_jsonl(path)

    print(f"runs_dir={root}")
    print(f"batches={len(batches)}")
    print()

    for obst in OBSTACLES:
        for path_gen in PATH_GENS:
            print("=" * 72)
            print(f"SUCCESS RATE (%)  —  {path_gen} × {obst}")
            print(f"{'level':8} " + " ".join(f"{a:>8}" for a in ALGOS) + f"  {'N':>5}")
            for level in LEVELS:
                cells = []
                n_ref = None
                for algo in ALGOS:
                    reps = batches.get((algo, path_gen, obst, level), [])
                    n = len(reps)
                    n_ref = n if n_ref is None else n_ref
                    if not reps:
                        cells.append(f"{'—':>8}")
                    else:
                        ok = sum(1 for r in reps if r.get("success"))
                        cells.append(f"{100.0 * ok / n:7.1f}%")
                print(f"{level:8} " + " ".join(cells) + f"  {n_ref or 0:5d}")

            print()
            print(f"MEDIAN time_ok (ms)  —  {path_gen} × {obst}")
            print(f"{'level':8} " + " ".join(f"{a:>10}" for a in ALGOS))
            for level in LEVELS:
                cells = []
                for algo in ALGOS:
                    reps = batches.get((algo, path_gen, obst, level), [])
                    times = [
                        float(r.get("total_time_ms") or 0)
                        for r in reps
                        if r.get("success")
                    ]
                    if not times:
                        cells.append(f"{'—':>10}")
                    else:
                        times.sort()
                        cells.append(f"{times[len(times) // 2]:9.1f}")
                print(f"{level:8} " + " ".join(cells))

            print()
            print(f"MEDIAN cost_ratio (success+witness)  —  {path_gen} × {obst}")
            print(f"{'level':8} " + " ".join(f"{a:>10}" for a in ALGOS))
            for level in LEVELS:
                cells = []
                for algo in ALGOS:
                    reps = batches.get((algo, path_gen, obst, level), [])
                    ratios = [
                        float(r["cost_ratio"])
                        for r in reps
                        if r.get("success")
                        and r.get("witness_compared")
                        and r.get("cost_ratio") is not None
                    ]
                    if not ratios:
                        cells.append(f"{'—':>10}")
                    else:
                        ratios.sort()
                        cells.append(f"{ratios[len(ratios) // 2]:9.3f}")
                print(f"{level:8} " + " ".join(cells))

            print()
            print(f"HARD fail_reason  —  {path_gen} × {obst}")
            for algo in ALGOS:
                reps = batches.get((algo, path_gen, obst, "hard"), [])
                fails = Counter(
                    (r.get("fail_reason") or "(empty)")
                    for r in reps
                    if not r.get("success")
                )
                ok = sum(1 for r in reps if r.get("success"))
                print(f"  {algo:3}  ok={ok}/{len(reps)}  fails={dict(fails)}")
            print()

    # Hard time CDF for each dataset
    print("=" * 72)
    print("HARD success cumulative by wall time (fails = +inf)")
    for obst in OBSTACLES:
        for path_gen in PATH_GENS:
            print(f"\n--- {path_gen} × {obst} ---")
            for algo in ALGOS:
                reps = batches.get((algo, path_gen, obst, "hard"), [])
                if not reps:
                    continue
                times = []
                for r in reps:
                    if r.get("success"):
                        times.append(float(r.get("total_time_ms") or 0))
                    else:
                        times.append(float("inf"))
                n = len(times)
                parts = []
                for T in (5000, 30000, 60000, 100000):
                    k = sum(1 for t in times if t <= T)
                    parts.append(f"@{T/1000:.0f}s={100*k/n:.1f}%")
                print(f"  {algo:3}  " + "  ".join(parts))

    # Sample config from first available batch
    print()
    print("=" * 72)
    print("CONFIG (from first report of each batch key; expect shared)")
    sample_key = next(iter(batches))
    r0 = batches[sample_key][0]
    print(
        f"  algorithm example key={sample_key}\n"
        f"  topology={r0.get('topology')}  low={r0.get('low_model')}  "
        f"h={r0.get('heuristic')}  action={r0.get('action_model')}\n"
        f"  budget={r0.get('budget_mode')}  max_time_ms={r0.get('max_time_ms')}\n"
        f"  meta={r0.get('meta')}"
    )

    # Per-batch N and max_time for sanity
    print()
    print("PER-BATCH N / max_time_ms / success%")
    print(
        f"{'algo':3} {'path':12} {'obst':14} {'level':6} "
        f"{'N':>4} {'ok%':>7} {'T_ms':>8}"
    )
    for key in sorted(batches.keys(), key=lambda k: (k[2], k[1], k[3], k[0])):
        algo, path_gen, obst, level = key
        reps = batches[key]
        n = len(reps)
        ok = sum(1 for r in reps if r.get("success"))
        tmax = reps[0].get("max_time_ms") if reps else None
        print(
            f"{algo:3} {path_gen:12} {obst:14} {level:6} "
            f"{n:4d} {100*ok/n:6.1f}% {tmax!s:>8}"
        )


if __name__ == "__main__":
    main()
