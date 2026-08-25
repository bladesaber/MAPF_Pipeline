#!/usr/bin/env python3
"""Summarize DesignReport JSONL batches under an export root.

Usage:
  python summarize_reports.py /path/to/runs
  python summarize_reports.py /path/to/runs --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_root", type=Path)
    ap.add_argument("--csv", type=Path, default=None, help="optional CSV path")
    args = ap.parse_args()

    rows = []
    for path in sorted(args.export_root.rglob("design_reports.jsonl")):
        n = ok = 0
        reasons: Counter[str] = Counter()
        times_ok: list[float] = []
        iters_ok: list[int] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                if r.get("success"):
                    ok += 1
                    times_ok.append(float(r.get("total_time_ms") or 0.0))
                    iters_ok.append(int(r.get("iterations") or 0))
                else:
                    reason = r.get("fail_reason") or r.get("status") or "unknown"
                    reasons[reason] += 1

        rel = path.relative_to(args.export_root)
        parts = rel.parts  # algo / gen / level / design_reports.jsonl
        algo = parts[0] if len(parts) >= 1 else "?"
        gen = parts[1] if len(parts) >= 2 else "?"
        level = parts[2] if len(parts) >= 3 else "?"
        rate = (100.0 * ok / n) if n else 0.0
        row = {
            "algorithm": algo,
            "generator": gen,
            "difficulty": level,
            "n": n,
            "success": ok,
            "success_rate_pct": round(rate, 1),
            "fail_reasons": dict(reasons),
            "mean_time_ms_ok": round(sum(times_ok) / len(times_ok), 1) if times_ok else None,
            "mean_iters_ok": round(sum(iters_ok) / len(iters_ok), 1) if iters_ok else None,
            "path": str(rel),
        }
        rows.append(row)

    if not rows:
        print(f"no design_reports.jsonl under {args.export_root}")
        return

    # Table
    print(f"{'algo':<6} {'gen':<12} {'level':<8} {'n':>4} {'ok':>4} {'rate%':>6}  fail_reasons")
    print("-" * 72)
    by_algo: dict[str, list] = defaultdict(list)
    for row in rows:
        fr = ", ".join(f"{k}={v}" for k, v in sorted(row["fail_reasons"].items())) or "-"
        print(
            f"{row['algorithm']:<6} {row['generator']:<12} {row['difficulty']:<8} "
            f"{row['n']:>4} {row['success']:>4} {row['success_rate_pct']:>6.1f}  {fr}"
        )
        by_algo[row["algorithm"]].append(row)

    print()
    print("by algorithm (all levels):")
    for algo, rs in sorted(by_algo.items()):
        n = sum(r["n"] for r in rs)
        ok = sum(r["success"] for r in rs)
        rate = (100.0 * ok / n) if n else 0.0
        print(f"  {algo}: {ok}/{n} = {rate:.1f}%")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "generator",
                    "difficulty",
                    "n",
                    "success",
                    "success_rate_pct",
                    "fail_reasons",
                    "mean_time_ms_ok",
                    "mean_iters_ok",
                    "path",
                ],
            )
            w.writeheader()
            for row in rows:
                out = dict(row)
                out["fail_reasons"] = json.dumps(row["fail_reasons"], sort_keys=True)
                w.writerow(out)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
