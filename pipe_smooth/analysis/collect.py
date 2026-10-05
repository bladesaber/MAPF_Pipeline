"""Walk ``runs/<set>/<method>/<clear_kind>/instance_*/report.json`` → flat rows."""

from __future__ import annotations

import json
from pathlib import Path


def collect(runs_dir: str | Path) -> list[dict]:
    rows = []
    root = Path(runs_dir)
    for path in sorted(root.glob("*/*/instance_*/report.json")):
        method, clear_kind, instance = path.parts[-4:-1]
        r = json.loads(path.read_text())
        c, p, s = r["clearance"], r["path"]["aggregate"], r["solve"]
        rows.append(
            {
                "method": method,
                "clear_kind": clear_kind,
                "instance": instance,
                "ok": bool(c["ok"]),
                "n_hit": int(c["n_hit"]),
                "n_hit_obs": int(c["n_hit_obs"]),
                "n_hit_grp": int(c["n_hit_grp"]),
                "max_shortfall": float(c["max_shortfall"]),
                "L": float(p["L"]),
                "kr_max": float(p["kr_max"]),
                "V": float(p["V"]),
                "wall_time": float(s["wall_time"]),
                "steps": int(s["steps"]),
                "J0": float(s["J0"]),
                "JT": float(s["JT"]),
                "best_J": float(s["best_J"]),
                "rel_drop": float(s["rel_drop"]),
                "late_std": float(s["late_std"]),
                "uphill_frac": float(s["uphill_frac"]),
                "path": str(path),
            }
        )
    return rows


if __name__ == "__main__":
    import argparse
    import csv
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", type=Path, help="e.g. pipe_smooth/runs/hard")
    ap.add_argument("-o", "--out", type=Path, help="write CSV (default: stdout JSON lines)")
    args = ap.parse_args()
    rows = collect(args.runs_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows → {args.out}", file=sys.stderr)
    else:
        for row in rows:
            print(json.dumps(row))
