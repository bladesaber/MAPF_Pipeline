"""Summarize collected rows by (method, clear_kind). Path metrics on accepted only."""

from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

from pipe_smooth.analysis.collect import collect

# all runs
ALL_KEYS = ("wall_time", "steps", "JT", "best_J", "rel_drop", "late_std", "uphill_frac",
            "n_hit", "max_shortfall")
# accepted (ok) only — §8.1 soft path metrics
OK_KEYS = ("L", "kr_max", "V", "JT", "best_J")


def _mean(xs: list[float]) -> float | None:
    return stats.mean(xs) if xs else None


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["clear_kind"])].append(row)

    out = []
    for (method, clear_kind), g in sorted(groups.items()):
        ok = [r for r in g if r["ok"]]
        entry = {
            "method": method,
            "clear_kind": clear_kind,
            "n": len(g),
            "n_ok": len(ok),
            "success_rate": len(ok) / len(g) if g else 0.0,
        }
        for k in ALL_KEYS:
            entry[f"mean_{k}"] = _mean([r[k] for r in g])
        for k in OK_KEYS:
            entry[f"mean_{k}_ok"] = _mean([r[k] for r in ok])
        out.append(entry)
    return out


def _print_table(summary: list[dict]) -> None:
    cols = [
        "method", "clear_kind", "n", "n_ok", "success_rate",
        "mean_wall_time", "mean_JT", "mean_n_hit", "mean_max_shortfall",
        "mean_L_ok", "mean_kr_max_ok", "mean_V_ok", "mean_JT_ok",
    ]
    widths = {c: max(len(c), *(len(f"{e.get(c)}"[:12]) for e in summary)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for e in summary:
        cells = []
        for c in cols:
            v = e.get(c)
            if isinstance(v, float):
                s = f"{v:.4g}"
            else:
                s = str(v)
            cells.append(s.ljust(widths[c]))
        print("  ".join(cells))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_dir", type=Path, nargs="?", default=Path("pipe_smooth/runs/hard"))
    ap.add_argument("-o", "--out", type=Path, help="write summary JSON")
    ap.add_argument("--json", action="store_true", help="print JSON instead of table")
    args = ap.parse_args()

    summary = summarize(collect(args.runs_dir))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")
    if args.json or not args.out:
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            _print_table(summary)
