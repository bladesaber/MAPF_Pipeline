#!/usr/bin/env python3
"""One-factor performance comparison on all_reports.csv from compare_search.py.

Factors: model_id, heap_id, heuristic_id
Environment: map_id, query_id

Vary exactly one factor; hold environment + the other two factors fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FACTORS = ("model_id", "heap_id", "heuristic_id")
ENV = ("map_id", "query_id")


def compare_one_factor(df: pd.DataFrame, factor: str, metric: str) -> pd.DataFrame:
    if factor not in FACTORS:
        raise SystemExit(f"--factor must be one of {FACTORS}")
    if metric not in df.columns:
        raise SystemExit(f"metric {metric!r} not in columns: {list(df.columns)}")

    others = [f for f in FACTORS if f != factor]
    keys = list(ENV) + others
    g = (
        df.groupby(keys + [factor], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    wide = g.pivot_table(
        index=keys,
        columns=factor,
        values=metric,
        aggfunc="mean",
    )
    wide.columns = [f"{metric}[{c}]" for c in wide.columns]
    return wide.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="all_reports.csv from compare_search.py",
    )
    ap.add_argument(
        "--factor",
        choices=FACTORS,
        required=True,
        help="vary this factor only",
    )
    ap.add_argument(
        "--metric",
        default="total_time_ms",
        help="metric column to compare (default: total_time_ms)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV (default: alongside input as compare_<factor>_<metric>.csv)",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"not found: {args.csv}")

    df = pd.read_csv(args.csv)
    cmp_df = compare_one_factor(df, args.factor, args.metric)

    out = args.out or args.csv.with_name(f"compare_{args.factor}_{args.metric}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(out, index=False)
    print(f"wrote {out} ({len(cmp_df)} rows)")
    with pd.option_context("display.max_columns", 20, "display.width", 120):
        print(cmp_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
