#!/usr/bin/env python3
"""Low-level search bench: run all tasks under a batch tree and write all_reports.csv.

Scope: SearchInstance / SearchReport only (pipe_design low-level).
Not for high-level / multi-group DesignInstance (see pipe_design/ROADMAP.md).

Batch layout (from gen/batch_movingai_grids.py):
  <batch-root>/<model_id>__<heap_id>__<heuristic_id>/<map>/
    instance.json
    reports.jsonl          # after pipe_run_low

For env-normalized multi-factor analysis (OLS weights, …), see factor_compare.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

FACTORS = ("model_id", "heap_id", "heuristic_id")
_SKIP_KEYS = frozenset({"path", "version"})

DEFAULT_RUNNER = (
    Path(__file__).resolve().parent.parent.parent / "build" / "tools" / "pipe_run_low"
)


def collect_frame(batch_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for inst_path in sorted(batch_root.rglob("instance.json")):
        inst = json.loads(inst_path.read_text())
        reports_path = inst_path.parent / "reports.jsonl"
        if not reports_path.is_file():
            print(f"skip (no reports.jsonl): {inst_path.parent}")
            continue
        fallback_ids = {k: inst.get(k, "") for k in FACTORS}
        for line in reports_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rep = json.loads(line)
            row = {k: v for k, v in rep.items() if k not in _SKIP_KEYS}
            for k in FACTORS:
                if not row.get(k):
                    row[k] = fallback_ids[k]
            row["instance_dir"] = str(inst_path.parent)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no reports found under {batch_root}")
    return pd.DataFrame(rows)


def run_all(batch_root: Path, runner: Path) -> None:
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")
    for inst_path in sorted(batch_root.rglob("instance.json")):
        cmd = [str(runner), "--instance", str(inst_path)]
        print("run:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="root from batch_movingai_grids (--output-dir)",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="run pipe_run_low on each instance.json before collecting",
    )
    ap.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help=f"pipe_run_low path (default: {DEFAULT_RUNNER})",
    )
    ap.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="output CSV path (default: <batch-dir>/bench/all_reports.csv)",
    )
    args = ap.parse_args()

    batch_root = args.batch_dir.resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"batch dir not found: {batch_root}")

    if args.run:
        run_all(batch_root, args.runner)

    df = collect_frame(batch_root)
    out_file = (args.out_file or (batch_root / "bench" / "all_reports.csv")).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"wrote {out_file} ({len(df)} rows)")


if __name__ == "__main__":
    main()
