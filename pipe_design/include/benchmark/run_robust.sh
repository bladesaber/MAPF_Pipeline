#!/usr/bin/env bash
# Batch high-level design runs over gen_robust.sh output.
# Run from pipe_design/build (or set PIPE_RUN_DESIGN).
#
# Usage:
#   ./run_robust.sh [BENCH_OUT] [EXPORT_ROOT] [MAX_TIME_MS] [MAX_ITERATIONS]
# Defaults: BENCH_OUT=./bench_out  EXPORT_ROOT=./runs  MAX_TIME_MS=60000  MAX_ITERATIONS=10000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_OUT="${1:-./bench_out}"
EXPORT_ROOT="${2:-./runs}"
MAX_TIME_MS="${3:-60000}"
MAX_ITERATIONS="${4:-10000}"
PIPE_RUN="${PIPE_RUN_DESIGN:-./pipe_run_design}"

if [[ ! -x "${PIPE_RUN}" ]]; then
  for cand in \
      "${SCRIPT_DIR}/../../../build/pipe_run_design" \
      "$(pwd)/pipe_run_design" \
      "$(pwd)/build/pipe_run_design"; do
    if [[ -x "${cand}" ]]; then
      PIPE_RUN="${cand}"
      break
    fi
  done
fi

if [[ ! -x "${PIPE_RUN}" ]]; then
  echo "error: pipe_run_design not found (build it, or set PIPE_RUN_DESIGN)" >&2
  exit 1
fi

for algo in cbs pp pbs; do
  for gen in axis_ray plane_slice; do
    for level in easy middle hard; do
      indir="${BENCH_OUT}/${gen}/${level}"
      [[ -d "${indir}" ]] || continue
      outdir="${EXPORT_ROOT}/${algo}/${gen}/${level}"
      echo "=== ${algo} / ${gen} / ${level} ==="
      "${PIPE_RUN}" \
        --instance_dir "${indir}" \
        --export_dir "${outdir}" \
        --algorithm "${algo}" \
        --max_time_ms "${MAX_TIME_MS}" \
        --max_iterations "${MAX_ITERATIONS}"
    done
  done
done

echo "done. reports under ${EXPORT_ROOT}/<algo>/<generator>/<level>/design_reports.jsonl"
