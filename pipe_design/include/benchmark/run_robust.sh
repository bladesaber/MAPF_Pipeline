#!/usr/bin/env bash
# Batch high-level design runs over gen_robust.sh output.
# Run from pipe_design/build (or set PIPE_RUN_DESIGN).
#
# Usage:
#   ./run_robust.sh [BENCH_OUT] [EXPORT_ROOT] [BUDGET_MODE] [BUDGET_VALUE]
# BUDGET_MODE: time | iterations
# BUDGET_VALUE: ms if time, iterations if iterations
# Defaults: BENCH_OUT=./bench_out  EXPORT_ROOT=./runs  BUDGET_MODE=time  BUDGET_VALUE=60000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_OUT="${1:-./bench_out}"
EXPORT_ROOT="${2:-./runs}"
BUDGET_MODE="${3:-time}"
BUDGET_VALUE="${4:-60000}"
PIPE_RUN="${PIPE_RUN_DESIGN:-./pipe_run_design}"

case "${BUDGET_MODE}" in
  time|iterations) ;;
  *)
    echo "error: BUDGET_MODE must be time | iterations (got '${BUDGET_MODE}')" >&2
    exit 1
    ;;
esac

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

budget_args=(--budget_mode "${BUDGET_MODE}")
if [[ "${BUDGET_MODE}" == "time" ]]; then
  budget_args+=(--max_time_ms "${BUDGET_VALUE}")
else
  budget_args+=(--max_iterations "${BUDGET_VALUE}")
fi

for algo in cbs pp pbs; do
  for gen in axis_ray plane_slice; do
    for level in easy middle hard; do
      indir="${BENCH_OUT}/${gen}/${level}"
      [[ -d "${indir}" ]] || continue
      outdir="${EXPORT_ROOT}/${algo}/${gen}/${level}"
      echo "=== ${algo} / ${gen} / ${level}  mode=${BUDGET_MODE} value=${BUDGET_VALUE} ==="
      "${PIPE_RUN}" \
        --instance_dir "${indir}" \
        --export_dir "${outdir}" \
        --algorithm "${algo}" \
        "${budget_args[@]}"
    done
  done
done

echo "done. reports under ${EXPORT_ROOT}/<algo>/<generator>/<level>/design_reports.jsonl"
