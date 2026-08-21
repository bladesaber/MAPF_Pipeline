#!/usr/bin/env bash
# Robust batch: {easy,middle,hard} × N instances for one or both generators.
# Run from pipe_design/build (or set MAIN_GEN).
#
# Rules:
#   - num_groups >= 2; hard > middle > easy (per generator)
#   - axis_ray groups: 2 / 4 / 6; plane_slice groups: 3 / 6 / 9
#   - each group terminals > 2; hard > middle > easy
#   - axis_ray: init_waypoints_num > attach_waypoints_num, both > 3
#     (interior segs = waypoints_num - 2 >= 1 when waypoints_num >= 3; we use > 3)
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM] [GENERATOR]
# GENERATOR: axis_ray | plane_slice | all   (default: all)
# Defaults: OUT_DIR=./bench_out  NUM=100  GENERATOR=all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-./bench_out}"
NUM="${2:-100}"
GENERATOR="${3:-all}"
MAIN_GEN="${MAIN_GEN:-./main_gen}"

case "${GENERATOR}" in
  axis_ray|plane_slice|all) ;;
  *)
    echo "error: GENERATOR must be axis_ray | plane_slice | all (got '${GENERATOR}')" >&2
    echo "usage: $0 [OUT_DIR] [NUM] [GENERATOR]" >&2
    exit 1
    ;;
esac

if [[ ! -x "${MAIN_GEN}" ]]; then
  for cand in \
      "${SCRIPT_DIR}/../../../build/main_gen" \
      "$(pwd)/main_gen" \
      "$(pwd)/build/main_gen" \
      "$(pwd)/pipe_design/build/main_gen"; do
    if [[ -x "${cand}" ]]; then
      MAIN_GEN="${cand}"
      break
    fi
  done
fi

if [[ ! -x "${MAIN_GEN}" ]]; then
  echo "error: main_gen not found (build it, or set MAIN_GEN=/path/to/main_gen)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/robust_summary.txt"
: > "${SUMMARY}"

log() { echo "$@" | tee -a "${SUMMARY}"; }

run_one() {
  local gen="$1" level="$2" seed="$3"
  shift 3
  local dest="${OUT_DIR}/${gen}/${level}"
  mkdir -p "${dest}"
  log "=== ${gen} / ${level}  seed=${seed}  num=${NUM}  out=${dest} ==="
  set +e
  "${MAIN_GEN}" --generator "${gen}" --out "${dest}" --num "${NUM}" --seed "${seed}" "$@"
  local rc=$?
  set -e
  local nfiles
  nfiles="$(find "${dest}" -maxdepth 1 -name 'instance_*.yaml' | wc -l | tr -d ' ')"
  log "exit=${rc}  yaml_count=${nfiles}/${NUM}"
  log ""
  return 0
}

run_axis_ray() {
  # easy: 2 groups × 3 terminals
  run_one axis_ray easy 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 2 --terminals 3,3 \
    --init_waypoints_num 4 --attach_waypoints_num 3 \
    --min_path_size 6 --max_path_size 15 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 256 --max_iterations 512

  # middle: 4 groups × 4 terminals
  run_one axis_ray middle 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 4 --terminals 4,4,4,4 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 512 --max_iterations 1024

  # hard: 6 groups × 5 terminals
  run_one axis_ray hard 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 5,5,5,5,5,5 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 2048
}

run_plane_slice() {
  # easy: 3 groups × 3 terminals
  run_one plane_slice easy 11000 \
    --size "${SIZE}" --radius 0.5 --num_groups 3 --terminals 3,3,3 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 256

  # middle: 6 groups × 4 terminals
  run_one plane_slice middle 12000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 4,4,4,4,4,4 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 512

  # hard: 9 groups × 5 terminals
  run_one plane_slice hard 13000 \
    --size "${SIZE}" --radius 0.5 --num_groups 9 --terminals 6,6,6,6,6,6,6,6,6 \
    --subdomain_n 4 --plane_gap 3 --max_iterations 512
}

# Shared across levels (difficulty from groups / terminals / waypoints only).
# Must keep BRANCH_GAP > PATH_GAP (AxisRayGenerator: branch_center_gap > path_center_gap).
SIZE=30
PATH_GAP=2
BRANCH_GAP=3

log "generator=${GENERATOR}  out=${OUT_DIR}  num=${NUM}"

case "${GENERATOR}" in
  axis_ray)    run_axis_ray ;;
  plane_slice) run_plane_slice ;;
  all)
    run_axis_ray
    run_plane_slice
    ;;
esac

log "done. summary: ${SUMMARY}"
