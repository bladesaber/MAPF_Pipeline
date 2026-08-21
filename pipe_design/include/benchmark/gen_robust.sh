#!/usr/bin/env bash
# Robust batch: 2 generators × {easy,middle,hard} × N instances.
# Run from pipe_design/build (or set MAIN_GEN).
#
# Rules:
#   - num_groups >= 2; hard == 6; hard > middle > easy
#   - each group terminals > 2; hard > middle > easy
#   - axis_ray: init_waypoints_num > attach_waypoints_num, both > 3
#     (interior segs = waypoints_num - 2 >= 1 when waypoints_num >= 3; we use > 3)
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM]
# Defaults: OUT_DIR=./bench_out  NUM=100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-./bench_out}"
NUM="${2:-100}"
MAIN_GEN="${MAIN_GEN:-./main_gen}"

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

# Shared across levels (difficulty from groups / terminals / waypoints only).
SIZE=48
PATH_GAP=2
BRANCH_GAP=3

# --- axis_ray ---------------------------------------------------------------
# easy: 2 groups × 3 terminals; init=5 > attach=4 (>3 → ≥1 interior)
run_one axis_ray easy 1000 \
  --size "${SIZE}" --radius 0.5 --num_groups 2 --terminals 3,3 \
  --init_waypoints_num 4 --attach_waypoints_num 3 \
  --min_path_size 2 --max_path_size 8 \
  --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
  --max_attempts 64 --max_iterations 256

# middle: 4 groups × 4 terminals; init=6 > attach=5
run_one axis_ray middle 2000 \
  --size "${SIZE}" --radius 0.5 --num_groups 4 --terminals 4,4,4,4 \
  --init_waypoints_num 5 --attach_waypoints_num 4 \
  --min_path_size 4 --max_path_size 10 \
  --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
  --max_attempts 256 --max_iterations 512

# hard: 6 groups × 5 terminals; init=8 > attach=6
run_one axis_ray hard 3000 \
  --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 5,5,5,5,5,5 \
  --init_waypoints_num 5 --attach_waypoints_num 4 \
  --min_path_size 4 --max_path_size 12 \
  --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
  --max_attempts 256 --max_iterations 512

# --- plane_slice ------------------------------------------------------------
# easy: 2 groups × 3 terminals
run_one plane_slice easy 11000 \
  --size "${SIZE}" --radius 0.5 --num_groups 2 --terminals 3,3 \
  --subdomain_n 3 --plane_gap 2 --max_iterations 256

# middle: 4 groups × 4 terminals
run_one plane_slice middle 12000 \
  --size "${SIZE}" --radius 0.5 --num_groups 4 --terminals 4,4,4,4 \
  --subdomain_n 3 --plane_gap 2 --max_iterations 256

# hard: 6 groups × 5 terminals
run_one plane_slice hard 13000 \
  --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 5,5,5,5,5,5 \
  --subdomain_n 4 --plane_gap 2 --max_iterations 512

log "done. summary: ${SUMMARY}"
