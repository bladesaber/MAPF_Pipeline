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
#   ./gen_robust.sh [OUT_DIR] [NUM] [GENERATOR] [SHELL_MARGIN]
# GENERATOR: axis_ray | plane_slice | all   (default: all)
# SHELL_MARGIN: optional float; if omitted, main_gen default (0)
# Obstacle: edit WITH_OBSTACLE / OBSTACLE_TYPE / … below (default off)
# Defaults: OUT_DIR=./bench_out  NUM=100  GENERATOR=all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-./bench_out}"
NUM="${2:-100}"
GENERATOR="${3:-all}"
SHELL_MARGIN="${4:-}"
MAIN_GEN="${MAIN_GEN:-./main_gen}"

case "${GENERATOR}" in
  axis_ray|plane_slice|all) ;;
  *)
    echo "error: GENERATOR must be axis_ray | plane_slice | all (got '${GENERATOR}')" >&2
    echo "usage: $0 [OUT_DIR] [NUM] [GENERATOR] [SHELL_MARGIN]" >&2
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

# Shared across levels (difficulty from groups / terminals / waypoints only).
# Must keep BRANCH_GAP > PATH_GAP (AxisRayGenerator: branch_center_gap > path_center_gap).
SIZE=32
PATH_GAP=2
BRANCH_GAP=3

# Obstacle paint (after corridor + shell). Set WITH_OBSTACLE=true to enable.
WITH_OBSTACLE=true
OBSTACLE_TYPE=pillar_field          # pillar_field | corridor_rack
OBSTACLE_DENSITY=0.4                # pillar_field target ρ
CORRIDOR_MARGIN=1                   # Chebyshev keep-out (>=0); -1 → radius_cells
OBSTACLE_MAX_ROUNDS=256             # pillar_field
S_MIN=2
CANDIDATES_PER_ROUND=32
ISOMETRIC=true                     # pillar_field isometric expand
DIST_CAP=16
DILATE_STEPS=2                      # corridor_rack k; -1 → CORRIDOR_MARGIN

obstacle_args() {
  if [[ "${WITH_OBSTACLE}" != "true" ]]; then
    return 0
  fi
  local args=(
    --obstacle "${OBSTACLE_TYPE}"
    --corridor_margin "${CORRIDOR_MARGIN}"
    --obstacle_max_rounds "${OBSTACLE_MAX_ROUNDS}"
  )
  case "${OBSTACLE_TYPE}" in
    pillar_field)
      args+=(
        --obstacle_density "${OBSTACLE_DENSITY}"
        --s_min "${S_MIN}"
        --candidates_per_round "${CANDIDATES_PER_ROUND}"
        --dist_cap "${DIST_CAP}"
      )
      if [[ "${ISOMETRIC}" == "true" ]]; then
        args+=(--isometric)
      fi
      ;;
    corridor_rack)
      args+=(--dilate_steps "${DILATE_STEPS}")
      ;;
    *)
      echo "error: OBSTACLE_TYPE must be pillar_field | corridor_rack (got '${OBSTACLE_TYPE}')" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "${args[@]}"
}

run_one() {
  local gen="$1" level="$2" seed="$3"
  shift 3
  local dest="${OUT_DIR}/${gen}/${level}"
  mkdir -p "${dest}"
  local shell_args=()
  if [[ -n "${SHELL_MARGIN:-}" ]]; then
    shell_args=(--shell_margin "${SHELL_MARGIN}")
  fi
  local obst_args=()
  if [[ "${WITH_OBSTACLE}" == "true" ]]; then
    mapfile -t obst_args < <(obstacle_args)
  fi
  log "=== ${gen} / ${level}  seed=${seed}  num=${NUM}  out=${dest} ==="
  if [[ "${WITH_OBSTACLE}" == "true" ]]; then
    log "obstacle=${OBSTACLE_TYPE}  corridor_margin=${CORRIDOR_MARGIN}"
  fi
  set +e
  "${MAIN_GEN}" --generator "${gen}" --out "${dest}" --num "${NUM}" --seed "${seed}" \
    "$@" "${shell_args[@]}" "${obst_args[@]}"
  local rc=$?
  set -e
  local nfiles
  nfiles="$(find "${dest}" -maxdepth 1 -name 'instance_*.yaml' | wc -l | tr -d ' ')"
  log "exit=${rc}  yaml_count=${nfiles}/${NUM}"
  log ""
  return 0
}

# Shared runners (use SIZE / PATH_GAP / BRANCH_GAP / obstacle_* above).
run_axis_ray() {
  # easy: 2 groups × 3 terminals
  run_one axis_ray easy 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 2 --terminals 3,3 \
    --init_waypoints_num 4 --attach_waypoints_num 3 \
    --min_path_size 6 --max_path_size 15 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 256 --max_iterations 128

  # middle: 4 groups × 4 terminals
  run_one axis_ray middle 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 4 --terminals 4,4,4,4 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 512 --max_iterations 128

  # hard: 6 groups × 5 terminals
  run_one axis_ray hard 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 5,5,5,5,5,5 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128
}

run_plane_slice() {
  # easy: 3 groups × 3 terminals
  run_one plane_slice easy 11000 \
    --size "${SIZE}" --radius 0.5 --num_groups 3 --terminals 3,3,3 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # middle: 6 groups × 4 terminals
  run_one plane_slice middle 12000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 4,4,4,4,4,4 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # hard: 9 groups × 5 terminals
  run_one plane_slice hard 13000 \
    --size "${SIZE}" --radius 0.5 --num_groups 9 --terminals 6,6,6,6,6,6,6,6,6 \
    --subdomain_n 4 --plane_gap 3 --max_iterations 128
}

log "generator=${GENERATOR}  out=${OUT_DIR}  num=${NUM}  shell_margin=${SHELL_MARGIN:-(default)}"
log "with_obstacle=${WITH_OBSTACLE}  obstacle_type=${OBSTACLE_TYPE}"

case "${GENERATOR}" in
  axis_ray)    run_axis_ray ;;
  plane_slice) run_plane_slice ;;
  all)
    run_axis_ray
    run_plane_slice
    ;;
esac

log "done. summary: ${SUMMARY}"
