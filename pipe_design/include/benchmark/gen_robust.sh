#!/usr/bin/env bash
# Robust batch: 4 datasets × {easy,middle,hard} → target NUM success YAMLs each.
#
# Datasets = path generator × obstacle painter:
#   axis_ray    × {pillar_field, corridor_rack}
#   plane_slice × {pillar_field, corridor_rack}
#
# Layout:
#   OUT_DIR/<axis_ray|plane_slice>/<pillar_field|corridor_rack>/<easy|middle|hard>/instance_*.yaml
#
# Rules:
#   - num_groups >= 2; hard > middle > easy (per path generator)
#   - axis_ray groups: 2 / 4 / 5; plane_slice groups: 3 / 6 / 10
#   - each group terminals > 2; hard > middle > easy
#   - axis_ray: init_waypoints_num > attach_waypoints_num, both > 3
#   - axis_ray fails often → fill loop with larger attempt batches until NUM ok
#     (or MAX_FILL_ROUNDS); plane_slice usually one shot
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM] [PATH_GEN] [OBSTACLE] [SHELL_MARGIN]
# PATH_GEN:  axis_ray | plane_slice | all          (default: all)
# OBSTACLE:  pillar_field | corridor_rack | all    (default: all)
# SHELL_MARGIN: optional float; if omitted, main_gen default (0)
# Defaults: OUT_DIR=./bench_out  NUM=100
#
# Run from pipe_design/build (or set MAIN_GEN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-./bench_out}"
NUM="${2:-100}"
PATH_GEN="${3:-all}"
OBSTACLE="${4:-all}"
SHELL_MARGIN="${5:-}"
MAIN_GEN="${MAIN_GEN:-./main_gen}"

# axis_ray: each fill round requests (remaining * MULT) attempts (min NUM).
AXIS_RAY_ATTEMPT_MULT="${AXIS_RAY_ATTEMPT_MULT:-5}"
MAX_FILL_ROUNDS="${MAX_FILL_ROUNDS:-20}"

case "${PATH_GEN}" in
  axis_ray|plane_slice|all) ;;
  *)
    echo "error: PATH_GEN must be axis_ray | plane_slice | all (got '${PATH_GEN}')" >&2
    echo "usage: $0 [OUT_DIR] [NUM] [PATH_GEN] [OBSTACLE] [SHELL_MARGIN]" >&2
    exit 1
    ;;
esac

case "${OBSTACLE}" in
  pillar_field|corridor_rack|all) ;;
  *)
    echo "error: OBSTACLE must be pillar_field | corridor_rack | all (got '${OBSTACLE}')" >&2
    echo "usage: $0 [OUT_DIR] [NUM] [PATH_GEN] [OBSTACLE] [SHELL_MARGIN]" >&2
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

# Obstacle paint (after corridor + shell). Always on for the four datasets.
# OBSTACLE_TYPE is set per dataset in run_one (pillar_field | corridor_rack).
WITH_OBSTACLE=true
OBSTACLE_DENSITY=0.4                # pillar_field target ρ
CORRIDOR_MARGIN=1                   # Chebyshev keep-out (>=0); -1 → radius_cells
OBSTACLE_MAX_ROUNDS=256             # pillar_field
MIN_EDGE=2
MAX_EDGE=8
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
        --min_edge "${MIN_EDGE}"
        --max_edge "${MAX_EDGE}"
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

# Count success YAMLs currently in dest.
count_yaml() {
  local dest="$1"
  find "${dest}" -maxdepth 1 -name 'instance_*.yaml' 2>/dev/null | wc -l | tr -d ' '
}

# Move staging successes into dest as contiguous instance_${have}.yaml …
ingest_staging() {
  local staging="$1" dest="$2"
  local have="$3"
  local idx f
  # Numeric sort on the instance index (GNU find -printf).
  for idx in $(find "${staging}" -maxdepth 1 -name 'instance_*.yaml' -printf '%f\n' \
      | sed -n 's/^instance_\([0-9][0-9]*\)\.yaml$/\1/p' | sort -n); do
    if (( have >= NUM )); then
      break
    fi
    f="${staging}/instance_${idx}.yaml"
    mv -f "${f}" "${dest}/instance_${have}.yaml"
    have=$((have + 1))
  done
  echo "${have}"
}

# Generate one (path_gen, obstacle, level) until NUM success files or rounds exhausted.
# Args: gen obst level seed_base  [main_gen path-gen flags...]
run_one() {
  local gen="$1" obst="$2" level="$3" seed_base="$4"
  shift 4
  OBSTACLE_TYPE="${obst}"

  local dest="${OUT_DIR}/${gen}/${obst}/${level}"
  mkdir -p "${dest}"
  # Fresh dataset dir (avoid mixing old runs).
  find "${dest}" -maxdepth 1 -name 'instance_*.yaml' -delete 2>/dev/null || true

  local shell_args=()
  if [[ -n "${SHELL_MARGIN:-}" ]]; then
    shell_args=(--shell_margin "${SHELL_MARGIN}")
  fi
  local obst_args=()
  mapfile -t obst_args < <(obstacle_args)

  log "=== ${gen} / ${obst} / ${level}  seed0=${seed_base}  target=${NUM}  out=${dest} ==="
  log "obstacle=${OBSTACLE_TYPE}  corridor_margin=${CORRIDOR_MARGIN}  density=${OBSTACLE_DENSITY}  dilate_steps=${DILATE_STEPS}"

  local have=0
  local seed="${seed_base}"
  local round=0
  local staging="${dest}/.staging"
  local last_rc=0

  while (( have < NUM && round < MAX_FILL_ROUNDS )); do
    round=$((round + 1))
    local remaining=$((NUM - have))
    local batch="${remaining}"
    if [[ "${gen}" == "axis_ray" ]]; then
      batch=$((remaining * AXIS_RAY_ATTEMPT_MULT))
      if (( batch < NUM )); then
        batch="${NUM}"
      fi
    fi

    rm -rf "${staging}"
    mkdir -p "${staging}"

    log "  fill round=${round}  have=${have}/${NUM}  batch_attempts=${batch}  seed=${seed}"
    set +e
    "${MAIN_GEN}" --generator "${gen}" --out "${staging}" --num "${batch}" --seed "${seed}" \
      "$@" "${shell_args[@]}" "${obst_args[@]}"
    last_rc=$?
    set -e

    have="$(ingest_staging "${staging}" "${dest}" "${have}")"
    rm -rf "${staging}"
    seed=$((seed + batch))
    log "  after round: yaml_count=${have}/${NUM}  main_gen_exit=${last_rc}"
  done

  if (( have < NUM )); then
    log "WARNING: only ${have}/${NUM} instances for ${gen}/${obst}/${level} after ${round} rounds"
  fi
  log ""
  return 0
}

# --- path generators (difficulty knobs) × current OBSTACLE_TYPE via run_dataset ---

run_axis_ray_levels() {
  local obst="$1"
  # easy: 3 groups × 3 terminals
  run_one axis_ray "${obst}" easy 3000 \
    --size "${SIZE}" --radius 0.5 --num_groups 3 --terminals 3,3,3 \
    --init_waypoints_num 4 --attach_waypoints_num 3 \
    --min_path_size 6 --max_path_size 15 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128

  # middle: 4 groups × 4 terminals
  run_one axis_ray "${obst}" middle 4000 \
    --size "${SIZE}" --radius 0.5 --num_groups 4 --terminals 4,4,4,4 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128

  # hard: 5 groups × 5 terminals
  run_one axis_ray "${obst}" hard 5000 \
    --size "${SIZE}" --radius 0.5 --num_groups 5 --terminals 5,5,5,5,5 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 6 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128
}

run_plane_slice_levels() {
  local obst="$1"
  # easy: 3 groups × 3 terminals
  run_one plane_slice "${obst}" easy 11000 \
    --size "${SIZE}" --radius 0.5 --num_groups 3 --terminals 3,3,3 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # middle: 6 groups × 6 terminals
  run_one plane_slice "${obst}" middle 12000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 6,6,6,6,6,6 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # hard: 10 groups × 10 terminals
  run_one plane_slice "${obst}" hard 13000 \
    --size "${SIZE}" --radius 0.5 --num_groups 10 --terminals 10,10,10,10,10,10,10,10,10,10 \
    --subdomain_n 4 --plane_gap 3 --max_iterations 256
}

run_dataset() {
  local gen="$1" obst="$2"
  case "${gen}" in
    axis_ray)    run_axis_ray_levels "${obst}" ;;
    plane_slice) run_plane_slice_levels "${obst}" ;;
  esac
}

# Resolve PATH_GEN × OBSTACLE → dataset list.
PATH_GENS=()
case "${PATH_GEN}" in
  all) PATH_GENS=(axis_ray plane_slice) ;;
  *)   PATH_GENS=("${PATH_GEN}") ;;
esac

OBSTACLES=()
case "${OBSTACLE}" in
  all) OBSTACLES=(pillar_field corridor_rack) ;;
  *)   OBSTACLES=("${OBSTACLE}") ;;
esac

log "path_gen=${PATH_GEN}  obstacle=${OBSTACLE}  out=${OUT_DIR}  target_num=${NUM}"
log "shell_margin=${SHELL_MARGIN:-(default)}  axis_ray_mult=${AXIS_RAY_ATTEMPT_MULT}  max_fill_rounds=${MAX_FILL_ROUNDS}"
log "pillar density=${OBSTACLE_DENSITY}  corridor_margin=${CORRIDOR_MARGIN}  dilate_steps=${DILATE_STEPS}"
log ""

for gen in "${PATH_GENS[@]}"; do
  for obst in "${OBSTACLES[@]}"; do
    log "######## dataset ${gen} × ${obst} ########"
    run_dataset "${gen}" "${obst}"
  done
done

log "done. summary: ${SUMMARY}"
