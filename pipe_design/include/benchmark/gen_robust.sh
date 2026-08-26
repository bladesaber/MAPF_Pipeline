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
#   - axis_ray groups×terms: 3 / 6 / 9   (path_gap=branch_gap=1, min_path_size=3)
#   - plane_slice groups×terms: 3 / 8 / 12
#   - each group terminals > 2; hard > middle > easy
#   - axis_ray: init_waypoints_num > attach_waypoints_num, both > 3
#   - low success rates → fill loop retries until NUM ok (or MAX_FILL_ROUNDS);
#     axis_ray uses AXIS_RAY_ATTEMPT_MULT; plane_slice uses PLANE_SLICE_ATTEMPT_MULT
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM] [PATH_GEN] [OBSTACLE] [SHELL_MARGIN]
# PATH_GEN:  axis_ray | plane_slice | all     (default: all → both generators)
# OBSTACLE:  pillar_field | corridor_rack | all    (default: all)
# SHELL_MARGIN: optional float; if omitted, main_gen default (0)
# Defaults: OUT_DIR=./bench_out  NUM=100  PATH_GEN=all
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

# Fill multipliers: each round requests (remaining * MULT) attempts (min NUM).
AXIS_RAY_ATTEMPT_MULT="${AXIS_RAY_ATTEMPT_MULT:-5}"
PLANE_SLICE_ATTEMPT_MULT="${PLANE_SLICE_ATTEMPT_MULT:-4}"
MAX_FILL_ROUNDS="${MAX_FILL_ROUNDS:-20}"

case "${PATH_GEN}" in
  plane_slice|axis_ray|all) ;;
  *)
    echo "error: PATH_GEN must be plane_slice | axis_ray | all (got '${PATH_GEN}')" >&2
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
# Must keep BRANCH_GAP >= PATH_GAP (AxisRayGenerator: branch_center_gap >= path_center_gap).
SIZE=32
PATH_GAP=1
BRANCH_GAP=1
TERMINAL_GAP=2                      # slack beyond diameter; terminal_center_gap = diam + this
                                    #   (main_gen default 2; face terminals Chebyshev clearance)

# Obstacle paint (after corridor + shell). Always on for the four datasets.
# OBSTACLE_TYPE is set per dataset in run_one (pillar_field | corridor_rack).
# Never invade thickened witnesses: corridors stay feasible for radius r.
WITH_OBSTACLE=true                  # false → corridors only (no static paint)
OBSTACLE_DENSITY=0.4                # pillar_field only: target ρ = V_obs / V_eligible0
CORRIDOR_MARGIN=1                   # Chebyshev keep-out around witness (>=0); -1 → radius_cells
OBSTACLE_MAX_ROUNDS=256             # pillar_field only: max seed/commit rounds (stop if stuck)
MIN_EDGE=2                          # pillar_field only: reject AABB if any edge < this (cells)
MAX_EDGE=8                          # pillar_field only: cap any AABB edge during expand (cells)
CANDIDATES_PER_ROUND=32             # pillar_field only: sample this many boxes/round, keep largest
ISOMETRIC=true                      # pillar_field only: true = grow 6 faces together; false = per-face max
DIST_CAP=16                         # pillar_field only: cap Chebyshev dist weight for seed sampling
DILATE_STEPS=2                      # corridor_rack only: dilate witness by k, complement = obstacles
                                    #   (-1 → use CORRIDOR_MARGIN)

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
  local shared_args=(--terminal_gap "${TERMINAL_GAP}")
  local obst_args=()
  mapfile -t obst_args < <(obstacle_args)

  log "=== ${gen} / ${obst} / ${level}  seed0=${seed_base}  target=${NUM}  out=${dest} ==="
  log "terminal_gap=${TERMINAL_GAP}  obstacle=${OBSTACLE_TYPE}  corridor_margin=${CORRIDOR_MARGIN}  density=${OBSTACLE_DENSITY}  dilate_steps=${DILATE_STEPS}"

  local have=0
  local seed="${seed_base}"
  local round=0
  local staging="${dest}/.staging"
  local last_rc=0

  while (( have < NUM && round < MAX_FILL_ROUNDS )); do
    round=$((round + 1))
    local remaining=$((NUM - have))
    local batch="${remaining}"
    local mult=1
    if [[ "${gen}" == "axis_ray" ]]; then
      mult="${AXIS_RAY_ATTEMPT_MULT}"
    elif [[ "${gen}" == "plane_slice" ]]; then
      mult="${PLANE_SLICE_ATTEMPT_MULT}"
    fi
    if (( mult > 1 )); then
      batch=$((remaining * mult))
      if (( batch < NUM )); then
        batch="${NUM}"
      fi
    fi

    rm -rf "${staging}"
    mkdir -p "${staging}"

    log "  fill round=${round}  have=${have}/${NUM}  batch_attempts=${batch}  seed=${seed}"
    set +e
    "${MAIN_GEN}" --generator "${gen}" --out "${staging}" --num "${batch}" --seed "${seed}" \
      "$@" "${shared_args[@]}" "${shell_args[@]}" "${obst_args[@]}"
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
    --min_path_size 3 --max_path_size 15 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128

  # middle: 6 groups × 6 terminals
  run_one axis_ray "${obst}" middle 4000 \
    --size "${SIZE}" --radius 0.5 --num_groups 6 --terminals 6,6,6,6,6,6 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 3 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128

  # hard: 9 groups × 9 terminals
  run_one axis_ray "${obst}" hard 5000 \
    --size "${SIZE}" --radius 0.5 --num_groups 9 --terminals 9,9,9,9,9,9,9,9,9 \
    --init_waypoints_num 5 --attach_waypoints_num 4 \
    --min_path_size 3 --max_path_size 20 \
    --path_gap "${PATH_GAP}" --branch_gap "${BRANCH_GAP}" \
    --max_attempts 1024 --max_iterations 128
}

run_plane_slice_levels() {
  local obst="$1"
  # easy: 3 groups × 3 terminals
  run_one plane_slice "${obst}" easy 11000 \
    --size "${SIZE}" --radius 0.5 --num_groups 3 --terminals 3,3,3 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # middle: 8 groups × 8 terminals
  run_one plane_slice "${obst}" middle 12000 \
    --size "${SIZE}" --radius 0.5 --num_groups 8 --terminals 8,8,8,8,8,8,8,8 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # hard: 12 groups × 12 terminals
  run_one plane_slice "${obst}" hard 13000 \
    --size "${SIZE}" --radius 0.5 --num_groups 12 --terminals 12,12,12,12,12,12,12,12,12,12,12,12 \
    --subdomain_n 4 --plane_gap 3 --max_iterations 256
}

run_dataset() {
  local gen="$1" obst="$2"
  case "${gen}" in
    axis_ray)    run_axis_ray_levels "${obst}" ;;
    plane_slice) run_plane_slice_levels "${obst}" ;;
  esac
}

# Resolve PATH_GEN × OBSTACLE → dataset list (default: both path gens × both obstacles).
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
log "shell_margin=${SHELL_MARGIN:-(default)}  axis_ray_mult=${AXIS_RAY_ATTEMPT_MULT}  plane_slice_mult=${PLANE_SLICE_ATTEMPT_MULT}  max_fill_rounds=${MAX_FILL_ROUNDS}"
log "path_gap=${PATH_GAP}  branch_gap=${BRANCH_GAP}  pillar density=${OBSTACLE_DENSITY}  corridor_margin=${CORRIDOR_MARGIN}  dilate_steps=${DILATE_STEPS}"
log ""

for gen in "${PATH_GENS[@]}"; do
  for obst in "${OBSTACLES[@]}"; do
    log "######## dataset ${gen} × ${obst} ########"
    run_dataset "${gen}" "${obst}"
  done
done

log "done. summary: ${SUMMARY}"
