#!/usr/bin/env bash
# Robust batch for pipe_smooth Stage-1 eval: plane_slice × {easy,middle,hard}.
#
# Unlike pipe_design/include/benchmark/gen_robust.sh:
#   - path generator fixed to plane_slice (no axis_ray)
#   - RADIUS is a required-style CLI arg (default larger than pipe_design's 0.5)
#
# Layout:
#   OUT_DIR/plane_slice/<pillar_field|corridor_rack>/<easy|middle|hard>/instance_*.yaml
#
# Rules (same plane_slice knobs as pipe_design):
#   - num_groups >= 2; hard > middle > easy
#   - plane_slice groups×terms: 3 / 8 / 12
#   - each group terminals > 2
#   - low success → fill loop until NUM ok (or MAX_FILL_ROUNDS)
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM] [RADIUS] [OBSTACLE] [SHELL_MARGIN]
# OBSTACLE:  pillar_field | corridor_rack | all    (default: all)
# SHELL_MARGIN: optional float; if omitted, main_gen default (0)
# Defaults: OUT_DIR=<pipe_smooth>/data/bench  NUM=50  RADIUS=1.5
#
# Needs pipe_design main_gen (build it, or set MAIN_GEN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/data/bench}"
NUM="${2:-50}"
RADIUS="${3:-1.5}"
OBSTACLE="${4:-all}"
SHELL_MARGIN="${5:-}"
MAIN_GEN="${MAIN_GEN:-}"

PLANE_SLICE_ATTEMPT_MULT="${PLANE_SLICE_ATTEMPT_MULT:-4}"
MAX_FILL_ROUNDS="${MAX_FILL_ROUNDS:-20}"

case "${OBSTACLE}" in
  pillar_field|corridor_rack|all) ;;
  *)
    echo "error: OBSTACLE must be pillar_field | corridor_rack | all (got '${OBSTACLE}')" >&2
    echo "usage: $0 [OUT_DIR] [NUM] [RADIUS] [OBSTACLE] [SHELL_MARGIN]" >&2
    exit 1
    ;;
esac

# Reject non-numeric / non-positive radius early.
if ! awk -v r="${RADIUS}" 'BEGIN { exit !(r == r && r > 0) }'; then
  echo "error: RADIUS must be a positive float (got '${RADIUS}')" >&2
  exit 1
fi

if [[ -z "${MAIN_GEN}" || ! -x "${MAIN_GEN}" ]]; then
  for cand in \
      "${SCRIPT_DIR}/../pipe_design/build/main_gen" \
      "${SCRIPT_DIR}/../../pipe_design/build/main_gen" \
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
  echo "error: main_gen not found (build pipe_design main_gen, or set MAIN_GEN=/path/to/main_gen)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/robust_summary.txt"
: > "${SUMMARY}"

log() { echo "$@" | tee -a "${SUMMARY}"; }

# Larger default grid than pipe_design (r=0.5 / size=32): continuous tubes need room.
SIZE="${SIZE:-48}"
TERMINAL_GAP="${TERMINAL_GAP:-2}"   # slack beyond diameter; terminal_center_gap = diam + this

# Obstacle paint (after corridor + shell).
# CORRIDOR_MARGIN=-1 → main_gen uses radius_cells (scales with RADIUS).
WITH_OBSTACLE=true
OBSTACLE_DENSITY=0.4
CORRIDOR_MARGIN="${CORRIDOR_MARGIN:--1}"
OBSTACLE_MAX_ROUNDS=256
MIN_EDGE=2
MAX_EDGE=8
CANDIDATES_PER_ROUND=32
ISOMETRIC=true
DIST_CAP=16
DILATE_STEPS=2

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

count_yaml() {
  local dest="$1"
  find "${dest}" -maxdepth 1 -name 'instance_*.yaml' 2>/dev/null | wc -l | tr -d ' '
}

ingest_staging() {
  local staging="$1" dest="$2"
  local have="$3"
  local idx f
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

# Generate one (obstacle, level) until NUM success files or rounds exhausted.
# Args: obst level seed_base  [main_gen plane_slice flags...]
run_one() {
  local obst="$1" level="$2" seed_base="$3"
  shift 3
  OBSTACLE_TYPE="${obst}"
  local gen="plane_slice"

  local dest="${OUT_DIR}/${gen}/${obst}/${level}"
  mkdir -p "${dest}"
  find "${dest}" -maxdepth 1 -name 'instance_*.yaml' -delete 2>/dev/null || true

  local shell_args=()
  if [[ -n "${SHELL_MARGIN:-}" ]]; then
    shell_args=(--shell_margin "${SHELL_MARGIN}")
  fi
  local shared_args=(--terminal_gap "${TERMINAL_GAP}" --radius "${RADIUS}")
  local obst_args=()
  mapfile -t obst_args < <(obstacle_args)

  log "=== ${gen} / ${obst} / ${level}  seed0=${seed_base}  radius=${RADIUS}  target=${NUM}  out=${dest} ==="
  log "size=${SIZE}  terminal_gap=${TERMINAL_GAP}  obstacle=${OBSTACLE_TYPE}  corridor_margin=${CORRIDOR_MARGIN}  density=${OBSTACLE_DENSITY}  dilate_steps=${DILATE_STEPS}"

  local have=0
  local seed="${seed_base}"
  local round=0
  local staging="${dest}/.staging"
  local last_rc=0

  while (( have < NUM && round < MAX_FILL_ROUNDS )); do
    round=$((round + 1))
    local remaining=$((NUM - have))
    local batch="${remaining}"
    local mult="${PLANE_SLICE_ATTEMPT_MULT}"
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

run_plane_slice_levels() {
  local obst="$1"
  # easy: 3 groups × 3 terminals
  run_one "${obst}" easy 11000 \
    --size "${SIZE}" --num_groups 3 --terminals 3,3,3 \
    --subdomain_n 3 --plane_gap 3 --max_iterations 128

  # hard: 8 groups × 8 terminals
  run_one "${obst}" hard 12000 \
    --size "${SIZE}" --num_groups 8 --terminals 8,8,8,8,8,8,8,8 \
    --subdomain_n 5 --plane_gap 3 --max_iterations 128
}

OBSTACLES=()
case "${OBSTACLE}" in
  all) OBSTACLES=(pillar_field corridor_rack) ;;
  *)   OBSTACLES=("${OBSTACLE}") ;;
esac

log "path_gen=plane_slice  obstacle=${OBSTACLE}  radius=${RADIUS}  out=${OUT_DIR}  target_num=${NUM}"
log "shell_margin=${SHELL_MARGIN:-(default)}  plane_slice_mult=${PLANE_SLICE_ATTEMPT_MULT}  max_fill_rounds=${MAX_FILL_ROUNDS}"
log "size=${SIZE}  corridor_margin=${CORRIDOR_MARGIN}  pillar density=${OBSTACLE_DENSITY}  dilate_steps=${DILATE_STEPS}"
log "main_gen=${MAIN_GEN}"
log ""

for obst in "${OBSTACLES[@]}"; do
  log "######## dataset plane_slice × ${obst} ########"
  run_plane_slice_levels "${obst}"
done

log "done. summary: ${SUMMARY}"
