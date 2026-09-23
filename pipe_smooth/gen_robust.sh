#!/usr/bin/env bash
# Robust batch for pipe_smooth Stage-1 eval: plane_slice + pillar_field × {easy,hard}.
#
# Unlike pipe_design/include/benchmark/gen_robust.sh:
#   - path generator fixed to plane_slice (no axis_ray)
#   - obstacle fixed to pillar_field
#   - RADIUS is a CLI arg (default larger than pipe_design's 0.5)
#
# Layout:
#   OUT_DIR/{easy,hard}/instance_*.yaml
#
# Rules:
#   - easy: 3 groups × 3 terminals; hard: 8 groups × 8 terminals
#   - low success → fill loop until NUM ok (or MAX_FILL_ROUNDS)
#
# Usage:
#   ./gen_robust.sh [OUT_DIR] [NUM] [RADIUS] [SHELL_MARGIN]
# SHELL_MARGIN: optional float; if omitted, main_gen default (0)
# Defaults: OUT_DIR=<pipe_smooth>/data/bench  NUM=50  RADIUS=1.5
#
# Needs pipe_design main_gen (build it, or set MAIN_GEN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/data/bench}"
NUM="${2:-50}"
RADIUS="${3:-1.5}"
SHELL_MARGIN="${4:-}"
MAIN_GEN="${MAIN_GEN:-}"

PLANE_SLICE_ATTEMPT_MULT="${PLANE_SLICE_ATTEMPT_MULT:-4}"
MAX_FILL_ROUNDS="${MAX_FILL_ROUNDS:-20}"

if ! awk -v r="${RADIUS}" 'BEGIN { exit !(r == r && r > 0) }'; then
  echo "error: RADIUS must be a positive float (got '${RADIUS}')" >&2
  echo "usage: $0 [OUT_DIR] [NUM] [RADIUS] [SHELL_MARGIN]" >&2
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

SIZE="${SIZE:-48}"
TERMINAL_GAP="${TERMINAL_GAP:-2}"
CORRIDOR_MARGIN="${CORRIDOR_MARGIN:--1}"
OBSTACLE_DENSITY=0.2
OBSTACLE_MAX_ROUNDS=256
MIN_EDGE=4
MAX_EDGE=10
CANDIDATES_PER_ROUND=32
ISOMETRIC=false
DIST_CAP=16

obstacle_args() {
  local args=(
    --obstacle pillar_field
    --corridor_margin "${CORRIDOR_MARGIN}"
    --obstacle_max_rounds "${OBSTACLE_MAX_ROUNDS}"
    --obstacle_density "${OBSTACLE_DENSITY}"
    --min_edge "${MIN_EDGE}"
    --max_edge "${MAX_EDGE}"
    --candidates_per_round "${CANDIDATES_PER_ROUND}"
    --dist_cap "${DIST_CAP}"
  )
  if [[ "${ISOMETRIC}" == "true" ]]; then
    args+=(--isometric)
  fi
  printf '%s\n' "${args[@]}"
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

# Args: level seed_base  [main_gen plane_slice flags...]
run_one() {
  local level="$1" seed_base="$2"
  shift 2
  local gen="plane_slice"
  local dest="${OUT_DIR}/${level}"
  mkdir -p "${dest}"
  find "${dest}" -maxdepth 1 -name 'instance_*.yaml' -delete 2>/dev/null || true

  local shell_args=()
  if [[ -n "${SHELL_MARGIN:-}" ]]; then
    shell_args=(--shell_margin "${SHELL_MARGIN}")
  fi
  local shared_args=(--terminal_gap "${TERMINAL_GAP}" --radius "${RADIUS}")
  local obst_args=()
  mapfile -t obst_args < <(obstacle_args)

  log "=== ${level}  (gen=${gen}, obst=pillar_field)  seed0=${seed_base}  radius=${RADIUS}  target=${NUM}  out=${dest} ==="
  log "size=${SIZE}  terminal_gap=${TERMINAL_GAP}  corridor_margin=${CORRIDOR_MARGIN}  density=${OBSTACLE_DENSITY}"

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
    log "WARNING: only ${have}/${NUM} instances for ${level} after ${round} rounds"
  fi
  log ""
  return 0
}

log "path_gen=plane_slice  obstacle=pillar_field  radius=${RADIUS}  out=${OUT_DIR}  target_num=${NUM}"
log "shell_margin=${SHELL_MARGIN:-(default)}  plane_slice_mult=${PLANE_SLICE_ATTEMPT_MULT}  max_fill_rounds=${MAX_FILL_ROUNDS}"
log "size=${SIZE}  corridor_margin=${CORRIDOR_MARGIN}  pillar density=${OBSTACLE_DENSITY}"
log "main_gen=${MAIN_GEN}"
log ""

run_one easy 11000 \
  --size "${SIZE}" --num_groups 3 --terminals 3,3,3 \
  --subdomain_n 3 --plane_gap 3 --max_iterations 128

run_one hard 12000 \
  --size "${SIZE}" --num_groups 8 --terminals 8,8,8,8,8,8,8,8 \
  --subdomain_n 5 --plane_gap 3 --max_iterations 128

log "done. summary: ${SUMMARY}"
