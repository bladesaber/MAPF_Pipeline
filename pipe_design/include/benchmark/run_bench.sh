#!/usr/bin/env bash
# Parallel batch for pipe_run_design over data/envs folders.
#
# Dataset layout (from gen_robust.sh):
#   BENCH/<axis_ray|plane_slice>/<pillar_field|corridor_rack>/<easy|middle|hard>/
#
# Edit BENCH_FOLDERS below (comment out lines to skip).
# Parallelism: MAX_PARALLEL jobs at once (default 4). Override:
#   MAX_PARALLEL=6 bash run_bench.sh
#
# Export: OUT/{algo}_{path}_{obstacle}_{level}.jsonl
# Meta: path_generator / obstacle / level / timestamp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DESIGN="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPE_RUN="${PIPE_RUN:-${PIPE_DESIGN}/build/pipe_run_design}"
BENCH="${BENCH:-${PIPE_DESIGN}/data/envs}"
OUT="${OUT:-${PIPE_DESIGN}/data/runs}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "${OUT}"

EASY_TIME_MS=30000
MIDDLE_TIME_MS=60000
HARD_TIME_MS=100000

if [[ ! -x "${PIPE_RUN}" ]]; then
  echo "error: pipe_run_design not found: ${PIPE_RUN}" >&2
  exit 1
fi

# =============================================================================
# Bench folders to run (relative to BENCH). Comment out to skip.
# =============================================================================
BENCH_FOLDERS=(
  axis_ray/pillar_field/easy
  axis_ray/pillar_field/middle
  axis_ray/pillar_field/hard
  axis_ray/corridor_rack/easy
  axis_ray/corridor_rack/middle
  axis_ray/corridor_rack/hard
  plane_slice/pillar_field/easy
  plane_slice/pillar_field/middle
  plane_slice/pillar_field/hard
  plane_slice/corridor_rack/easy
  plane_slice/corridor_rack/middle
  plane_slice/corridor_rack/hard
)

# Algorithms (comment out to skip).
ALGOS=(
  cbs
  pbs
  pp
)

time_ms_for_level() {
  case "$1" in
    easy)   echo "${EASY_TIME_MS}" ;;
    middle) echo "${MIDDLE_TIME_MS}" ;;
    hard)   echo "${HARD_TIME_MS}" ;;
    *)
      echo "error: unknown level '$1'" >&2
      return 1
      ;;
  esac
}

# One (folder × algo) job. Logs to OUT/logs/...
run_job() {
  local folder="$1"
  local algo="$2"

  local path_gen obstacle level
  IFS=/ read -r path_gen obstacle level <<< "${folder}"

  local instance_dir="${BENCH}/${folder}"
  if [[ ! -d "${instance_dir}" ]]; then
    echo "[SKIP] missing dir: ${instance_dir}" >&2
    return 0
  fi

  local time_ms
  time_ms="$(time_ms_for_level "${level}")"

  local tag="${algo}_${path_gen}_${obstacle}_${level}"
  local export_path="${OUT}/${tag}.jsonl"
  local log_path="${OUT}/logs/${tag}.log"
  mkdir -p "${OUT}/logs"

  local meta
  meta="$(printf '{"path_generator":"%s","obstacle":"%s","level":"%s","timestamp":"%s"}' \
    "${path_gen}" "${obstacle}" "${level}" "${TS}")"

  local -a cmd=(
    "${PIPE_RUN}"
    --instance_dir "${instance_dir}"
    --export "${export_path}"
    --algorithm "${algo}"
    --topology pure_mst --radius_merge min --cost_type euclidean
    --heuristic manhattan --action_model TwentySixConnected
    --budget_mode time --max_time_ms "${time_ms}"
    --meta "${meta}"
  )

  case "${algo}" in
    cbs)
      cmd+=(--conflict_analyzer near_terminal)
      ;;
    pbs)
      cmd+=(--conflict_analyzer near_terminal --pbs_mode depth_first)
      ;;
    pp)
      ;;
    *)
      echo "[SKIP] unknown algo: ${algo}" >&2
      return 0
      ;;
  esac

  echo "[START] ${tag}  (budget ${time_ms} ms) → ${export_path}"
  # Isolate failures so one job does not abort the whole parallel batch.
  set +e
  "${cmd[@]}" >"${log_path}" 2>&1
  local rc=$?
  set -e
  if (( rc == 0 )); then
    echo "[DONE]  ${tag}  rc=0"
  else
    echo "[FAIL]  ${tag}  rc=${rc}  see ${log_path}" >&2
  fi
  return 0
}

# Wait until fewer than MAX_PARALLEL background jobs are running.
wait_for_slot() {
  while true; do
    local n
    n="$(jobs -rp | wc -l | tr -d ' ')"
    if (( n < MAX_PARALLEL )); then
      break
    fi
    # bash 4.3+: wait for any child; ignore non-zero from failed jobs
    wait -n || true
  done
}

echo "BENCH=${BENCH}"
echo "OUT=${OUT}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "folders=${#BENCH_FOLDERS[@]}  algos=${#ALGOS[@]}  jobs=$(( ${#BENCH_FOLDERS[@]} * ${#ALGOS[@]} ))"
echo "timestamp=${TS}"
echo ""

for folder in "${BENCH_FOLDERS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    wait_for_slot
    run_job "${folder}" "${algo}" &
  done
done

# Drain remaining jobs
wait || true

echo ""
echo "all jobs finished. JSONL under: ${OUT}"
echo "logs under: ${OUT}/logs"
