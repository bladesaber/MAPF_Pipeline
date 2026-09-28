#!/usr/bin/env bash
# Run Stage-1 pipeline on instance_*.yaml in one or more folders (optionally parallel).
#
# Usage:
#   ./run_instances.sh DATA_DIR [DATA_DIR ...] [options...]
#
# Examples:
#   ./run_instances.sh data/easy
#   ./run_instances.sh data/easy data/hard -j 8 -m adam --clear-kind capsule,ignore
#   ./run_instances.sh data/easy data/hard -j 8 -m all --clear-kind all
#   ./run_instances.sh data/easy -m adam,adam_annealed --coarse-step 30 --refine-step 50
#
# Results → pipe_smooth/runs/<folder>/<method>/<clear_kind>/instance_*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

ALL_METHODS=(adam adam_annealed langevin elastic_qp interior_nlp)
ALL_CLEAR_KINDS=(capsule hull ignore)

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
METHODS=()
CLEAR_KINDS=()
UNION_KIND="${UNION_KIND:-chamfer}"
MARGIN_RATIO="${MARGIN_RATIO:-1.5}"
P_MAX_OBS="${P_MAX_OBS:-128}"
P_MAX_GRP="${P_MAX_GRP:-128}"
P_MAX_UNION="${P_MAX_UNION:-64}"
REFRESH_EVERY="${REFRESH_EVERY:-10}"
TWO_STAGE="${TWO_STAGE:-1}"
COARSE_STEPS="${COARSE_STEPS:-50}"
REFINE_STEPS="${REFINE_STEPS:-50}"
COARSE_STEP_SIZE="${COARSE_STEP_SIZE:-0.1}"
REFINE_STEP_SIZE="${REFINE_STEP_SIZE:-0.01}"

usage() {
  cat <<EOF
usage: $0 DATA_DIR [DATA_DIR ...] [options...]

  DATA_DIR                  one or more folders with instance_*.yaml
                            (resolved relative to cwd or pipe_smooth/)
  -j, --jobs N              parallel workers (default: nproc)
  -m, --method LIST         comma-separated methods, or 'all'
                            (${ALL_METHODS[*]})
                            repeatable: -m adam -m langevin
  --clear-kind LIST         comma-separated, or 'all'
                            (${ALL_CLEAR_KINDS[*]})
                            repeatable: --clear-kind capsule --clear-kind ignore
  --union-kind KIND         chamfer|ignore
  --margin-ratio R          clear pad = R * stage step_size (default 1.5)
  --p-max-obs/--p-max-grp/--p-max-union N
  --refresh-every N
  --two-stage / --no-two-stage
  --coarse-step N           coarse iteration count (default 50)
  --refine-step N           refine iteration count; also used if --no-two-stage
  --coarse-step-size F      coarse lr/delta/trust (default 0.1)
  --refine-step-size F      refine lr/delta/trust (default 0.01)
  -h, --help

Env: JOBS METHOD(S) CLEAR_KIND(S) UNION_KIND MARGIN_RATIO P_MAX_* REFRESH_EVERY
     TWO_STAGE COARSE_STEPS REFINE_STEPS COARSE_STEP_SIZE REFINE_STEP_SIZE PYTHON
EOF
}

append_list() {
  # append_list DEST_VAR ALL_ARRAY_NAME raw
  local -n _dest="$1"
  local -n _all="$2"
  local raw="$3" part
  IFS=',' read -ra parts <<<"${raw}"
  for part in "${parts[@]}"; do
    part="${part// /}"
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == "all" ]]; then
      _dest+=("${_all[@]}")
    else
      _dest+=("${part}")
    fi
  done
}

dedup_validate() {
  # dedup_validate DEST_VAR ALL_ARRAY_NAME label
  local -n _src="$1"
  local -n _all="$2"
  local label="$3"
  local -a _out=()
  local m skip x ok
  for m in "${_src[@]}"; do
    ok=0
    for x in "${_all[@]}"; do
      if [[ "${m}" == "${x}" ]]; then ok=1; break; fi
    done
    if ((ok == 0)); then
      echo "error: unknown ${label} '${m}' (want: ${_all[*]} | all)" >&2
      return 1
    fi
    skip=0
    for x in "${_out[@]+"${_out[@]}"}"; do
      [[ "${x}" == "${m}" ]] && skip=1 && break
    done
    ((skip)) || _out+=("${m}")
  done
  _src=("${_out[@]}")
}

resolve_data_dir() {
  local d="$1"
  if [[ -d "${d}" ]]; then
    (cd "${d}" && pwd)
  elif [[ -d "${SCRIPT_DIR}/${d}" ]]; then
    (cd "${SCRIPT_DIR}/${d}" && pwd)
  else
    return 1
  fi
}

DATA_DIRS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -j|--jobs) JOBS="$2"; shift 2 ;;
    -m|--method) append_list METHODS ALL_METHODS "$2"; shift 2 ;;
    --clear-kind) append_list CLEAR_KINDS ALL_CLEAR_KINDS "$2"; shift 2 ;;
    --union-kind) UNION_KIND="$2"; shift 2 ;;
    --margin-ratio) MARGIN_RATIO="$2"; shift 2 ;;
    --p-max-obs) P_MAX_OBS="$2"; shift 2 ;;
    --p-max-grp) P_MAX_GRP="$2"; shift 2 ;;
    --p-max-union) P_MAX_UNION="$2"; shift 2 ;;
    --refresh-every) REFRESH_EVERY="$2"; shift 2 ;;
    --two-stage) TWO_STAGE=1; shift ;;
    --no-two-stage) TWO_STAGE=0; shift ;;
    --coarse-step) COARSE_STEPS="$2"; shift 2 ;;
    --refine-step) REFINE_STEPS="$2"; shift 2 ;;
    --coarse-step-size) COARSE_STEP_SIZE="$2"; shift 2 ;;
    --refine-step-size) REFINE_STEP_SIZE="$2"; shift 2 ;;
    -*)
      echo "error: unknown option $1 (see --help)" >&2
      exit 1
      ;;
    *)
      DATA_DIRS+=("$1")
      shift
      ;;
  esac
done

if ((${#DATA_DIRS[@]} == 0)); then
  usage >&2
  exit 1
fi

if ((${#METHODS[@]} == 0)); then
  append_list METHODS ALL_METHODS "${METHOD:-adam}"
fi
if ((${#CLEAR_KINDS[@]} == 0)); then
  append_list CLEAR_KINDS ALL_CLEAR_KINDS "${CLEAR_KIND:-capsule}"
fi
dedup_validate METHODS ALL_METHODS method || exit 1
dedup_validate CLEAR_KINDS ALL_CLEAR_KINDS clear-kind || exit 1

RESOLVED_DIRS=()
for d in "${DATA_DIRS[@]}"; do
  if ! rd="$(resolve_data_dir "${d}")"; then
    echo "error: DATA_DIR not a directory: ${d}" >&2
    exit 1
  fi
  RESOLVED_DIRS+=("${rd}")
done

PIPE_ARGS=(
  --union-kind "${UNION_KIND}"
  --margin-ratio "${MARGIN_RATIO}"
  --p-max-obs "${P_MAX_OBS}"
  --p-max-grp "${P_MAX_GRP}"
  --p-max-union "${P_MAX_UNION}"
  --refresh-every "${REFRESH_EVERY}"
  --coarse-step "${COARSE_STEPS}"
  --refine-step "${REFINE_STEPS}"
  --coarse-step-size "${COARSE_STEP_SIZE}"
  --refine-step-size "${REFINE_STEP_SIZE}"
)
if [[ "${TWO_STAGE}" == "1" ]]; then
  PIPE_ARGS+=(--two-stage)
else
  PIPE_ARGS+=(--no-two-stage)
fi
PIPE_ARGS_STR="$(printf '%q ' "${PIPE_ARGS[@]}")"

# Build job list: env|out_root|method|clear_kind|tag
JOBS_LIST=()
shopt -s nullglob
for data_dir in "${RESOLVED_DIRS[@]}"; do
  folder="$(basename "${data_dir}")"
  files=("${data_dir}"/instance_*.yaml)
  if ((${#files[@]} == 0)); then
    echo "error: no instance_*.yaml in ${data_dir}" >&2
    exit 1
  fi
  for method in "${METHODS[@]}"; do
    for clear_kind in "${CLEAR_KINDS[@]}"; do
      out_root="${SCRIPT_DIR}/runs/${folder}/${method}/${clear_kind}"
      mkdir -p "${out_root}"
      for env in "${files[@]}"; do
        name="$(basename "${env}" .yaml)"
        JOBS_LIST+=("${env}|${out_root}|${method}|${clear_kind}|${folder}/${method}/${clear_kind}/${name}")
      done
    done
  done
done

echo "data=${RESOLVED_DIRS[*]}"
echo "methods=${METHODS[*]}"
echo "clear_kinds=${CLEAR_KINDS[*]}"
echo "jobs_total=${#JOBS_LIST[@]}  parallel=${JOBS}"
echo "union=${UNION_KIND}  margin_ratio=${MARGIN_RATIO}"
echo "iters coarse=${COARSE_STEPS} refine=${REFINE_STEPS}  two_stage=${TWO_STAGE}"
echo "step_size coarse=${COARSE_STEP_SIZE} refine=${REFINE_STEP_SIZE}"

run_one() {
  local spec="$1" env out_root method clear_kind tag name out log rc
  IFS='|' read -r env out_root method clear_kind tag <<<"${spec}"
  name="$(basename "${env}" .yaml)"
  out="${out_root}/${name}"
  mkdir -p "${out}"
  log="${out}/run.log"
  echo "=== ${tag} → ${out} ==="
  set +e
  # shellcheck disable=SC2086
  eval "\"${PYTHON}\" -m pipe_smooth.pipeline \"${env}\" -o \"${out}\" -m \"${method}\" --clear-kind \"${clear_kind}\" ${PIPE_ARGS_STR}" \
    >"${log}" 2>&1
  rc=$?
  set -e
  if ((rc == 0)); then
    echo "OK  ${tag}"
  else
    echo "FAIL ${tag} (rc=${rc})  see ${log}" >&2
  fi
  return "${rc}"
}
export -f run_one
export PYTHON PIPE_ARGS_STR

cd "${ROOT}"
fail=0
running=0
pids=()
for spec in "${JOBS_LIST[@]}"; do
  while (( running >= JOBS )); do
    if ! wait -n; then
      fail=1
    fi
    running=$((running - 1))
  done
  run_one "${spec}" &
  pids+=("$!")
  running=$((running + 1))
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

echo "done → ${SCRIPT_DIR}/runs/<folder>/<method>/<clear_kind>/  fail=${fail}"
exit "${fail}"
