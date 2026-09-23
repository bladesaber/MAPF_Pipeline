#!/usr/bin/env bash
# Run Stage-1 pipeline on instance_*.yaml in a folder.
#
# Usage:
#   ./run_instances.sh DATA_DIR [METHOD] [STEPS] [CLEAR_KIND]
# Example:
#   ./run_instances.sh data/easy
#   ./run_instances.sh data/hard adam 100 ignore
# Defaults: METHOD=adam  STEPS=50  CLEAR_KIND=capsule
# Results → pipe_smooth/runs/<DATA_DIR basename>/instance_*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 DATA_DIR [METHOD] [STEPS] [CLEAR_KIND]" >&2
  exit 1
fi

DATA_DIR="$(cd "${1}" && pwd)"
METHOD="${2:-adam}"
STEPS="${3:-50}"
CLEAR_KIND="${4:-capsule}"
OUT_ROOT="${SCRIPT_DIR}/runs/$(basename "${DATA_DIR}")"

mkdir -p "${OUT_ROOT}"
shopt -s nullglob
files=("${DATA_DIR}"/instance_*.yaml)
if ((${#files[@]} == 0)); then
  echo "error: no instance_*.yaml in ${DATA_DIR}" >&2
  exit 1
fi

echo "data=${DATA_DIR}  method=${METHOD}  steps=${STEPS}  clear=${CLEAR_KIND}  n=${#files[@]}"
cd "${ROOT}"
for env in "${files[@]}"; do
  name="$(basename "${env}" .yaml)"
  out="${OUT_ROOT}/${name}"
  echo "=== ${name} → ${out} ==="
  python -m pipe_smooth.pipeline \
    "${env}" \
    -o "${out}" \
    -m "${METHOD}" \
    --steps "${STEPS}" \
    --clear-kind "${CLEAR_KIND}"
done
echo "done → ${OUT_ROOT}"
