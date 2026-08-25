#!/usr/bin/env bash
# Plain batch commands for pipe_run_design.
# Edit / copy lines below to add new runs.
#
# Matrix: {cbs,pp,pbs} × {axis_ray,plane_slice} × {easy,middle,hard}
#   easy:   commented (already run for axis_ray)
#   middle: max_time_ms 30000
#   hard:   max_time_ms 90000
#
# --export PATH.jsonl  (file, not directory)
# --meta JSON object → DesignReport.meta (generator / level / timestamp / …)
set -euo pipefail

PIPE_RUN="/home/admin123456/Desktop/work/MAPF_Pipeline/pipe_design/build/pipe_run_design"
BENCH="/home/admin123456/Desktop/work/mapf_3d_benchmark"
OUT="/home/admin123456/Desktop/work/mapf_3d_benchmark/runs_results"
# Shared batch timestamp (UTC) for all commands below.
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# =============================================================================
# --- CBS ---
# =============================================================================

# --- CBS / axis_ray / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/axis_ray/easy" \
#   --export "${OUT}/cbs/axis_ray/easy.jsonl" \
#   --algorithm cbs \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --conflict_analyzer near_terminal \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"axis_ray\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- CBS / axis_ray / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export "${OUT}/cbs/axis_ray/middle.jsonl" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- CBS / axis_ray / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/hard" \
  --export "${OUT}/cbs/axis_ray/hard.jsonl" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"

# --- CBS / plane_slice / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/plane_slice/easy" \
#   --export "${OUT}/cbs/plane_slice/easy.jsonl" \
#   --algorithm cbs \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --conflict_analyzer near_terminal \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"plane_slice\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- CBS / plane_slice / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/middle" \
  --export "${OUT}/cbs/plane_slice/middle.jsonl" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- CBS / plane_slice / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/hard" \
  --export "${OUT}/cbs/plane_slice/hard.jsonl" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"

# =============================================================================
# --- PP ---
# =============================================================================

# --- PP / axis_ray / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/axis_ray/easy" \
#   --export "${OUT}/pp/axis_ray/easy.jsonl" \
#   --algorithm pp \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"axis_ray\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- PP / axis_ray / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export "${OUT}/pp/axis_ray/middle.jsonl" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- PP / axis_ray / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/hard" \
  --export "${OUT}/pp/axis_ray/hard.jsonl" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"

# --- PP / plane_slice / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/plane_slice/easy" \
#   --export "${OUT}/pp/plane_slice/easy.jsonl" \
#   --algorithm pp \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"plane_slice\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- PP / plane_slice / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/middle" \
  --export "${OUT}/pp/plane_slice/middle.jsonl" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- PP / plane_slice / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/hard" \
  --export "${OUT}/pp/plane_slice/hard.jsonl" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"

# =============================================================================
# --- PBS ---
# =============================================================================

# --- PBS / axis_ray / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/axis_ray/easy" \
#   --export "${OUT}/pbs/axis_ray/easy.jsonl" \
#   --algorithm pbs \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --conflict_analyzer near_terminal --pbs_mode depth_first \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"axis_ray\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- PBS / axis_ray / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export "${OUT}/pbs/axis_ray/middle.jsonl" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- PBS / axis_ray / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/hard" \
  --export "${OUT}/pbs/axis_ray/hard.jsonl" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"axis_ray\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"

# --- PBS / plane_slice / easy ---
# "${PIPE_RUN}" \
#   --instance_dir "${BENCH}/plane_slice/easy" \
#   --export "${OUT}/pbs/plane_slice/easy.jsonl" \
#   --algorithm pbs \
#   --topology pure_mst --radius_merge min --cost_type euclidean \
#   --heuristic manhattan --action_model TwentySixConnected \
#   --conflict_analyzer near_terminal --pbs_mode depth_first \
#   --budget_mode time --max_time_ms 30000 \
#   --meta "{\"generator\":\"plane_slice\",\"level\":\"easy\",\"timestamp\":\"${TS}\"}"

# --- PBS / plane_slice / middle ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/middle" \
  --export "${OUT}/pbs/plane_slice/middle.jsonl" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 30000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"middle\",\"timestamp\":\"${TS}\"}"

# --- PBS / plane_slice / hard ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/plane_slice/hard" \
  --export "${OUT}/pbs/plane_slice/hard.jsonl" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 90000 \
  --meta "{\"generator\":\"plane_slice\",\"level\":\"hard\",\"timestamp\":\"${TS}\"}"
