#!/usr/bin/env bash
# Plain batch commands for pipe_run_design.
# Edit / copy lines below to add new runs.
set -euo pipefail

PIPE_RUN="/home/admin123456/Desktop/work/MAPF_Pipeline/pipe_design/build/pipe_run_design"
BENCH="/home/admin123456/Desktop/work/mapf_3d_benchmark"
OUT="/home/admin123456/Desktop/work/mapf_3d_benchmark/runs_easy_middle"

# --- CBS ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/easy" \
  --export_dir "${OUT}/cbs/axis_ray/easy" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 30000

"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export_dir "${OUT}/cbs/axis_ray/middle" \
  --algorithm cbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal \
  --budget_mode time --max_time_ms 30000

# --- PP ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/easy" \
  --export_dir "${OUT}/pp/axis_ray/easy" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 30000

"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export_dir "${OUT}/pp/axis_ray/middle" \
  --algorithm pp \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --budget_mode time --max_time_ms 30000

# --- PBS ---
"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/easy" \
  --export_dir "${OUT}/pbs/axis_ray/easy" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 30000

"${PIPE_RUN}" \
  --instance_dir "${BENCH}/axis_ray/middle" \
  --export_dir "${OUT}/pbs/axis_ray/middle" \
  --algorithm pbs \
  --topology pure_mst --radius_merge min --cost_type euclidean \
  --heuristic manhattan --action_model TwentySixConnected \
  --conflict_analyzer near_terminal --pbs_mode depth_first \
  --budget_mode time --max_time_ms 30000
