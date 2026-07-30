# Visualization consumers (Python). C++ only exports data under SearchInstance.output_dir.

## Inputs (from C++ tools)

| File | Producer | Used for |
|------|----------|----------|
| `reports.jsonl` | `pipe_run_low` | path overlay, success–time aggregates |
| `closed_set/closed_q*.json` | `pipe_run_low` if `export_closed_set` | closed-set / expansion-order maps |
| `posthoc/trace_q*.json` | `pipe_run_low` if `export_posthoc` | Posthoc conversion / deep replay |
| `heuristic_q*.json` | `pipe_dump_heuristic` | heuristic heatmaps |

## Planned scripts (TODO)

- `plot_closed.py` — closed-set colored by `t`
- `plot_heuristic.py` — heatmap of `h`
- `to_posthoc.py` — `posthoc/trace_q*.json` → Posthoc `.trace.json`

Do not call into `pipe_planner` from here; read JSON only.
