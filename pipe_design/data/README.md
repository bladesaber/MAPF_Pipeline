# data/

Shared experiment assets for `pipe_design` (used by multiple papers).

| Path | Content |
|------|---------|
| `envs/` | Generated instances (`path_gen/obstacle/level/instance_*.yaml`) |
| `runs/` | Solver JSONL + logs from `include/benchmark/run_bench.sh` |

Both are gitignored (large). Regenerate envs with `gen_robust.sh`; re-run solvers with `run_bench.sh`.
