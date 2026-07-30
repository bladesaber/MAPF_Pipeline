# tools/viz

Python plots / adapters over C++ dump files under `SearchInstance.output_dir`.
**Read JSON only** — do not call into `pipe_planner`.

## Inputs

| File | Producer | Script |
|------|----------|--------|
| `heuristic_q{id}.json` | `pipe_dump_heuristic` | `plot_heuristic.py` |
| `posthoc/trace_q{id}.json` | `pipe_run_low` + `export_posthoc` | `to_posthoc.py` |
| `reports.jsonl` | `pipe_run_low` | (TODO path / aggregates) |
| `closed_set/closed_q*.json` | `pipe_run_low` + `export_closed_set` | (TODO) |

---

## Heuristic heatmap

```bash
pipe_dump_heuristic --instance /path/to/runs/arena/instance.json
# → heuristic_q0.json …

python3 pipe_design/tools/viz/plot_heuristic.py \
  -i /path/to/runs/arena/heuristic_q0.json \
  -o /path/to/runs/arena/heuristic_q0.png
```

Free cells colored by `h`; blocked gray; goals as red ×. `--z N` for `size_z>1`.
Depends on `matplotlib` + `numpy`.

---

## Posthoc export

C++ writes a Posthoc-ish dump (`version: 1.4.0`, events `source` / `expand` / `end`).
Python turns it into a drop-in [search-trace](https://posthoc.pathfinding.ai/docs/search-trace) file and an optional MovingAI `.grid` map layer.

```bash
# run with export_posthoc: true in the instance
pipe_run_low --instance /path/to/runs/arena/instance.json
# → posthoc/trace_q0.json

python3 pipe_design/tools/viz/to_posthoc.py \
  -i /path/to/runs/arena/posthoc/trace_q0.json
# → posthoc/trace_q0.trace.yaml
# → posthoc/arena.grid          (from ../instance.json obstacles)
```

| Flag | Meaning |
|------|---------|
| `--instance PATH` | SearchInstance for `.grid` (default: `../instance.json`) |
| `--grid-out PATH` | where to write `.grid` |
| `--no-grid` | skip map export |
| `-o PATH` | output `.trace.yaml` path |

In Posthoc: add a **Map** layer ← `.grid`, add a **Trace** layer ← `.trace.yaml`, then step/play.

Depends on `PyYAML`.

---

## Planned

- `plot_path.py` — path + obstacles from `reports.jsonl` + instance
- `plot_closed.py` — closed-set colored by expansion index `t`
