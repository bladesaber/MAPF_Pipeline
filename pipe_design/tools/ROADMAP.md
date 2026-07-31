# pipe_design/tools Roadmap

Visualization, benchmarks, and statistics for `pipe_design` planners.
**Core libraries stay clean:** planners emit data + stats; this folder owns I/O, viz, and batch runs.

Parent catalog: [`../ROADMAP.md`](../ROADMAP.md) (§4.2 low-level, §4.3 middle, §4.4 heuristics).

---

## 1. Goals

| Goal | Deliverable / criterion |
|------|-------------------------|
| Low/mid SOTA compare | Same MovingAI `.map`/`.scen` (P0) + voxels (P1); report success rate, time, expansions, path cost vs `reference_cost` when present |
| Search-shape debug | Path + heading overlay; closed-set colored by expansion order — catch “search away from goal” before merging a new planner |
| Heuristic correctness | Heatmap of `h` and `h − true_dist`; fail CI/sample checks if `h > true` (inadmissible) |
| Perf / memory profile | Per-run `SearchReport`: wall time, open peak, closed/nodes, reexpansions, bytes estimate — compare algos on identical queries |
| Aggregate plots | Success–time (or expansions) curves; anytime profiles when ARA*/ANA* exist |
| High-level stress | Path-then-obstacle instances (≥1 reference path, optional 2nd); difficulty via obstacle size + occupancy density (+ corridor/choke) |
| One report schema | Every runner fills the same `SearchReport` JSON; new planner = one adapter, not a new format |

**Rule:** low/mid tools ignore replan and pipe radius (pure 2D/3D grid search). High-level generators keep radius / clearance / multi-solution semantics.

---

## 2. Layout (target)

```text
tools/
  ROADMAP.md          # this file
  data/
    movingai/         # raw MovingAI downloads (grids/ P0, voxels/ P1) — not committed
  formats/            # schemas + Python loaders/writers for standard instance / report
  gen/                # Python: MovingAI → instance; synthetic / high-level generators
  runners/            # C++ (thin CLI): read instance → pipe_planner → SearchReport (+ Trace)
  bench/              # Python: batch compare, JSONL/CSV tables
  viz/                # Python: path, heuristic, closed-set, aggregate plots; Posthoc export
```

Do not put CBS / topology / production routing logic here.
Raw benchmark files live under `data/movingai/` (see that README); loaders must not hard-code machine-specific paths outside `tools/data/`.
Generated instances and C++ dumps go to a caller-chosen `--output-dir` (not under `data/` by default).

---

## 3. Language split

**Policy:** Python builds standard **inputs**; C++ runs the planner and emits standard **outputs**; visualization is primarily Python (Posthoc optional). Do **not** implement the whole tools stack in C++ — high cost, low leverage vs search itself.

```text
Python (gen / load)     →  standard instance (.map/.scen and/or JSON)
        ↓
C++ runner (thin)       →  pipe_design::AStar… → SearchReport (+ optional Trace)
        ↓
Python (viz / bench)    →  plots, tables, Posthoc .trace.json export
```

| Layer | Language | Does | Does *not* |
|-------|----------|------|------------|
| Input | **Python** | MovingAI parse, synthetic maps, high-level path-then-obstacle, scenario batching | A\* hot path |
| Solve | **C++** | Call `pipe_planner`; write `SearchReport` / `Trace` | Plotting; many benchmark dialects |
| Viz / aggregate | **Python** (default) | Path/closed/heatmap, success–time, JSONL rollups | Reimplement search |
| Posthoc | Python adapter | Internal `Trace` → Posthoc search-trace | Coupling into core libs |

**Why this split**

- Gen + I/O iterate fast in Python; same work in C++ is slow to ship.
- CPU matters only in expand — keep the runner thin (`read → reset/search → write JSON`).
- Full C++ viz (Qt/OpenGL) is P3 product work; matplotlib / plotly / Posthoc cover P0–P2.
- One instance schema + one `SearchReport` keeps both sides simple.

**Rules**

1. Freeze schemas early (§5); every algo shares the same in/out files.
2. C++ runner stays thin (CLI / small `main`); stats live on planner or a thin Runner wrapper.
3. Viz dependencies stay out of `pipe_grid` / `pipe_planner`.
4. Correctness unit tests remain C++ `tests/` + `ctest`; tools own batch regression and diagnosis.
5. If Python I/O is too slow on huge voxels later, add a C++ binary reader — not before; P0 2D MovingAI is fine in Python.
6. Avoid early whole-library pybind; add only if a thin binding clearly helps the runner.

**Anti-patterns:** all-C++ tools; all-Python planner rewrite; per-algo custom JSON.

---

## 4. Open benchmarks (URLs)

| URL | Brief |
|-----|--------|
| https://www.movingai.com/benchmarks/ | Hub: 2D grids, 3D voxels, MAPF |
| https://www.movingai.com/benchmarks/grids.html | ★ 2D maps + scenarios (rooms, mazes, dao, …) — P0 |
| https://movingai.com/benchmarks/formats.html | `.map` / `.scen` format |
| https://www.movingai.com/benchmarks/voxels.html | Warframe 3D voxels |
| https://benchmarks.pathfinding.ai/ | Shortest Path Lab hub / mirrors |
| https://benchmarks.pathfinding.ai/3d-benchmarks/voxel-maps/ | ★ 3D voxels: Plants (pipe-like), Descent, Sandstone, Warframe — P1 |
| https://bitbucket.org/shortestpathlab/benchmarks | Download mirror (`grid-maps/`, `voxel-maps/`) |
| https://mapf.info/index.php?n=Main.Benchmarks | MAPF hub: MovingAI MAPF scen, asprilo, 3D warehouse MAPF, [MAPF Tracker](https://tracker.pathfinding.ai/) |
| https://posthoc.pathfinding.ai/ | ★ Posthoc — search-trace viz / replay ([posthoc-app](https://github.com/shortestpathlab/posthoc-app)) — optional P2 |

Python loaders target MovingAI `.map` / `.scen` first (P0); voxel maps next (P1). Prefer these over inventing private map formats.

---

## 5. Data contracts

Keep schemas versioned and small. Prefer JSON for interchange; binary optional later.
**Python writes instances; C++ reads instances and writes reports** — both sides must agree on these contracts.

### 5.1 `SearchInstance` JSON (low / mid) — `tools/formats/SearchInstance`

Canonical file-bridge input for **low- and middle-level** grid search.
Implemented by `pipe_design::tools::SearchInstance`.
Python writes; C++ loads. Defaults for degraded 2D grid search: `resolution=1`,
`pipe_radius=0.1`, sphere `cell_radius=0.1`, `size_z=1`.

Do **not** reuse this schema for full pipe-design / multi-group problems — that is §5.2 `DesignInstance` (future).

| Field | Type | Notes |
|-------|------|--------|
| `version` | int | Schema version (`1`) |
| `instance_id` | string | Id for this instance / experiment slice |
| `map_id` | string | Map provenance, e.g. `dao/arena` |
| `size` | `[sx,sy]` or `[sx,sy,sz]` | Grid dimensions |
| `resolution` | float | Default `1.0` (world cell length; default `step_cost`) |
| `clearance_cap` | float | Default `1.0` |
| `pipe_radius` | float | Default `0.1` |
| `action_model` | string | `"SixConnected"` \| `"EightConnected"` \| `"TwentySixConnected"` |
| `action_model_kwargs` | object | e.g. `step_cost` (default = `resolution`), `turn_weight` |
| `heuristic` | string | `"manhattan"` \| `"euclidean"` \| `"octile"` \| `"dijkstra"` |
| `heuristic_kwargs` | object | e.g. `step_cost` (default = `resolution`) |
| `heuristic_id` | string | Preset provenance id (optional; default empty) |
| `heap` | string | `"LazyBinaryHeap"` \| `"BinaryHeap"` \| `"PairingHeap"` \| `"BucketHeap"` |
| `heap_kwargs` | object | e.g. BucketHeap `delta`, `max_cost` |
| `heap_id` | string | Preset provenance id (optional; default empty) |
| `model` | string | `"astar"` \| `"directed_astar"` (more later) |
| `model_kwargs` | object | e.g. `heuristic_weight`; directed: `num_yaw_bins`, `num_pitch_bins` |
| `model_id` | string | Preset provenance id (optional; default empty) |
| `max_expansions` | int? | Omit / null = unlimited |
| `max_time_ms` | float | Cap on **sum of `step()` time** per query; `<= 0` = unlimited (default `0`) |
| `output_dir` | string | Where C++ writes reports / dumps; empty = stdout only (no files). For `gen/movingai_grids.py`: one map → one instance dir → this field equals that dir. |
| `export_closed_set` | bool | If true, write under `output_dir/closed_set/` |
| `export_posthoc` | bool | If true, write under `output_dir/posthoc/` (requires `size_z==1`) |
| `obstacles` | **list** of objects | See below — not a single dict |
| `queries` | list | Each query may have **multiple** starts/goals |

**`obstacles[]` entries** (only two modes; load fills `spheres` / `capsules`):

| `mode` | Fields | Effect |
|--------|--------|--------|
| `"sphere"` | `cell_radius`, `cells: [[x,y,z], …]` | Sphere at each cell center |
| `"capsules"` | `cell_radius`, `from_point`, `to_point` | One capsule (world coords) |

**`queries[]` entries** (`SearchQuery`):

| Field | Type | Notes |
|-------|------|--------|
| `query_id` | int | Optional; auto-assigned if omitted |
| `start` | `[x,y,z]` **or** `[[x,y,z], …]` | Non-empty; shorthand single cell allowed |
| `goal` | same | Non-empty |
| `reference_cost` | float? | Benchmark or seed reference (e.g. MovingAI scen cost) |

Example:

```json
{
  "version": 1,
  "instance_id": "dao/arena/smoke",
  "map_id": "dao/arena",
  "size": [sx, sy, sz],
  "resolution": 1.0,
  "clearance_cap": 1.0,
  "pipe_radius": 0.1,
  "action_model": "EightConnected",
  "action_model_kwargs": {"step_cost": 1.0},
  "heuristic": "manhattan",
  "heuristic_kwargs": {"step_cost": 1.0},
  "heap": "LazyBinaryHeap",
  "heap_kwargs": {},
  "model": "astar",
  "model_kwargs": {"heuristic_weight": 1.0},
  "max_expansions": null,
  "max_time_ms": 0,
  "output_dir": "runs/smoke",
  "export_closed_set": false,
  "export_posthoc": false,
  "obstacles": [
    {"mode": "sphere", "cell_radius": 0.1, "cells": [[x,y,z], ...]},
    {"mode": "capsules", "cell_radius": 0.1,
     "from_point": [x,y,z], "to_point": [x,y,z]}
  ],
  "queries": [
    {"query_id": 0,
     "start": [[x,y,z], ...],
     "goal":  [[x,y,z], ...],
     "reference_cost": 3.2}
  ]
}
```

**Preferred map source:** §4 open benchmarks (2D grids P0; 3D voxels P1). Thin 3D fallback = `nz=1` extrusion only when a voxel set is unavailable.

### 5.2 `DesignInstance` (high-level / project-facing) — **future**

> **TODO (not implemented):** `tools/formats/DesignInstance` — do not conflate with `SearchInstance`.

Intended for real pipe-design conditions (radius, clearance, multi-path seeds, groups), not pure grid SOTA benches.

Pipeline sketch: **seed ≥1 feasible reference paths → place obstacles** (size + volume/area occupancy density = difficulty knobs).

Also emit: reference path(s), optional second solution, domain bounds, pipe radius, static obstacle list. Difficulty axes beyond density: corridor width, maze-ness, local choke points.

When implementing: new type + schema; keep §5.1 `SearchInstance` unchanged for low/mid runners.

### 5.3 `SearchReport` (one run → one file) — `tools/formats/SearchReport`

Emitted by the **C++ runner** after each query (`SearchInstance` / low–mid).
JSON object; batch runs use **JSONL** (`write_search_reports_jsonl`).

| Field | Meaning |
|-------|---------|
| `success` / `status` | Success / Failure / Running (hit expansion or time cap) |
| `total_time_ms` | Sum of `planner.step()` only (excludes export / limit checks) |
| `expansions` | Closed pops |
| `generated` | Currently ≈ `nodes_stored` (fine-grained counters TBD in planner) |
| `reexpansions` | Reopen count — **0 until A\* instruments reopen** |
| `open_peak` / `open_final` | Open-set size (peak tracked in runner step loop) |
| `closed` / `nodes_stored` | Closed-status count / `nodes_.size()` |
| `bytes_estimate` | Rough: `nodes_stored * 64 + open_final * 16` |
| `path_cost` / `path_length` / `path_turns` | Solution quality (`path` cell list included) |
| `model`, `heuristic`, `action_model`, `heap` | Solver type names from `SearchInstance` |
| `model_id`, `heap_id`, `heuristic_id` | Preset provenance ids from `SearchInstance` (may be empty) |
| `instance_id`, `map_id`, `query_id` | Provenance |
| `reference_cost` | Copied from query when present |

### 5.3.1 Viz / debug dumps (under `output_dir`, not in `SearchReport`)

`SearchReport` stays lean (final result + counters). Process viz uses optional side files:

| File | Producer | When | Content |
|------|----------|------|---------|
| `reports.jsonl` | `pipe_run_low` | `output_dir` non-empty | One `SearchReport` per line |
| `closed_set/closed_q{id}.json` | `pipe_run_low` | `export_closed_set` | `{size, closed:[{x,y,z,t},…]}` — `t` = expansion index |
| `posthoc/trace_q{id}.json` | `pipe_run_low` | `export_posthoc` | Posthoc-ish events (`source`/`expand`/`end`); requires `size_z==1` |
| `heuristic_q{id}.json` | `pipe_dump_heuristic` | always (needs `output_dir`) | Free cells `{x,y,z,h}` via `Heuristic::estimate` (no search) |

CLI: `pipe_run_low --instance …` and `pipe_dump_heuristic --instance …` — **no `--out`**; path comes from `SearchInstance.output_dir`.
Python `viz/` reads these JSON files only (see `viz/README.md`). Posthoc format conversion stays in Python.

### 5.4 `DesignReport` (high-level / project-facing) — **future**

> **TODO (not implemented):** `tools/formats/DesignReport` — do not conflate with `SearchReport`.

Intended output for `DesignInstance` solves (multi-pipe / groups / clearance quality), e.g. per-group success, conflict counts, total pipe length, clearance margins. Keep separate schema when high-level runners land.

### 5.5 Planner instrumentation (library side, not tools)

`AStar` today exposes `expansions()` and `nodes()`. Runner derives open peak / closed from open set + node status.
Fine-grained `generated` / `reexpansions` need planner counters later; tools must not re-implement search bookkeeping.

---

## 6. Component catalog

Legend: **P0** first → **P3** later.

| Component | Priority | Lang | Notes |
|-----------|----------|------|-------|
| ★ `SearchInstance` + `SearchReport` schemas | P0 | Py + C++ | Low/mid (§5.1 / §5.3); `DesignInstance` / `DesignReport` later |
| ★ C++ thin runner | P0 | C++ | `pipe_run_low`: instance → A* → `reports.jsonl` (+ optional closed/trace) |
| ★ C++ heuristic dump | P1 | C++ | `pipe_dump_heuristic`: `estimate` heatmap JSON (no search) |
| ★ Path + obstacle grid visualizer | P0 | Py | Mark blocked cells; draw path; show heading per step |
| ★ Closed-set overlay | P0 | Py | Color by first-closed expansion index (2D slice) |
| ★ MovingAI loader + batch driver | P0 | Py | Load `.map`/`.scen` → instance; invoke C++ runner |
| Heuristic heatmap | P1 | Py | `h`, `h − true_dist`; assert admissibility / optional consistency |
| Success–time (or expansions) plots | P1 | Py | Scatter now; anytime profiles when ARA*/ANA* land |
| Synthetic low/mid map gen | P1 | Py | Rooms / corridors / random walls for regression |
| High-level path-then-obstacle gen | P2 | Py | Real-condition stress; multi-path seeds |
| ★ Posthoc search-trace export | P2 | Py | Adapter: internal `Trace` → Posthoc `.trace.json`/`.yaml`; browser replay — not P0 batch plots |
| 3D viz | P2 | Py | Default: z-slices or path + obstacle points; full volume later |
| Memory / allocator profiles | P3 | C++/Py | Prefer node counts for algo compare |

---

## 7. Visualization notes

- **C++ runner:** dump `SearchReport` / optional `Trace` only — no GUI.
- **Python (`viz/`, `bench/`):** matplotlib / plotly for P0/P1 batch plots (path, closed, success–time).
- **Posthoc (P2):** single-run deep debug — step/replay via [search traces](https://posthoc.pathfinding.ai/docs/search-trace). Keep internal `Trace` / `SearchReport`; add a Python exporter; do not couple core to Posthoc.
- **3D:** slice first; avoid heavy engines until needed.
- **Heuristic tools** are diagnostics (admissibility), not demos.
- **Closed-set maps** answer “did search go toward the goal?” — highest debug ROI after path draw.

---

## 8. Implementation phases

### Phase T0 — Foundation

- [x] Define `SearchInstance` / `SearchReport` schemas (§5.1 / §5.3)
- [ ] Instrument planner for fine-grained `generated` / `reexpansions` (§5.5)
- [x] **C++** thin runner: instance → search → `reports.jsonl` under `output_dir` (no `--out` CLI)
- [x] **C++** optional dumps: `export_closed_set` / `export_posthoc` → `closed_q*` / `trace_q*`
- [x] **C++** `pipe_dump_heuristic`: independent heatmap via `Heuristic::estimate`
- [ ] **Python** `viz/`: 2D path + obstacles + headings (read report + map)
- [ ] **Python** `viz/`: closed-set heatmap on same grid

### Phase T1 — Benchmark loop

- [x] **Python** MovingAI `.map` / `.scen` → standard instance (§4 grids — P0) (`tools/gen/movingai_grids.py`)
- [x] **Python** MovingAI batch gen (model × heap × heuristic presets) → instances (`tools/gen/batch_movingai_grids.py`)
- [x] **Python** low/mid search bench: run batch → `all_reports.csv` (`tools/bench/compare_search.py`); one-factor compare (`tools/bench/factor_compare.py`)
- [ ] **Python** aggregate reports → JSONL/CSV tables / plots
- [ ] Success vs time/expansions aggregate plot
- [ ] Heuristic vs Dijkstra true-distance heatmap + admissibility checks
- [ ] Wire into comparing new low-level algos (Theta*, weighted, …)
- [ ] Voxel map loader (§4 3D — P1; Plants / Descent / …)

### Phase T2 — Generators & high-level

- [ ] **Python** synthetic grid generator (controllable density / topology)
- [ ] **Python** high-level: reference path(s) → obstacle placement (size + occupancy)
- [ ] Multi-query / multi-solution instance support
- [ ] Optional 3D slice viewer
- [x] **Python** Posthoc exporter: `posthoc/trace_q*` → `.trace.yaml` + optional `.grid` (`tools/viz/to_posthoc.py`)

### Phase T3 — Polish

- [ ] Expansion replay (native `Trace` and/or drop into [Posthoc](https://posthoc.pathfinding.ai/))
- [ ] Algo comparison dashboard (same scenario set)
- [ ] Document: add a planner = thin C++ runner entry + same schemas (&lt;10 lines of glue)

---

## 9. How to extend

1. New low-level planner → implement in `pipe_design` → one **C++** runner entry that fills `SearchReport` (same schema).
2. New map source → **Python** loader to §5.1 (prefer §4 URLs); do not fork report format.
3. New plot → **Python** reads JSONL of reports or a single `Trace`; do not call into search internals.
4. When adding a tool, tick the checklist above and keep this file’s catalog in sync.

### Suggested first PR order

Schemas → C++ runner (`SearchReport`) → Python path/closed viz → Python MovingAI batch → heuristic heatmap → generators → Posthoc export.

---

## 10. Non-goals

- Replacing `tests/` unit tests (tools complement, not substitute).
- Production CAD export / UI product.
- Embedding viz dependencies into `pipe_grid` / `pipe_planner` libraries.
- Rewriting planners in Python, or building the entire tools stack in C++.
- Early full-library pybind for visualization convenience.
