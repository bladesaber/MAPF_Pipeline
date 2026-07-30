# pipe_design/tools Roadmap

Visualization, benchmarks, and statistics for `pipe_design` planners.
**Core libraries stay clean:** planners emit data + stats; this folder owns I/O, viz, and batch runs.

Parent catalog: [`../ROADMAP.md`](../ROADMAP.md) (§4.2 low-level, §4.3 middle, §4.4 heuristics).

---

## 1. Goals

| Goal | Deliverable / criterion |
|------|-------------------------|
| Low/mid SOTA compare | Same MovingAI `.map`/`.scen` (P0) + voxels (P1); report success rate, time, expansions, path cost vs known `opt_cost` when present |
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
  formats/            # schemas + Python loaders/writers for standard instance / report
  gen/                # Python: MovingAI → instance; synthetic / high-level generators
  runners/            # C++ (thin CLI): read instance → pipe_planner → SearchReport (+ Trace)
  bench/              # Python: batch compare, JSONL/CSV tables
  viz/                # Python: path, heuristic, closed-set, aggregate plots; Posthoc export
```

Do not put CBS / topology / production routing logic here.

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

### 5.1 Grid map + scenario (low / mid)

| Field | Notes |
|-------|--------|
| `width, height, depth` | `depth=1` for 2D |
| `obstacles` / grid cells | blocked vs free |
| `connectivity` | `6` / `8` / `26` |
| `starts[]`, `goals[]` | one or many queries per map |
| `opt_cost` (optional) | known optimal for regression |

**Preferred source:** §4 open benchmarks (2D grids P0; 3D voxels P1). Thin 3D fallback = `nz=1` extrusion only when a voxel set is unavailable.

### 5.2 High-level instance (project-facing)

Pipeline: **seed ≥1 feasible reference paths → place obstacles** (size + volume/area occupancy density = difficulty knobs).

Also emit: reference path(s), optional second solution, domain bounds, pipe radius, static obstacle list. Difficulty axes beyond density: corridor width, maze-ness, local choke points.

### 5.3 `SearchReport` (one run → one file)

Emitted by the **C++ runner** after each query.

| Field | Meaning |
|-------|---------|
| `success` / `status` | Success / Failure / expansion cap |
| `total_time_ms` | Wall clock |
| `expansions` | Closed pops |
| `generated` | Nodes created / relaxed |
| `reexpansions` | Reopen count (0 if consistent h + no reopen) |
| `open_peak` / `open_final` | Open-set size |
| `closed` / `nodes_stored` | Closed / `nodes_.size()` |
| `bytes_estimate` | Rough: nodes × sizeof + open (RSS optional appendix) |
| `path_cost` / `path_length` / `path_turns` | Solution quality |
| `algo`, `heuristic`, `map_id`, `query_id` | Provenance |

Optional trace dump for viz: closed order (cell + expansion index), final path cells + headings, open frontier snapshot.

### 5.4 Planner instrumentation (library side, not tools)

`AStar` today exposes mainly `expansions()`. Before rich viz, add counters (on planner or a thin Runner wrapper): `generated`, `reexpansions`, `open_peak`, timing. Tools must not re-implement search bookkeeping.

---

## 6. Component catalog

Legend: **P0** first → **P3** later.

| Component | Priority | Lang | Notes |
|-----------|----------|------|-------|
| ★ Instance + `SearchReport` schemas | P0 | Py + C++ | Shared contracts (§5) |
| ★ C++ thin runner | P0 | C++ | instance → A* → `SearchReport` JSON |
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

- [ ] Define `formats/` schemas (`Map`, `Scenario`, `SearchReport`, optional `Trace`) — language-agnostic
- [ ] Instrument planner / Runner: stats fields in §5.3
- [ ] **C++** thin runner: instance → A* → `SearchReport` JSON
- [ ] **Python** `viz/`: 2D path + obstacles + headings (read report + map)
- [ ] **Python** `viz/`: closed-set heatmap on same grid

### Phase T1 — Benchmark loop

- [ ] **Python** MovingAI `.map` / `.scen` → standard instance (§4 grids — P0)
- [ ] **Python** batch driver → invoke C++ runner → JSONL/CSV of reports
- [ ] Success vs time/expansions aggregate plot
- [ ] Heuristic vs Dijkstra true-distance heatmap + admissibility checks
- [ ] Wire into comparing new low-level algos (Theta*, weighted, …)
- [ ] Voxel map loader (§4 3D — P1; Plants / Descent / …)

### Phase T2 — Generators & high-level

- [ ] **Python** synthetic grid generator (controllable density / topology)
- [ ] **Python** high-level: reference path(s) → obstacle placement (size + occupancy)
- [ ] Multi-query / multi-solution instance support
- [ ] Optional 3D slice viewer
- [ ] **Python** Posthoc exporter: `Trace` → `.trace.json` / `.yaml` ([docs](https://posthoc.pathfinding.ai/docs/search-trace))

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
