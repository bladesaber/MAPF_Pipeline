# Tech Note v0 — Multi-Group Pipe Routing under Permanent Geometric Occupancy

**Status:** internal draft (not a submission)  
**Date:** 2026-08-26  
**Code:** `pipe_design/`  
**Primary results:** full suite under `pipe_design/data/runs/` (36 JSONL batches;
analyze with `scripts/analyze_runs.py`).

**Purpose of this note**

1. Lock a publishable *problem story* and contribution boundary.  
2. Record the first closed experiment (CBS / PBS / PP).  
3. List what is still missing before a workshop / applied-conference draft.

This is **not** a claim of optimality, industrial readiness, or MAPF SOTA.

---

## 1. Problem

### 1.1 Informal statement

Route multiple **pipe groups** inside a bounded 3D design domain (static obstacles +
already-routed geometry). Unlike classical multi-agent pathfinding (MAPF):

| Classical MAPF | This problem |
|----------------|--------------|
| Agents free space after moving | A routed pipe **permanently occupies** volume |
| Conflicts are time–space (vertex/edge @ \(t\)) | Conflicts are **geometric clearance** between pipe volumes |
| One agent ↔ one path | One **group** ↔ multi-terminal **tree** (topology) |
| Same-cell clash | Two **witness points** (often different positions) + radii |

**Hard rule:** occupancy conflict between **different groups** is forbidden.  
**Soft / configurable:** within a group, paths may share / branch via topology
(default: MST-style multi-terminal plan).

### 1.2 Geometry

Obstacles and pipes are approximated by **spheres** and **capsules** (sphere swept
along a segment). A placement is blocked iff clearance is below the querying pipe
radius. Cross-group conflicts are therefore **not** classical same-cell clashes:
two pipes may collide at different world positions. Each recorded conflict stores
both witness points, radii, and deficit (see `include/conflict/`).

### 1.3 Input / output (operational)

**Input**

- Grid / domain parameters  
- Static obstacles (spheres / capsules; optional painted fields e.g. pillar_field)  
- Groups of face terminals with (currently uniform) pipe radius  
- Solver config: topology builder, low-level planner, high-level algorithm, budget  

**Output**

- Per-group set of task paths (cell polylines + costs), or failure under budget  
- DesignReport JSONL: success, time, iterations, cost vs constructive **witness**

**Objective in current experiments:** maximize **success rate under a fixed time
budget**. Path cost is secondary and reported vs witness when successful.

### 1.4 What we do *not* claim yet

- Optimal CBS / admissible high-level \(h\) (no WDG / EWMVC)  
- Continuous disjoint splitting  
- Kinematic pipes (Hybrid A\* / Dubins)  
- Process objectives (bend count, manufacturing, multi-radius mixing)

---

## 2. Method stack (as implemented)

Layered architecture (see `pipe_design/ROADMAP.md`):

```text
[T] Topology     terminals → GroupTopologyPlan / GlobalTopologyPlan (MST, …)
      ↓
 multi/          GroupSession / PrioritySequencer / PartialOrderSequencer
      ↓
[V] Conflict     sample paths → dual-witness geometric conflicts
      ↓
[F] High-level   CBS / PBS / PP search over leaves
      ↓
 low/            A* (runner default) on CompositeGrid = static ⋈ ObstacleChain
```

### 2.1 Topology `[T]`

Default in reported runs: **`pure_mst`** with Euclidean edge costs and
`radius_merge=min`. Builds per-group tasks (edges) and an intra-group grow order.
Same-group routed segments become connectable goals (tree-like sharing), not
cross-group obstacles.

### 2.2 Occupancy

- `StaticClearanceGrid`: baked static geometry  
- `ObstacleChain`: copy-on-write negative constraints (other groups’ paths + CBS
  witness spheres)  
- `CompositeGrid::is_free(cell, pipe_radius)`: hot-path freeness  

### 2.3 Conflict `[V]`

`ConflictAnalyzer` (+ `NearTerminalAnalyzer` in CBS/PBS runs) finds / clusters /
ranks cross-group geometric hits. Detection must **not** be stricter than
occupancy (otherwise CBS can loop: flag conflict → constrain → low-level still
sees free). Same-group hits are ignored.

### 2.4 High-level algorithms (experiment axis)

| Algo | Leaf / state | Repair mechanism | Conflict analyzer in runs |
|------|----------------|------------------|---------------------------|
| **CBS** | Constraint tree; add opposing witness sphere(s) to chain | Replan affected group via `GroupSession` | `near_terminal` |
| **PBS** | Priority DAG (`pbs_mode=depth_first`) | `PartialOrderSequencer` | `near_terminal` |
| **PP** | Total task order | Failure-driven sequence repair (`GroupLayerSwapReplanner`) | none (order search only) |

Low-level fixed: **A\***, heuristic **manhattan**, action model
**TwentySixConnected**.

**Contribution boundary (honest):** we **adapt** classical high-level MAPF
controllers to permanent geometric occupancy and multi-terminal topology. Novelty
is in the *problem coupling* (geometry + groups + chain constraints), not in
renaming CBS/PBS/PP.

---

## 3. Benchmark

### 3.1 Instance generation

Constructive pipeline (`include/benchmark/`):

1. Grow **witness corridors** (feasible tree per group)  
2. Paint static obstacles in remaining free space (`pillar_field` or
   `corridor_rack`)  
3. Emit YAML: environment + witness `solution`  

Generators (all four datasets present in `data/envs` / `data/runs`):

| Name | Idea | In suite? |
|------|------|-----------|
| `axis_ray` | LOS ray tree (preferred) | **Yes** |
| `plane_slice` | Planar A\* on axis slices | **Yes** |
| `pillar_field` | Large AABB clutter to density ρ | **Yes** |
| `corridor_rack` | Dilate witness; mesh complement | **Yes** |

Difficulty knobs (see `gen_robust.sh`): group count × terminals per group
(axis_ray ≈ 3×3 / 4×4 / 5×5; plane_slice ≈ 3×3 / 8×8 / 13×13).

Witness proves **feasibility of the generator**, not optimality. Solver
`cost_ratio = total_cost / reference_cost` can be **&lt; 1**.

### 3.2 Runner / report

- Binary: `pipe_run_design`  
- One DesignReport JSON object per instance → JSONL batch  
- Batch summary: `scripts/analyze_runs.py` (success rate, median time / cost_ratio,
  hard fail reasons, success@T); per-file browse: `scripts/view_report.py` Stats

### 3.3 Experiment config (current suite)

| Level | N | `max_time_ms` |
|-------|---|---------------|
| easy | 100 | 30 000 |
| middle | 100 | 60 000 |
| hard | 100 | 100 000 |

Shared: `topology=pure_mst`, `low_model=astar`, `heuristic=manhattan`,
`action_model=TwentySixConnected`, `budget_mode=time`.

Matrix: `{cbs,pbs,pp} × {axis_ray,plane_slice} × {pillar_field,corridor_rack} ×
{easy,middle,hard}` → **36** JSONL files under `pipe_design/data/runs/`.

---

## 4. Results (full suite, 2026-08-26)

Reproduce tables:

```bash
python3 pipe_design/scripts/analyze_runs.py
# or: python3 pipe_design/scripts/analyze_runs.py pipe_design/data/runs
```

### 4.1 Success rate (%)

**axis_ray × pillar_field**

| Level | CBS | PBS | PP |
|-------|-----|-----|-----|
| easy | 100 | 100 | 100 |
| middle | 100 | 100 | 100 |
| hard | **100** | 99 | 99 |

**axis_ray × corridor_rack** — all cells **100%** (CBS/PBS/PP × easy/middle/hard).

**plane_slice × pillar_field**

| Level | CBS | PBS | PP |
|-------|-----|-----|-----|
| easy | 100 | 100 | 100 |
| middle | **100** | 94 | 94 |
| hard | **93** | 56 | 7 |

**plane_slice × corridor_rack**

| Level | CBS | PBS | PP |
|-------|-----|-----|-----|
| easy | 100 | 100 | 100 |
| middle | **100** | 95 | 95 |
| hard | **88** | 51 | 11 |

**Takeaway:** `axis_ray` is too easy for ranking under this schedule (near-perfect
success). Discriminating stress comes from **plane_slice hard**, where
**CBS ≫ PBS ≫ PP** on both obstacle families.

### 4.2 Median success time (ms)

**plane_slice × pillar_field** (the informative family): easy CBS/PBS ~34 ms, PP
~1 ms; middle CBS ~560 ms; hard CBS ~15.7 s, PBS ~9.8 s, PP ~50 s (few successes).

**axis_ray hard:** all medians &lt; 300 ms (CBS/PBS); PP often fastest when it
succeeds (~18 ms on pillar_field hard).

### 4.3 Hard success cumulative by wall time

**plane_slice × pillar_field**

| \(T\) | CBS | PBS | PP |
|-------|-----|-----|-----|
| 5 s | 1% | 6% | 0% |
| 30 s | 70% | 51% | 2% |
| 60 s | 87% | 54% | 4% |
| 100 s | 93% | 56% | 7% |

**plane_slice × corridor_rack:** similar shape (CBS 88% @100 s, PBS 51%, PP 11%).

**axis_ray hard:** essentially solved by 5 s for all three algos (≥90–100%).

PBS/PP on plane_slice hard barely improve from 60 s → 100 s: structure / dead
orders dominate, not “slightly more time.”

### 4.4 Failure modes (hard)

| Dataset | CBS | PBS | PP |
|---------|-----|-----|-----|
| plane_slice × pillar_field | timeout 7 | timeout 41 + no_sol 3 | timeout 73 + no_sol 20 |
| plane_slice × corridor_rack | timeout 12 | timeout 46 + no_sol 3 | timeout 71 + no_sol 18 |
| axis_ray × pillar_field | — | no_sol 1 | no_sol 1 |
| axis_ray × corridor_rack | — | — | — |

### 4.5 Cost vs witness

Median `cost_ratio` on successes is **&lt; 1** on all non-trivial cells (often
~0.55–0.80 on hard). Constructive witnesses are loose; use for reference length,
**not** optimality gap. Cross-algo quality ranking on plane_slice hard remains
weak (few common successes when PP almost never finishes).

---

## 5. Discussion

### 5.1 What the data support

1. Permanent geometric multi-group routing is a **meaningful** stress test
   **when** the corridor generator creates hard interaction
   (`plane_slice` + many groups).  
2. Constraint-centric **CBS** transfers well on the hard cells
   (negative spheres on `ObstacleChain`).  
3. **PBS** (depth-first) and especially **PP** degrade on plane_slice hard —
   order/priority brittleness when prior pipes are hard obstacles.  
4. Easy / `axis_ray` alone would falsely suggest “all methods work.”  
5. Obstacle family (`pillar_field` vs `corridor_rack`) does **not** change the
   high-level ranking on plane_slice; both are useful for robustness of the claim.

### 5.2 What the data do *not* support

1. CBS is “optimal” or best possible — only best among {CBS, PBS, PP} here.  
2. That `axis_ray` hard is intrinsically easy for all future methods — only under
   the current size / group schedule.  
3. Topology / analyzer / low-level / PBS mode matter — **not ablated**.  
4. External CAD / ILP / published pipe routers — **no baseline**.

### 5.3 Likely bottlenecks (hypothesis → next engineering)

| Symptom | Hypothesis | Next probe |
|---------|------------|------------|
| PBS hard timeouts (plane_slice) | DF priority + bad early orders | Try `best_first`; log priority depth |
| PP `no_solution` | Sequence neighborhood exhausted | Inspect failed orders; stronger replanner |
| PP hard timeouts | Repeated full re-solves under bad prefixes | CAT soft guidance or better init order |
| CBS residual timeout | Deep CT / expensive replans | Conflict selection / bypass (later) |
| axis_ray saturated | Difficulty schedule too mild | Raise groups/terminals or density for a harder axis_ray tier |

---

## 6. Related work (sketch — to expand)

**MAPF high-level:** CBS, ICBS, ECBS, PBS, priority planning, LNS — we reuse
control logic, replace time–space conflicts with geometric dual-witness
conflicts and permanent occupancy.

**Pipe / cable / harness routing:** CAD sequential routing, Steiner trees in
obstacles, swarm / MOPSO waypoint methods (e.g. aeroengine MSPR) — typically
PP-like sequential or continuous optimization; rarely MAPF-style CT search over
groups.

**Positioning sentence (draft):**  
*We study multi-group 3D pipe routing as permanent-occupancy multi-agent design,
adapt CBS/PBS/PP to geometric clearance conflicts and topology-driven
multi-terminal groups, and evaluate on constructive 3D instances with painted
obstacles.*

---

## 7. Limitations & missing pieces for a paper

**Must-have before workshop-style draft**

- [x] Full matrix: axis_ray + plane_slice × both obstacles × levels × {cbs,pbs,pp}  
- [ ] At least **one ablation** (analyzer *or* topology *or* PBS mode)  
- [x] Formal problem draft in `papers/01_classical/latex/`  
- [ ] Expanded related work (cite MAPF + pipe / AM valves properly)  
- [x] One-command table script: `scripts/analyze_runs.py`  

**Nice-to-have**

- [ ] External baseline (greedy sequential, small ILP, or published heuristic)  
- [ ] Success@\(T\) plots as figures  
- [ ] Failure-case visualization set  
- [ ] Harder `axis_ray` tier if we want that generator to discriminate  

**Explicitly deferred (future work / `papers/02_advanced`)**

- CAT soft occupancy  
- Disjoint splitting / continuous partition  
- WDG / EWMVC high-level \(h\)  
- Any-angle path densify (`FIXBUG.md`)  
- LNS, Hybrid A\*, Steiner topology, window `[W]`

---

## 8. Recommended next steps (priority)

1. **One ablation** on plane_slice hard (PBS mode or conflict analyzer).  
2. Draft Method section (CBS/PBS/PP on this stack) + pull §4 tables into LaTeX.  
3. Cite AM valve + MAPF related work; freeze suite + `analyze_runs.py` in paper.  
4. Only then open `02_advanced` features if ablation leaves a clear gap.

---

## 9. Appendix — how stats were computed

`scripts/analyze_runs.py` over `data/runs/*.jsonl`:

- Parse stem `{algo}_{path_gen}_{obstacle}_{level}`  
- Success rate; median `total_time_ms` on successes; median `cost_ratio` on
  success + `witness_compared`  
- Hard `fail_reason` counts; hard success cumulative at 5/30/60/100 s  

Same field meanings as `view_report.py` Stats.

---

## Changelog

| Ver | Date | Notes |
|-----|------|-------|
| v0 | 2026-08-26 | First tech note; early plane_slice + pillar_field only |
| v0.1 | 2026-08-26 | Full 36-batch suite; `analyze_runs.py`; axis_ray easy vs plane_slice hard story |
