# Tech Note v0 — Multi-Group Pipe Routing under Permanent Geometric Occupancy

**Status:** internal draft (not a submission)  
**Date:** 2026-08-26  
**Code:** `pipe_design/`  
**Primary results:** early pillar_field plane_slice runs; full suite under
`pipe_design/data/runs/` (see `run_bench.sh`).

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
2. Paint static obstacles in remaining free space (**pillar_field** in this note)  
3. Emit YAML: environment + witness `solution`  

Generators:

| Name | Idea | In this result folder? |
|------|------|------------------------|
| `plane_slice` | Planar A\* on axis slices | **Yes** |
| `axis_ray` | LOS ray tree (preferred generator) | **No — TODO** |

Difficulty knobs (see `gen_robust.sh`): group count × terminals per group
(plane_slice easy/middle/hard ≈ 3×3 / 6×4 / 9×6).

Witness proves **feasibility of the generator**, not optimality. Solver
`cost_ratio = total_cost / reference_cost` can be **&lt; 1**.

### 3.2 Runner / report

- Binary: `pipe_run_design`  
- One DesignReport JSON object per instance → JSONL batch  
- Stats style: success rate, fail_reason, `total_time_ms` five-number summary,
  `cost_ratio` on success+witness (same spirit as `scripts/view_report.py` Stats)

### 3.3 Experiment config (pillar_field / plane_slice)

| Level | N | `max_time_ms` |
|-------|---|---------------|
| easy | 150 | 30 000 |
| middle | 150 | 60 000 |
| hard | 138 | 100 000 |

Shared: `topology=pure_mst`, `low_model=astar`, `heuristic=manhattan`,
`action_model=TwentySixConnected`, `budget_mode=time`.

Data path (current suite): `pipe_design/data/envs/` and JSONL under
`pipe_design/data/runs/`.

---

## 4. Results (plane_slice + pillar_field)

### 4.1 Success rate

| Level | CBS | PBS | PP |
|-------|-----|-----|-----|
| easy | **100.0%** | 100.0% | 100.0% |
| middle | **100.0%** | 96.0% | 95.3% |
| hard | **97.1%** | 52.9% | 11.6% |

**Takeaway:** easy is saturated (no ranking power). Hard separates algorithms
clearly: **CBS ≫ PBS ≫ PP** under the same stack.

### 4.2 Runtime (median `total_time_ms`, successes only)

| Level | CBS | PBS | PP |
|-------|-----|-----|-----|
| easy | 8.2 | 8.3 | **0.8** |
| middle | **78** | 101 | 251 |
| hard | **4 253** | 6 160 | 72 313 |

On easy, PP is cheapest when order luck works. On hard, PP spends most of the
budget and still fails.

### 4.3 Hard: cumulative success by wall time

Approximate success if solved within \(T\) (failures counted as \(+\infty\)):

| \(T\) | CBS | PBS | PP |
|-------|-----|-----|-----|
| 5 s | 52.9% | 21.0% | 0.0% |
| 30 s | 89.9% | 48.6% | 0.7% |
| 60 s | 95.7% | 52.2% | 5.1% |
| 100 s | 97.1% | 52.9% | 11.6% |

PBS/PP barely improve from 60 s → 100 s: failures are not “just need more time”
in a simple way (search structure / dead orders dominate).

### 4.4 Failure modes (hard)

| Algo | timeout | no_solution | notes |
|------|---------|-------------|-------|
| CBS | 4 | 0 | almost always finishes |
| PBS | 60 | 5 | depth-first priority search stalls |
| PP | 87 | 35 | many exhaustive order failures |

Middle failures for PBS/PP are almost entirely `no_solution` (6–7 instances,
largely overlapping).

### 4.5 Cost vs witness

On middle/hard **successful** runs, median `cost_ratio` ≈ **0.90 / 0.80**
(all three algos). Hard CBS: **134/134** successes have `cost_ratio < 1`.

**Interpretation:** constructive corridors are loose feasibility witnesses;
grid search often finds shorter trees. Use witness for regression / reference
length, **not** as an optimality gap.

### 4.6 Paired hard successes

| Set | Count |
|-----|------:|
| CBS ∩ PBS ∩ PP success | 13 |
| CBS success, PBS fail | 64 |
| PBS success, CBS fail | 3 |
| All three fail | 1 |

Quality ranking across algos on hard is weak (only 13 common successes). Primary
metric for this note remains **success under budget**.

---

## 5. Discussion

### 5.1 What the data support

1. Permanent geometric multi-group routing is a **meaningful** stress test:
   difficulty scales with groups/terminals under pillar clutter.  
2. Constraint-centric **CBS** transfers well when conflicts become negative
   spheres on `ObstacleChain`.  
3. **PBS** (depth-first) and especially **PP** degrade sharply when many groups
   interact — consistent with order/priority brittleness when prior pipes are
   hard obstacles.  
4. Easy instances alone would falsely suggest “all methods work.”

### 5.2 What the data do *not* support

1. CBS is “optimal” or best possible — only best among {CBS, PBS, PP} here.  
2. Results generalize to `axis_ray` or other obstacle painters — **not run**.  
3. Topology / analyzer / low-level choices matter — **not ablated**.  
4. External CAD / ILP / published pipe routers — **no baseline**.

### 5.3 Likely bottlenecks (hypothesis → next engineering)

| Symptom | Hypothesis | Next probe |
|---------|------------|------------|
| PBS hard timeouts | DF priority expansion + bad early orders | Try `best_first`; log priority depth |
| PP `no_solution` | Sequence neighborhood exhausted | Inspect failed orders; stronger replanner |
| PP hard timeouts | Repeated full re-solves under bad prefixes | CAT-style soft guidance or better init order |
| CBS rare timeout | Deep CT / expensive replans | Conflict selection / bypass (later) |

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

- [ ] Run **axis_ray** under the same algos / budgets (second generator)  
- [ ] At least **one ablation** (analyzer *or* topology *or* PBS mode)  
- [ ] Formal problem definition section (math notation for conflict / freeness)  
- [ ] Expanded related work (cite MAPF + pipe routing properly)  
- [ ] Reproducibility: relative paths, seed list, one-command table script  

**Nice-to-have**

- [ ] External baseline (greedy sequential, small ILP, or published heuristic)  
- [ ] Success@\(T\) plots as figures  
- [ ] Failure-case visualization set  

**Explicitly deferred (future work, not blockers for v0 story)**

- CAT soft occupancy  
- Disjoint splitting / continuous partition  
- WDG / EWMVC high-level \(h\)  
- Any-angle path densify (`FIXBUG.md`)  
- LNS, Hybrid A\*, Steiner topology, window `[W]`

---

## 8. Recommended next steps (priority)

1. **Keep this note updated** as results land (treat as living skeleton).  
2. **Finish experiment matrix:** axis_ray × {easy,middle,hard} × {cbs,pbs,pp}.  
3. **One ablation** driven by §5.3 (PBS mode or conflict analyzer).  
4. **Short architecture appendix** (optional notebook): how to reproduce Table 4.1
   from JSONL — not a substitute for this note.  
5. Only then invest in CAT / densify / WDG if the second generator confirms the
   same ranking and a clear failure mode remains.

---

## 9. Appendix — how stats were computed

For each JSONL batch (same fields as `view_report.py` Stats):

- `N`, `success` count / rate  
- `fail_reason` histogram on failures  
- five-number summary of `total_time_ms` (all / success-only)  
- five-number summary of `cost_ratio` where `success ∧ witness_compared`  

Paired hard sets keyed by `instance_id`.

---

## Changelog

| Ver | Date | Notes |
|-----|------|-------|
| v0 | 2026-08-26 | First tech note; plane_slice + pillar_field only |
