# benchmark — Constructive instance generation

Companion to [`../../ROADMAP.md`](../../ROADMAP.md) §1.

**Primary deliverable:** a self-contained **YAML** instance file with full environment
detail **and** a **valid witness solution** (constructive corridors). Loaders map the
problem half into `problem::Problem`; the witness proves feasibility and supports
regression / debug. Solvers need not re-prove that a solution exists.

Generation order: corridors first (terminals + witness paths), then static obstacles
in the remaining free space (never blocking the witness).

**In scope:** environment parameters only.  
**Out of scope:** topology policy, kinematics, radii mix.

All distances below are in **grid cells** (`resolution = 1.0`).

---

## 0. Pipeline

```text
params (seed, r, groups, …)
  → [1] Corridor generation   # Method 3 (preferred) | plane_slice/ (Method 1) | Method 2 (future)
  → [2] Obstacle generation   # layout family + density (outside corridors)
  → [3] Emit YAML             # full instance + valid witness solution
```

**Code layout:**

```text
include/benchmark/
  GeneratorTypes.h          # base GenParams, GroupCorridors, Instance (shared)
  GeneratorUtils.h          # shared terminal / bake helpers
  Subdomain.h               # N×N×N partition (shared by all generators)
  plane_slice/              # Method 1
    Types.h                 # extends GenParams (+ Plane, AxisKind)
    Utils.h                 # on_plane, intersects_plane, …
    PlaneSlicedGenerator.h
  <method2>/ …              # future
src/benchmark/
  Subdomain.cpp
  plane_slice/ …            # mirrors
```

Each generator folder has its own `Types.h` (derived `GenParams` + method-only
types) and one generator class. Shared corridor result shape lives in
`GeneratorTypes.h` for obstacle paint / YAML.

---

## Global rules

| Rule | Value / meaning |
|------|-----------------|
| Pipe radius | **Uniform** — every terminal uses the same `r` |
| Grid resolution | `1.0` |
| Shell | `shell_cells = ceil((shell_margin + radius) / res)`; plan on core `size-2m`, then extend face terminals to the full surface |
| Terminal placement | On domain face: at least one of `{x,y,z}` equals `0` or `max` |
| Terminal gap | Face terminals: Chebyshev `> terminal_center_gap()` where
| | `terminal_center_gap = ceil(2r/res) + terminal_gap` (`terminal_gap` = slack) |

**Method 1 / 2 only (subdomains):** domain split `N×N×N` (`N ∈ {3,4,5}`); same-group
at most one terminal per subdomain; cross-group share allowed; max terminals / group
≤ # surface subdomains.

Obstacle generation must not violate corridor clearance for `r`.

Shared conventions:

- Occupied corridors (any group, inflated by `r` / gaps) block later growth.
- Groups advance in **round-robin** at attach-step granularity: `G0 → G1 → G0 → …`.
- Goal: diverse **tree-like** witnesses, not shortest paths.

---

## 1. Corridor generation

### 1.A Method 1 — plane-sliced multi-source growth

**Code:** `benchmark/plane_slice/` → class `PlaneSlicedGenerator`.  
**Status:** implemented; Method 3 is the preferred simpler alternative going forward.

Grow one tree-like witness per group by attaching new face terminals on axis-aligned
planes. Same-group corridors **off the active plane are ignored** during search
(diversity; same-group crossing OK).

**Hard rule — planar A\* only:** every A\* call in Method 1 searches **only the active
axis plane** (2D slice), **not** the full 3D grid. 3D detours are out of scope; if the
slice is disconnected, **retry** another plane / G / axis — do not fall back to 3D A\*.
Feasibility means *planar-step* connectability under inter-group obstacles on that slice.
That is a multi-plane tree.

#### 1.A.1 Axis plane

An **axis plane** is one of `{xy, xz, yz}` with fixed coordinate on the third axis,
e.g. xy-plane at `z = z_0`.

#### 1.A.2 Per-group plane memory + plane gap

For each group, record every plane used (`axis` + fixed coordinate).

**Plane gap rule:** any two planes of the **same group** and **same axis type**
(e.g. two xy-planes at `z_i`, `z_j`) must satisfy

```text
|fixed_i - fixed_j| > plane_center_gap
```

where `plane_center_gap = ceil(2r/res) + plane_gap` (`plane_gap` = slack).
Different axis types are not compared.

#### 1.A.3 Plane validity

Example: axis `xy`, fixed `z = z_0`.

1. List all subdomains that intersect this plane.
2. If **every** such subdomain already has a terminal (no free slot for a new face
   terminal under global rules + same-group subdomain rule), the sample is **invalid**.
3. Also require: **plane gap** vs this group’s recorded planes of the same axis type.

Start occupancy is **not** checked here. Seed places S/G on the new plane; attach
(Step 2) derives the plane from an existing path cell, so a multi-source start set
exists by construction.

#### 1.A.4 Step 1 — seed segment

1. Select `P_1` ∈ `{xy, xz, yz}`; sample fixed coordinate so the plane is **valid**;
   record it.
2. Choose two **surface** subdomains intersecting the plane that can host terminals.
3. Place **S** and **G** on domain faces in those subdomains (same fixed coordinate).
4. **Planar** A\* on `P_1` only (other groups’ corridors ∩ plane = obstacles).
   Store path; terminal count = 2.

#### 1.A.5 Step 2 — attach one terminal

1. Select axis type `P_t` **≠** this group’s last plane `P_{t-1}`.
2. From this group’s existing path cells, sample a point that can define a plane of
   type `P_t` (fixed = that cell’s coordinate on the free axis) and whose plane is
   **valid** (gap + free surface subdomain for a new **G**). Reject points that fail.
3. **Multi-source starts:** all of this group’s occupied cells **on this plane**
   (non-empty because the plane was taken from a path cell).
4. **Obstacles:** corridors of **other** groups (inflated by `r`).
5. Pick new **G** from a valid surface subdomain on the plane.
6. **Planar** multi-source A\* on this plane only → on success append path, record
   plane, add terminal.

#### 1.A.6 Step 3

Repeat Step 2 until `n_g` terminals (or no valid plane/G).

---

### 1.B Method 2 — waypoint tree growth (future alternative)

**Status:** deferred until Method 1 is stable. Use if plane-sliced paths are too
monotone / planar and more Steiner-like branching is needed.

Idea: grow an explicit **tree** with interior **waypoints** (Steiner-like hubs), not
only face-to-face plane slices. Waypoints need not lie on domain faces; **terminals**
still obey global face + subdomain + gap rules.

#### 1.B.1 Step 1 — seed with one waypoint (two segments)

1. Sample face terminal **S** in a valid surface subdomain.
2. Sample an interior **waypoint** `W` in a different subdomain (not required to be on
   a face; must keep clearance vs other groups’ corridors and `terminal_gap` vs existing
   terminals if `W` is treated as a junction sample).
3. Sample face terminal **G** in another valid surface subdomain.
4. A\* two segments: `S → W` and `W → G` (other groups’ corridors = obstacles).
5. **Loop removal:** if the two segments overlap in cells (or revisit a junction),
   trim / reject the overlapping loop so the kept geometry is a simple tree edge pair
   meeting at `W` (exact predicate TBD at implement time).
6. Store both segments as the group’s corridor; terminal count = 2; record `W` as a
   junction of the group tree.

#### 1.B.2 Step 2 — attach via a new waypoint

1. Sample a new waypoint `W'` (subdomain / clearance / diversity rules TBD).
2. **Multi-source starts:** all of this group’s **existing** occupied cells (full tree,
   not plane-restricted) — or connect only from existing junctions/terminals (TBD).
3. A\* from that start set to `W'` (other groups = obstacles).
4. Optionally place a new face terminal **G'** and A\* `W' → G'`, or defer the new
   terminal to a later attach that ends on a face (must reach `n_g` face terminals).
5. On success: append path(s), update tree; increase face-terminal count when a new
   face **G** is added.

#### 1.B.3 Step 3

Repeat Step 2 until the group has `n_g` face terminals (or sampling/A\* exhausted).

#### 1.B.4 Notes vs Method 1

| | Method 1 | Method 2 |
|---|----------|----------|
| Branching cue | Axis planes + multi-source on plane | Explicit waypoints |
| A\* domain | **Active plane only** (no full-grid 3D) | Typically full tree / 3D (TBD) |
| Same-group ignore | Off-plane corridors ignored | Full tree usually used as attach base |
| Complexity | Lower | Higher (loop removal, waypoint validity) |
| When to use | Legacy planar generator | If still needed after Method 3 |

Open at implement time: waypoint subdomain policy, whether `W` counts toward
`terminal_gap`, and precise loop-removal geometry.

---

### 1.C Method 3 — axis-aligned ray tree (preferred)

**Code:** `benchmark/axis_ray/` → `AxisRayGenerator`. **No subdomain partition.** No A\*.

Grow an axis-aligned polyline tree by **line-of-sight (LOS) scans**. Params:
`init_waypoints_num`, `attach_waypoints_num`, `min_path_size`, `max_path_size`
(diameter multiples → `min/max_path_length()` cells), `path_gap`, `branch_gap`.

**Waypoints:** turning points, face terminals, and split points.  
**Rollback:** failed seed/attach attempt restores occupied / paths / waypoints.

#### Hard rules

1. **Ray block:** never enter any written path cell (any group). Separately,
   Chebyshev to occupied must be `> path_gap`, except occupied cells on the
   **spine line** through the ray origin (same axis as the edge being left;
   requires `branch_gap > path_gap`). Other lines / groups still enforce `path_gap`.
2. **Turn:** `D_t ∈ {±x,±y,±z}` must **not** share the axis of `D_{t-1}` (first
   segment: any direction).
3. **Interior length:** longest free LOS = `L`; reject if `L < min_path_length()`;
   else pick length ∈ `[min_path_length(), min(L, max_path_length())]`
   (`*_path_length()` = `radius_diameter_cells() * *_path_size`).
4. **Last segment** (seed or attach): **no** `max_path_length()`; extend to a domain
   **face**; still check radius / `path_gap` along the segment.
5. **Branch point:** on the current tree; distance to every existing **waypoint**
   `> branch_gap`. Require **`branch_gap > path_gap`** (ctor-enforced).
6. **Branch direction:** if the point is **not** a waypoint, it lies on a single-axis
   edge — new `D` must **not** use that edge’s axis (side exit only).

#### Step 1 — seed chain

1. Sample first **face** terminal (surface; `terminal_gap` / clearance OK).
2. Add `init_waypoints_num - 2` interior segments (rules 2–3); if `init_waypoints_num==2`,
   skip (single face→face segment).
3. Final segment: rule 4 (hit surface). Record turning points + terminals as waypoints.

#### Step 2 — attach branch

1. Pick a path point with distance to all waypoints `> branch_gap`.
2. Pick `D_t` (rules 2, 6).
3. Add `attach_waypoints_num - 2` interior segments, then last segment to surface
   (rule 4). Promote new face terminal / turns / split to waypoints.

Repeat Step 2 (round-robin across groups) until each group has enough face terminals
or retries / stack exhausted.

| | Method 1 | Method 3 |
|---|----------|----------|
| Search | Planar A\* | Axis LOS scan |
| Subdomains | Yes | **No** |
| Path cross | Same-group may cross | **No** path may cross any written path |

---

## 2. Obstacle generation

**Goal:** paint static obstacles in cells **outside** the union of all groups’
thickened corridor volumes so witnesses stay feasible. Two knobs, same stage:

- **Layout family** — *pattern* of obstacles (where structure sits).
- **Density** — *how much* of the non-corridor volume is blocked.

**Invariant:** never block a witness corridor cell (or its `r`-neighborhood required
by occupancy).

### 2.1 Layout family

Structural obstacle pattern (draft; fill in later):

| Family (draft names) | Intent |
|----------------------|--------|
| `EmptyBox` | No / minimal structural obstacles (density may still add clutter) |
| `PillarField` | Vertical pillars |
| `CorridorRack` | Parallel racks / layer channels |
| `ManifoldEnvelope` | Equipment blocks; ports remain on domain faces |

Family parameters (pillar count, rack spacing, …) are separate from density.

### 2.2 Density

How aggressively to fill eligible (non-corridor) cells after the family pattern is
applied (draft; fill in later):

| Topic | Options / questions |
|-------|---------------------|
| Definition | Fraction of **non-corridor** cells blocked, vs whole domain |
| Sweep | Discrete levels e.g. `{0, 0.2, 0.4, 0.6}` |
| Interaction | Family places structured obstacles; density adds residual clutter (or scales family) |

---

## 3. Emit — YAML instance + valid solution

**Agreed:** generator output is **YAML**, not an in-memory `Problem` alone.

Each file must be enough to reproduce and check the instance:

| Block | Contents |
|-------|----------|
| `meta` | `seed`, generator method, layout family, density, `r`, gaps, … |
| `grid` | size, `resolution: 1.0`, … → `GridParams` |
| `obstacles` | spheres / capsules (post §2) |
| `groups` | group id + face terminals (`id`, `pos`, `radius`, …) |
| `solution` | **valid witness**: per-group corridor paths (waypoints if any) |

**Rules:**

1. `solution` is **mandatory** for generated benchmarks — constructive proof of
   feasibility under global rules and clearance `r`.
2. After obstacle paint, witness paths must remain free (`clearance ≥ r`).
3. Runtime solvers consume `grid` + `obstacles` + `groups` via `problem::Problem`;
   `solution` is for validation, baselines, and debugging (not required as a warm
   start unless a runner opts in).

Exact YAML schema: TBD at implement time; keep it stable once tests depend on it.

---

## 4. Future plan — radius-aware gaps

Today most generator knobs are **cell counts** and only loosely related to `radius`
(`terminal_gap`, `path_gap`, `branch_gap`, Method 1 `plane_gap`, LOS / occupancy
checks). Witnesses can be cell-feasible yet **geometrically** invalid once pipes are
spheres/capsules of radius `r` (or when `r` changes under fixed gaps).

**Later (not blocking §1–§3):**

1. Derive or validate gaps from `r` (and `resolution`), e.g. require
   `terminal_gap`, `path_gap`, `branch_gap`, `plane_gap` large enough that centerline
   separation implies `clearance ≥ r` (typical floor ≈ ceil of a function of `2r`).
2. Make LOS / occupancy **radius-aware**: inflate written paths by `r` (or query
   `ObstacleChain` / clearance) instead of Chebyshev-to-cells alone.
3. When painting obstacles (§2), keep `clearance ≥ r` on the witness tubes explicitly.
4. Optional: expose gaps as “margin above `2r`” so sweeping `r` does not silently
   break constructive feasibility.

Until then: treat current integer gaps as **provisional** under a fixed `r`; re-tune
or replace them before serious multi-`r` experiments.
