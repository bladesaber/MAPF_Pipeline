# topology — Connection Plan (per group)

Detail companion to [`../ROADMAP.md`](../ROADMAP.md) §4.4.  
Consumers: [`planner/multi/`](../planner/multi/ROADMAP.md), [`conflict/`](../conflict/ROADMAP.md)
(optional Task ids on `Conflict` for `[T]` replan feedback).

---

## Why Topology

Low-level planners (A\*, Theta\*, …) answer a **single** query: one start → one goal on the grid.

A pipe **group** usually has **many terminals** that must form one connected network (MST / spine / Steiner; see main ROADMAP §1.4). There is no native “multi-endpoint A\*” that both chooses *which* pairs to link and routes them under permanent occupancy.

**Topology** fills that gap: on a cheap abstract graph it builds a **TopologyPlan** (which terminal pairs / edges + optional intra-group order). Multi-query drivers then run per-edge low-level search (**TopologyPlan + GroupSession / PrioritySequencer**). Without `[T]`, multi-terminal routing has nothing to decompose into path queries.

It is an **initialization** only: the edge set and `grow_order` may later be reshuffled (e.g. LNS re-entering `[T]`). It does not resolve group–group conflicts or run grid search.

---

## Role

**`[T]` = build TopologyPlan(s).**

Topology turns a group’s terminals into a **connection plan**: which pairs to route, at what radius/level, and an optional **intra-group grow order**.

It is a **cheap heuristic on an abstract graph** (Euclidean / volume / static estimates). It does **not** search the grid, update ConnectedSets, or choose **cross-group** Task order.

| Stage | Answers |
|-------|---------|
| Topology `[T]` | *What* to connect within a group → `GroupTopologyPlan` |
| `GroupSession` | *How* to route one group (fixed obstacles; ConnectedSets) |
| `PrioritySequencer` | *How* to route mixed groups under a global order |
| Low-level | *Where* each pair goes in space |

---

## Interface (target)

```text
Terminal     = { id, pos, radius, heading?, role? }   # heading: geom::Heading; none = unspecified
TopologyEdge = { id, u, v, radius, level?, kind? }     # kind: spine | branch | bridge

# Default [T] output — one group (GroupSession input)
GroupTopologyPlan = {
  group_id,
  terminals: Terminal[],
  edges: TopologyEdge[],       # unordered connection set
  grow_order: EdgeId[],        # intra-group lay hint (large radius → small)
}

# Aggregate for PrioritySequencer / Window — not a flat anonymous edge soup
GlobalTopologyPlan = {
  groups: GroupTopologyPlan[]
}

# Optional schedulable view (+ multi result slots; priority lives outside Topology)
Task = {
  task_id, group_id, edge_id,   # identity; u/v/radius via find_edge(edge_id)
  status, path                  # Ready/empty from [T]; filled by multi/ step_task()
}

GroupTopologyPlan::enumerate_tasks() -> Task[]   # all edges, edges[] order
GroupTopologyPlan::ordered_tasks()  -> Task[]   # grow_order then remaining (overrideable)

GlobalTopologyPlan::enumerate_tasks() -> Task[]  # concat groups' enumerate
GlobalTopologyPlan::ordered_tasks()  -> Task[]  # concat groups' ordered (overrideable)

# [T] generator — owns GlobalTopologyPlan for CBS / PP / …
TopologyBuilder {
  plan_: GlobalTopologyPlan
  build(problem: Problem)         # fill plan_ from problem.groups()
  replan()                        # default no-op; LNS / retopologize
  plan() -> GlobalTopologyPlan&
}
```

- `cost_fn` — abstract weight only (`dist`, `dist/r²`, …); never a full A\* call; lives on concrete builders.
- **`GroupSession`** uses `plan.ordered_tasks()`; writes each Task’s `status`/`path`.
- **`PrioritySequencer`** uses `enumerate_tasks()` for the job set; schedule from `order` or `ordered_tasks()` if `order` empty.
- **CBS / PP** read `TopologyBuilder::plan()`; do not own the plan.

Alias: in prose, “topology plan” for one group means `GroupTopologyPlan` unless “global” is stated.

---

## Default algorithm: Radius-spine

Preferred P1 hybrid (`RadiusSpineMSTBuilder`). Legacy Python greedy merge is `RadiusPriorityMergeBuilder` (`HyperRadiusGrowthTree`).

1. **Bin by radius** — discrete levels; largest radius = spine level.
2. **Within level** — MST / Prim on that level’s terminals (`cost_fn`).
3. **Bridge to all prior bins** — nearest terminal–terminal pair vs the **union** of thicker layers; append **before** within-bin MST edges (`grow_order`) so `GroupSession` ConnectedSets see prior layers first.
4. **Emit** `GroupTopologyPlan`: `edges` + `grow_order` (bridge then within-bin, thick → thin).
5. **Singleton bins** — size-1 bins skip within-bin MST; only a Bridge to prior layers (first bin alone waits for the next bridge).
Same-group branching / sharing follows ROADMAP §1.4; topology only proposes the multi-terminal graph.

---

## Catalog

| Builder / piece | Priority | Notes |
|-----------------|----------|-------|
| Types: `Terminal`, `TopologyEdge`, `GroupTopologyPlan`, `GlobalTopologyPlan` | P0 | Stable contracts |
| `Mst.h` util | P0 | Dense Prim: cost map → `MstEdge[]` |
| `TopologyBuilder` | P0 | Owns `GlobalTopologyPlan`; `build` / `replan` |
| `PureMST` | P0 | Baseline builder; `CostType`: Euclidean / Dijkstra / Volume |
| ★ `RadiusSpineMSTBuilder` | P1 | Levels + MST + bridge; `: PureMST` |
| ★ `RadiusPriorityMergeBuilder` | P1 | Python `HyperRadiusGrowthTree` greedy merge |
| Cost: `dist` / `dist/r²` (volume-priority) | P0 | On `PureMST` via `CostType::Volume` |
| Cost: Dijkstra on `Problem::static_grid` | P0 | Via `heuristic::Dijkstra` |
| Bridge: terminal–terminal | P1 | Match current Python behavior |
| Bridge: attach to trunk path | P2 | Shorter trees; needs soft geometry |
| Steiner / RSMT-style | P2 | Better topology, heavier |
| Schematic constraints | P2–P3 | Inlet/outlet hop / cluster rules |
| LNS may rebuild plan | P2+ | High-level `[F]` re-enters `[T]` |

---

## Layout (target)

```text
include/topology/
  README.md            # developer quick start (this folder)
  ROADMAP.md           # design detail / backlog
  Types.h              # Terminal, TopologyEdge, GroupTopologyPlan, GlobalTopologyPlan, Task
  Mst.h                # dense Prim util (cost map → edges)
  TopologyBuilder.h    # base; owns GlobalTopologyPlan
  PureMST.h            # baseline: one MST per group; virtual cost_map
  RadiusSpineMSTBuilder.h  # hybrid: radius bins + MST + bridge
  RadiusPriorityMergeBuilder.h  # Python HyperRadiusGrowthTree greedy merge

src/topology/          # mirrors headers (builders later)
```
