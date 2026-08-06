# topology — Connection Plan (per group)

Detail companion to [`../ROADMAP.md`](../ROADMAP.md) §4.4.  
Consumers: [`planner/multi/`](../planner/multi/ROADMAP.md) (`GroupSession`, `PrioritySequencer`).

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
GroupTopologyPlan::replan()                     # default no-op; LNS hook

GlobalTopologyPlan::enumerate_tasks() -> Task[]  # concat groups' enumerate
GlobalTopologyPlan::ordered_tasks()  -> Task[]  # concat groups' ordered (overrideable)
GlobalTopologyPlan::replan()                    # default no-op (not group::replan loop)

TopologyBuilder::build(terminals, cost_fn) -> GroupTopologyPlan
# Assemble: GlobalTopologyPlan{ groups: [plan_A, plan_B, …] }
```

- `cost_fn` — abstract weight only (`dist`, `dist·r²`, …); never a full A\* call.
- **`GroupSession`** uses `plan.ordered_tasks()`; writes each Task’s `status`/`path`.
- **`PrioritySequencer`** uses `enumerate_tasks()` for the job set; schedule from `order` or `ordered_tasks()` if `order` empty.

Alias: in prose, “topology plan” for one group means `GroupTopologyPlan` unless “global” is stated.

---

## Default algorithm: Radius-spine

Preferred P1 builder (`RadiusSpineTopology`), aligned with legacy `HyperRadiusGrowthTree`:

1. **Bin by radius** — discrete levels; largest radius = spine level.
2. **Within level** — MST / Prim on that level’s terminals (`cost_fn`).
3. **Bridge to previous** — attach each new component to the already-connected set (`main_leaf` / prior levels): nearest terminal–terminal pair (P1); optionally tap onto routed trunk geometry later (P2).
4. **Emit** `GroupTopologyPlan`: `edges` + `grow_order` (level descending).

Same-group branching / sharing follows ROADMAP §1.4; topology only proposes the multi-terminal graph.

---

## Catalog

| Builder / piece | Priority | Notes |
|-----------------|----------|-------|
| Types: `Terminal`, `TopologyEdge`, `GroupTopologyPlan`, `GlobalTopologyPlan` | P0 | Stable contracts |
| `PureMST` | P0 | Baseline; all terminals, one MST |
| ★ `RadiusSpineTopology` | P1 | Levels + MST + bridge; default |
| Cost: `dist` / `dist·r²` (volume) | P1 | Volume often better for thick pipes |
| Bridge: terminal–terminal | P1 | Match current Python behavior |
| Bridge: attach to trunk path | P2 | Shorter trees; needs soft geometry |
| Steiner / RSMT-style | P2 | Better topology, heavier |
| Schematic constraints | P2–P3 | Inlet/outlet hop / cluster rules |
| LNS may rebuild plan | P2+ | High-level `[F]` re-enters `[T]` |

---

## Layout (target)

```text
include/topology/
  ROADMAP.md           # this file
  Types.h              # Terminal, TopologyEdge, GroupTopologyPlan, GlobalTopologyPlan, Task
  # TopologyBuilder.h, PureMST.h, RadiusSpineTopology.h (planned)

src/topology/          # mirrors headers (builders later)
```
