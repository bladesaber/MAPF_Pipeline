# topology — Connection Plan `[T]`

Turns multi-terminal **groups** into which pairs to route and in what order.  
Does **not** run grid search or resolve group–group conflicts.

Detail / backlog: [`ROADMAP.md`](ROADMAP.md). Input: [`problem/Problem.h`](../problem/Problem.h). Consumers: `planner/multi/` (`GroupSession`, `PrioritySequencer`), `planner/high/` (CBS / PBS).

---

## Mental model

```text
Problem (frozen)
  groups[].terminals
        │
        ▼
 TopologyBuilder::build(problem)     ← PureMST / RadiusSpine / RadiusPriorityMerge
        │
        ▼
 GlobalTopologyPlan
   └── GroupTopologyPlan { terminals, edges, grow_order }
        │
        ▼
 GroupSession::ordered_tasks()       ← grow_order matters for ConnectedSets
```

| Owns | Does not own |
|------|----------------|
| `GlobalTopologyPlan` (on builder) | `StaticClearanceGrid` (on `Problem`) |
| Edge set + `grow_order` | Routed `Path` / `ObstacleChain` |

`grow_order` is the lay sequence. **`GroupSession` merges same-group geometry into ConnectedSets as tasks finish** — later edges can attach to earlier ones. Wrong order (e.g. bridge after within-bin edges) breaks that.

---

## Files

| File | Role |
|------|------|
| `Types.h` | `Terminal`, `TopologyEdge`, `Task`, `GroupTopologyPlan`, `GlobalTopologyPlan` |
| `Mst.h` | Dense Prim: `n×n` cost map → `MstEdge[]` |
| `TopologyBuilder.h` | Base: `build(Problem)`, `replan()`, owns `plan_` |
| `PureMST.h` | Flat MST per group; `CostType` / `RadiusMerge`; Dijkstra cache |
| `RadiusSpineMSTBuilder.h` | Radius bins → bridge-to-prior → within-bin MST |
| `RadiusPriorityMergeBuilder.h` | Python `HyperRadiusGrowthTree` greedy merge |

All builders are header-only for now.

---

## Data contracts

```text
Terminal       = { id, pos, radius, heading?, role? }
TopologyEdge   = { id, u, v, radius, level?, kind? }   # Spine | Branch | Bridge
GroupTopologyPlan = { group_id, terminals[], edges[], grow_order[] }
GlobalTopologyPlan = { groups[] }

Task = { task_id, group_id, edge_id, status, path }
  enumerate_tasks()  → all edges (edges[] order)
  ordered_tasks()    → grow_order first, then remaining
```

Plans are **data**. Generation / LNS live on `TopologyBuilder`, not on the plan types.

---

## Builders

### Shared (`PureMST`)

- `RadiusMerge::{Min, Average}` — edge radius from endpoint radii (Python `merge_leafs`).
- `CostType::{Euclidean, Dijkstra, Volume}` — pairwise abstract cost  
  - `Euclidean`: `‖p_i−p_j‖`  
  - `Dijkstra`: true distance on `Problem::static_grid()` (cached per source cell)  
  - `Volume`: `dist / r²` (thicker preferred)
- `pair_cost` / `cost_map` / `get_dijkstra` — reuse in subclasses; call `clear_dijkstra_cache()` at `build` start.

### `PureMST`

One MST over **all** terminals in each group. Baseline.

### `RadiusSpineMSTBuilder` (`: PureMST`)

1. Bin by exact radius (thick → thin).  
2. Non-first bin: **Bridge** to union of all prior bins (nearest pair) — **first** in `grow_order`.  
3. If bin size ≥ 2: MST within bin (`Spine` / `Branch`). Size-1 = bridge only.  

Preferred hybrid when you want explicit spine / branch / bridge structure.

### `RadiusPriorityMergeBuilder` (`: PureMST`)

Faithful port of Python `HyperRadiusGrowthTree`:

1. Each terminal starts as its own component.  
2. Prefer max-radius component(s); if unique, second priority = next radius.  
3. Nearest terminal pair between those sets → edge; merge components (`RadiusMerge`).  
4. Repeat until one component. Edge order = `grow_order`.

Use this to match legacy Python topology.

---

## Usage sketch

```cpp
problem::Problem problem(grid, spheres, capsules, groups);  // bakes static grid

topology::RadiusSpineMSTBuilder builder(
    topology::RadiusMerge::Min,
    topology::CostType::Euclidean);
builder.build(problem);

const auto& global = builder.plan();
auto tasks = global.groups[0].ordered_tasks();  // → GroupSession
```

---

## Extension tips

- New algorithm: inherit `PureMST` (or `TopologyBuilder`), override `build`, reuse `pair_cost` / `cost_map` / `minimum_spanning_tree`.  
- Keep abstract costs cheap — no full A\* inside `[T]`. Dijkstra cost uses the **static** field only.  
- Preserve `grow_order` semantics for `GroupSession` (connect to already-laid geometry first).  
- LNS: override `TopologyBuilder::replan()`; plans stay replaceable data.
