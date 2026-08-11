# planner/multi — Multi-query session drivers

Runs **many** start→goal queries from a TopologyPlan while updating session state.  
Not high-level conflict resolution (CBS / PP / LNS → `planner/high/`).

Detail / backlog: [`ROADMAP.md`](ROADMAP.md). Plans: [`topology/`](../../topology/README.md). Grid: [`grid/`](../../grid/README.md).

---

## Mental model

```text
TopologyPlan (+ grow_order / task order)
        │
        ▼
 MultiQueryPlanner::reset → step_task / search
        │
   ┌────┴────┐
   ▼         ▼
GroupSession   PrioritySequencer
 one group      mixed groups
 obstacles      same-group → ConnectedSets
 fixed          other-group → ObstacleChain
        │
        ▼
 LowLevelPlanner (A*) per Task
```

| After a successful path | `GroupSession` | `PrioritySequencer` |
|-------------------------|----------------|---------------------|
| Same group | → ConnectedSet (starts/goals) | → ConnectedSet |
| Other group | unchanged (already in entry grid) | → obstacles for later tasks |

Cross-group **Task order** is irrelevant for a lone `GroupSession` (CBS leaf).  
It **defines** the solution for `PrioritySequencer` (PP).

---

## Files

| File | Role |
|------|------|
| `MultiQueryPlanner.h` | Base: `step_task` / `search` / `status` / `expansions` / `last_path` |
| `GroupSession.h` + `.cpp` | One group; fixed `CompositeGrid`; ConnectedSets |
| `PrioritySequencer.h` + `.cpp` | Multi-group order; temp grid view; foreign paths → chain |

---

## Control API (parallel to low-level)

| Low (`LowLevelPlanner`) | Multi (`MultiQueryPlanner`) |
|-------------------------|-----------------------------|
| `reset` / `step` / `search` | `reset` / `step_task` / `search` |
| `path()` | `last_path()` + derived `tasks()` copy |
| `expansions_` | sum of low-level expansions |

**Do not** inherit `LowLevelPlanner` — arity and result shape differ.

### `step_task()`

1. Resolve edge `u`/`v` → ConnectedSet cells as starts/goals  
2. `low.reset(...)` → `low.search()`  
3. On success: write `Task.status` / `Task.path`, commit (merge ConnectedSet; PP also records foreign obstacles)  
4. Advance cursor → `Running` / `Success` / `Failure`

Fine-grained `step()` (one low expansion) is planned for debug/viz only.

---

## ConnectedSets

Each terminal starts as its own component (`{world_to_cell(pos)}`).  
After a successful path `u→v`, `merge_path` unions the two components and inserts interior path cells. Later tasks whose endpoints share a component can start/goal from the **whole** laid geometry (branching / sharing).

**Implication for `[T]`:** `grow_order` must lay bridges / spine so that later edges see prior geometry. See `topology/README.md`.

---

## GroupSession

- Input: one `GroupTopologyPlan` via `plan.ordered_tasks()`.  
- `grid` is **read-only** for the whole session.  
- Same-group only: every commit is ConnectedSet merge.

```cpp
GroupSession session;
session.reset(plan, grid, low, heuristic, open, actions);
session.search();
auto results = session.tasks();  // copy; safe after another reset
```

Typical caller: CBS node replan / single-group solve.

---

## PrioritySequencer

- Input: `GlobalTopologyPlan` + optional `order` (task_id list).  
  Empty order overload → `plans.ordered_tasks()`.  
- Entry `CompositeGrid` is **not** mutated. Each `step_task` builds a **temporary** view:  
  `static_grid` + `base_chain` + capsules/spheres from **other groups'** committed paths.  
- Same-group paths → ConnectedSets only (not obstacles to self).

```cpp
PrioritySequencer seq;
seq.reset(plans, /*order=*/pp_task_ids, grid, low, heuristic, open, actions);
seq.search();
```

Typical caller: PP / priority sequential design.

---

## Ownership

```text
Caller owns:  TopologyPlan(s), CompositeGrid, LowLevelPlanner, OpenSet, Heuristic, ActionModel
Multi owns:   task cursor, ConnectedSets, Task results, status_ / expansions_
              (PP: committed_ list for foreign obstacle rebuild)
```

Terminal IDs must be unique across groups in a `PrioritySequencer` session.

---

## Extension tips

- New driver: inherit `MultiQueryPlanner`, implement `reset` / `step_task` / `last_path` (+ `tasks()`).  
- Keep entry grid immutable; PP-style extras go on a temp `ObstacleChain` / `CompositeGrid` view.  
- `expansions_` should accumulate `low.expansions()` after each search.  
- High-level (CBS/PP) should **re-invoke** multi after changing chain or task order — not reinvent ConnectedSets inside high/.
