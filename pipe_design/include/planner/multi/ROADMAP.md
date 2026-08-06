# planner/multi — Multi-query session drivers

Detail companion to [`../../ROADMAP.md`](../../ROADMAP.md) §2 / §8.

Drivers that wrap a `LowLevelPlanner` and run **many** start–goal queries while mutating
session state (connected sets, obstacles). They are still planners — not high-level conflict
resolution (CBS/PBS/LNS live in `planner/high/`).

Legacy analogy: `pathSearch_pkg(legacy)` `SetPlanner` / `SetPairNodePlanner` (logic only).

---

## Why multi/

Low-level answers **one** query. A group (or PBS schedule) needs **many** queries where:

- **Same-group** previous paths update **starts / goals** (connected sets → tree / sharing).
- **Other-group** previous paths become **obstacles** (PBS), or stay fixed in the grid view (CBS group replan).

Without this layer, TopologyPlan edges have no session that turns them into sequenced searches.

---

## Role

| Driver | Scope | Typical caller |
|--------|--------|----------------|
| ★ `GroupSession` | One group; obstacles **fixed** for the session | CBS node replan; single-group solve |
| ★ `PrioritySequencer` | Mix of groups; after each path, update context by group | PBS; priority-based sequential design |

**Contract line:** `[T]` builds `TopologyPlan`(s); **multi/** routes them; **high/** resolves residual group conflicts by changing chain / priority and re-invoking multi/.

---

## API alignment with low-level

**Agree: mirror the low-level steppable contract** so callers (CBS leaf, benches, tools) drive multi the same way they drive A\*.

| Low (`LowLevelPlanner`) | Multi (`MultiQueryPlanner`) |
|-------------------------|-----------------------------|
| `SearchStatus` | **same** enum (`Ready` / `Running` / `Success` / `Failure`) |
| `reset(...)` | `reset(...)` — bind plan + context + low (+ open/heuristic/actions as needed) |
| `step()` | `step()` — advance **one Task** (see below) |
| `search(max_expansions)` | `search(max_expansions)` — loop `step` while `Running` |
| `status()` / `expansions()` | same; `expansions_` = **sum of low-level expansions** |
| `path()` | `last_path()` — last finished Task path |
| — | derived `tasks()` copy — full Task results (reuse-safe) |

Do **not** inherit `LowLevelPlanner`: `reset` arity and `paths()` differ. Prefer a **parallel base** with the same control surface:

```text
class MultiQueryPlanner {   # parallel to LowLevelPlanner
  virtual SearchStatus step() = 0;
  virtual Path last_path() const = 0;
  # derived: tasks() -> vector<Task> copy
  virtual SearchStatus search(max_expansions = kUnlimited);
  SearchStatus status() const;
  std::size_t expansions() const;
protected:
  SearchStatus status_;
  std::size_t expansions_;                 # aggregate low expansions
};
```

### What one `step()` means

Default (P1): **one Task** = prepare starts/goals (and maybe ctx) → `low.reset(...)` → `low.search()` to completion → on success, commit (`after_task`: ConnectedSet / obstacles) → bump task cursor.

- If low fails → session `Failure`.
- If no tasks left after commit → session `Success`.
- While tasks remain → stay `Running`.

Optional later: `step_low()` / fine-grained mode that forwards a single `low.step()` for visualization; not required for CBS/PBS.

### Ownership (same spirit as low)

```text
Caller owns:  TopologyPlan(s), CompositeGrid (read-only entry), LowLevelPlanner, OpenSet, Heuristic, ActionModel
Multi owns:   session cursor, ConnectedSets, Task list (results on Task), status_/expansions_
```

Per-Task geometry lives on **`Task.status` / `Task.path`**. PBS builds a temporary `CompositeGrid(static, chain)` per step — entry grid is never mutated.

---

## Interface (target)

```text

GroupTopologyPlan = { group_id, terminals, edges, grow_order }
GlobalTopologyPlan = { groups: GroupTopologyPlan[] }

Task = {
  task_id, group_id, edge_id, # u/v/radius on TopologyEdge via find_edge
  status, path                # [T] → Ready + empty; multi step() fills
}

ConnectedSet map: TerminalId | component → cells / points

# --- GroupSession ---
GroupSession : MultiQueryPlanner
  reset(plan: GroupTopologyPlan, grid: CompositeGrid&, low: LowLevelPlanner,
        heuristic, open, actions) -> void
  # builds Task[] from plan.ordered_tasks(); grid read-only
  step()   # one Task: low.search → task.set_result(...); merge ConnectedSets
  last_path() / tasks() /* by value */ / search() / status() / expansions()

# --- PrioritySequencer ---
PrioritySequencer : MultiQueryPlanner
  reset(plans, ctx, …) | reset(plans, order, ctx, …)  # order optional → ordered_tasks()
        low, heuristic, open, actions) -> void
  step()   # one Task in order; update ConnectedSet vs obstacles; set_result; commit
  last_path() / tasks() / search() / status() / expansions()
```

Usage sketch (same shape as A\*):

```text
GroupSession session;
session.reset(plan, ctx, low, h, open, actions);
session.search();                 # or while Running: session.step()
if (session.status() == Success) {
  auto all = session.tasks();  # copy; planner may reset again
  auto last = session.last_path();
}
```

### State transfer (important)

| After a successful path | `GroupSession` | `PrioritySequencer` |
|-------------------------|----------------|---------------------|
| Same-group path | → ConnectedSet (next starts/goals) | → ConnectedSet |
| Other-group path | Already in `ctx` obstacles; **unchanged** during session | → **obstacles** for later lower-priority tasks |

Cross-group Task order does **not** matter inside a lone `GroupSession` (CBS-style).  
It **does** matter for `PrioritySequencer` (PBS-style).

---

## Catalog

| Piece | Priority | Notes |
|-------|----------|-------|
| `MultiQueryPlanner` base | P0 | Align with `LowLevelPlanner` control API |
| `GroupSession` | P1 | SetPlanner-like; ★ for CBS / single group |
| `PrioritySequencer` | P1–P2 | PBS sequential; owns cross-group order application |
| Task view from `TopologyEdge` | P0 | Export helper; priority table separate |
| README + unit tests (toy grids) | P1 | Steppable session like `test_astar` |

---

## Layout (target)

```text
include/planner/multi/
  ROADMAP.md
  MultiQueryPlanner.h
  GroupSession.h
  PrioritySequencer.h

src/planner/multi/
  GroupSession.cpp
  PrioritySequencer.cpp
```
