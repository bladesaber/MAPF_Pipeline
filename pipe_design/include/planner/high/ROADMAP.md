# planner/high — Group-level conflict resolution

Detail companion to [`../../ROADMAP.md`](../../ROADMAP.md) §4.5 / §8.

Resolvers that detect **group–group** geometric conflicts and fix them by mutating
constraints / priorities / neighborhoods, then **re-invoke** `planner/multi/` drivers.
They do **not** own cell search or TopologyPlan construction.

Legacy analogy: classical CBS / PBS / LNS outer loops (logic only; occupancy is permanent).

---

## Why high/

`multi/` turns a frozen TopologyPlan into paths for one leaf (one constraint set or one
priority order). Residual **cross-group** conflicts need a search over those leaves:

| Method | State mutated in `[F]` | Re-invoke |
|--------|------------------------|-----------|
| CBS | `ObstacleChain` (sphere/capsule constraints) | `GroupSession` for affected group(s) |
| PBS | priority / Task order | `PrioritySequencer` |
| LNS / ALNS | destroy Task subset; optional retopologize | `[T]` and/or `GroupSession` / sequencer |

Without this layer, multi/ can only produce a single schedule; it cannot branch or repair.

---

## Role

| Piece | Scope | Typical callee |
|-------|--------|----------------|
| ★ `HighLevelPlanner` | Steppable base (status / expansions / hooks) | tools, window `[W]`, benches |
| ★ CBS (+ disjoint split) | Constraint tree; geometric `ObstacleChain` | `GroupSession` |
| ECBS | Bounded-suboptimal CBS | `GroupSession` |
| ★ PBS | Priority DAG / order search | `PrioritySequencer` |
| ★ LNS / ALNS | Destroy–repair neighborhoods | multi/ ± `[T]` |
| ICTS / ILP-SAT | Later / small instances | — |

**Contract line:** `[T]` builds plans → **multi/** routes a leaf → **high/** searches over leaves → outer `[W]` grows `active_set`.

---

## API alignment with low-level

**Agree: mirror the low-level steppable contract** (same spirit as `MultiQueryPlanner`).

| Low (`LowLevelPlanner`) | Multi (`MultiQueryPlanner`) | High (`HighLevelPlanner`) |
|-------------------------|-----------------------------|---------------------------|
| `SearchStatus` | same | same |
| `reset(...)` | `reset(...)` | derived `reset(...)` |
| `step()` — one cell expand | `step_task()` — one Task | `step()` — one conflict-fix iteration |
| `search(max_expansions)` | same | same |
| `status()` / `expansions()` | same | same |
| `path()` | `last_path()` / `tasks()` | `solution()` → `vector<Task>` |
| before/after step hooks | — | same as low |

Do **not** inherit `LowLevelPlanner` or `MultiQueryPlanner`: `reset` arity and result shape differ.
Prefer a **parallel base** with the same control surface:

```text
class HighLevelPlanner {   # parallel to LowLevelPlanner
  virtual SearchStatus step() = 0;
  virtual vector<Task> solution() const = 0;   # incumbent, by value
  virtual SearchStatus search(max_expansions = kUnlimited);
  SearchStatus status() const;
  std::size_t expansions() const;
  # optional before/after step hooks (debug / timing)
protected:
  SearchStatus status_;
  std::size_t expansions_;   # high-level iterations (not low cell expands)
};
```

`ROADMAP.md` §3’s `HighLevelPlanner::solve(...)` is shorthand for `reset` + `search`.

### What `step()` means

One **high-level iteration**, not one cell and not necessarily one Task:

| Algorithm | Typical `step()` |
|-----------|------------------|
| CBS | Pop CT node → `[V]` → if conflict, branch (extend chains) + `GroupSession` children; else Success |
| PBS | Expand priority / order → `PrioritySequencer` → `[V]` → update precedence on conflict |
| LNS | Destroy subset → optional `[T]` → repair via multi/ → accept/reject |

- If a leaf is conflict-free → high `Success`; `solution()` holds that leaf’s Tasks.
- If open set / neighborhood exhausted with no feasible leaf → `Failure`.
- While searching → `Running`; bump `expansions_` once per high-level iteration.

Fine-grained “one multi `step_task` inside a leaf” stays inside the multi driver — not exposed on the high base.

### Ownership

```text
Caller owns:  TopologyBuilder (holds GlobalTopologyPlan), CompositeGrid / static field,
              LowLevelPlanner, OpenSet, Heuristic, ActionModel, multi drivers (or high owns them)
High owns:    conflict-tree / priority state / LNS neighborhood, incumbent Tasks,
              status_ / expansions_
```

Topology is usually **frozen** for one high-level episode (CBS / PBS). LNS may call
`TopologyBuilder::replan()` or rebuild via `build()`.

---

## Interface (target)

```text
GlobalTopologyPlan = { groups: GroupTopologyPlan[] }
Task               = { task_id, group_id, edge_id, status, path }

# --- base ---
HighLevelPlanner
  step() / search() / status() / expansions() / solution()

# --- CBS ---
CBS : HighLevelPlanner
  reset(plans, grid: CompositeGrid, low, heuristic, open, actions,
        session: GroupSession&, analyzer: ConflictAnalyzer&) -> void
  step()  # pop CT → [V] → branch constrain() + GroupSession for one group
  constrain(chain, conflict, group)  # virtual; default = other witness sphere
  solution()

# --- PBS (planned) ---
PBS : HighLevelPlanner
  reset(plans, grid, low, …, sequencer: PrioritySequencer&) -> void
  step()  # priority expand + PrioritySequencer replan
  solution()

# --- LNS (planned) ---
LNS : HighLevelPlanner
  reset(... incumbent ...) -> void
  step()  # destroy–repair one neighborhood
  solution()
```

Usage sketch (same shape as A\* / GroupSession):

```text
CBS cbs;
cbs.reset(plans, grid, low, h, open, actions, session);
cbs.search();
if (cbs.status() == Success) {
  auto all = cbs.solution();  # copy; planner may reset again
}
```

---

## Catalog

| Piece | Priority | Notes |
|-------|----------|-------|
| `HighLevelPlanner` base | P0 | Align with `LowLevelPlanner` control API |
| ★ CBS + geometric constraints | P2 | Constraint = other witness sphere on `ObstacleChain` |
| ★ Disjoint splitting | P2 | Override `CBS::constrain` |
| ECBS | P2 | Bounded suboptimal |
| ★ PBS | P2 | Replan via `PrioritySequencer` |
| ★ LNS / ALNS | P3 | Scale-out when CBS explodes |
| ICTS | P3 | Porting cost high for permanent occupancy |
| ILP / SAT on candidates | P4 | Small instances only |
| README + unit tests (two-group conflict) | P2 | `test_cbs` |

---

## Layout (target)

```text
include/planner/high/
  ROADMAP.md
  HighLevelPlanner.h
  CBS.h
  PBS.h            # planned
  LNS.h            # planned

src/planner/high/
  CBS.cpp
  PBS.cpp          # planned
  LNS.cpp          # planned
```
